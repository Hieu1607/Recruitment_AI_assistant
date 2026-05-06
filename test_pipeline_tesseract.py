"""
End-to-end test: pdfs/Clear (first 10) → hieuailearning/resume-ocr-tesseract → Groq LLM → validate JSON
API flow: POST /ocr/submit → poll GET /ocr/status/{job_id} → DELETE /ocr/job/{job_id}
Run: python test_pipeline_tesseract.py
Requires: pip install requests groq python-dotenv
"""

import json
import os
import time

import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── Config ────────────────────────────────────────────────────────────────────
PDF_FOLDER = "pdfs/Unclear"
MAX_FILES = 10
GROQ_MODEL = "openai/gpt-oss-120b"
OCR_BASE = "https://hieuailearning-resume-ocr-tesseract.hf.space"
POLL_INTERVAL = 2  # seconds between status checks
POLL_TIMEOUT = 300  # max seconds to wait for a job
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


# ── Copied prompt (kept intact from build_prompts.py) ─────────────────────────
def build_cv_parsing_prompt(cv_text: str, max_chars: int = 24000) -> str:
    text = cv_text.strip()
    if len(text) > max_chars:
        head = text[: int(max_chars * 0.7)]
        tail = text[-int(max_chars * 0.3) :]
        text = f"{head}\n\n...[TRUNCATED]...\n\n{tail}"

    return f"""
	Analyze the CV text and return ONLY one valid JSON object (no markdown, no explanation).

	Required schema:
	{{
	"name": string|null,
	"phone": string|null,
	"email": string|null,
	"location": string|null,
	"contact": string|null,
	"current_job_title": string|null,
	"educated": boolean,
	"ever_studied_abroad": boolean,
	"major": string|null,
	"cpa": string|null,
	"education": string|null,
	"experience": string|null,
	"experience_years": number|null,
	"skills": string|null,
	"languages": string|null,
	"projects": string|null,
	"summary": string|null,
	"achievements": string|null,
	"publications": string|null,
	"certifications": string|null,
	"references": string|null,
	"other": string|null
	}}

	Rules:
	- Use null when unknown.
	- Keep extracted text concise and faithful to CV.
	- experience_years must be numeric (e.g., 3 or 4.5) or null.

	CV text:
	{text}
	""".strip()


# ── Copied JSON extractor (from resume_service.py) ────────────────────────────
def extract_json_object(raw_text: str) -> dict:
    content = (raw_text or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1]).strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("LLM did not return a valid JSON object")
    return json.loads(content[start : end + 1])


# ── Validation ────────────────────────────────────────────────────────────────
REQUIRED_KEYS = {
    "name",
    "phone",
    "email",
    "location",
    "contact",
    "current_job_title",
    "educated",
    "ever_studied_abroad",
    "major",
    "cpa",
    "education",
    "experience",
    "experience_years",
    "skills",
    "languages",
    "projects",
    "summary",
    "achievements",
    "publications",
    "certifications",
    "references",
    "other",
}
HIGH_PRIORITY = {
    "name",
    "experience",
    "skills",
    "current_job_title",
    "experience_years",
}


def validate(parsed: dict) -> dict:
    missing = sorted(REQUIRED_KEYS - parsed.keys())
    type_errs = []
    for f in ("educated", "ever_studied_abroad"):
        v = parsed.get(f)
        if v is not None and not isinstance(v, bool):
            type_errs.append(f"{f}={v!r} not bool")
    exp = parsed.get("experience_years")
    if exp is not None and not isinstance(exp, (int, float)):
        type_errs.append(f"experience_years={exp!r} not number")
    null_hp = sorted(f for f in HIGH_PRIORITY if not parsed.get(f))
    return {
        "missing_keys": missing,
        "type_errors": type_errs,
        "null_high_priority": null_hp,
        "ok": not missing and not type_errs and not null_hp,
    }


# ── OCR via hieuailearning/resume-ocr-tesseract ───────────────────────────────
def ocr_pdf(pdf_path: str) -> tuple:
    """Returns (text, submit_s, wait_s). Sends PDF directly — no image conversion."""
    with open(pdf_path, "rb") as f:
        t_submit = time.perf_counter()
        resp = requests.post(
            f"{OCR_BASE}/ocr/submit",
            files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
            timeout=60,
        )
        resp.raise_for_status()
        submit_s = round(time.perf_counter() - t_submit, 2)

    job_id = resp.json()["job_id"]

    # Poll until done
    t_wait = time.perf_counter()
    deadline = t_wait + POLL_TIMEOUT
    while time.perf_counter() < deadline:
        status_resp = requests.get(f"{OCR_BASE}/ocr/status/{job_id}", timeout=10)
        status_resp.raise_for_status()
        data = status_resp.json()
        if data["status"] == "done":
            wait_s = round(time.perf_counter() - t_wait, 2)
            text = data["text"]
            requests.delete(f"{OCR_BASE}/ocr/job/{job_id}", timeout=10)
            return text, submit_s, wait_s
        if data["status"] == "error":
            raise RuntimeError(f"OCR job failed: {data.get('error')}")
        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"OCR job {job_id} timed out after {POLL_TIMEOUT}s")


# ── Main ──────────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

pdf_files = sorted(f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf"))[
    :MAX_FILES
]
print(
    f"Testing {len(pdf_files)} PDFs  |  OCR: {OCR_BASE}  |  LLM: {GROQ_MODEL}\n{'='*70}"
)

summary = []
total_start = time.perf_counter()

for i, pdf_name in enumerate(pdf_files, 1):
    print(f"\n[{i}/{len(pdf_files)}] {pdf_name}")
    row = {
        "file": pdf_name,
        "submit_s": None,
        "wait_s": None,
        "llm_s": None,
        "valid": False,
        "error": None,
    }

    # Step 1: OCR
    try:
        ocr_text, submit_s, wait_s = ocr_pdf(os.path.join(PDF_FOLDER, pdf_name))
        row["submit_s"] = submit_s
        row["wait_s"] = wait_s
        print(f"  Submit: {submit_s}s  |  Wait: {wait_s}s  |  {len(ocr_text)} chars")
    except Exception as e:
        row["error"] = f"OCR failed: {e}"
        print(f"  OCR   : FAILED — {e}")
        summary.append(row)
        continue

    # Step 2: LLM parse
    try:
        prompt = build_cv_parsing_prompt(ocr_text)
        t0 = time.perf_counter()
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4096,
        )
        llm_s = round(time.perf_counter() - t0, 2)
        row["llm_s"] = llm_s
        raw = (completion.choices[0].message.content or "").strip()
        print(f"  LLM   : {llm_s}s")
    except Exception as e:
        row["error"] = f"LLM failed: {e}"
        print(f"  LLM   : FAILED — {e}")
        summary.append(row)
        continue

    # Step 3: Parse + Validate
    try:
        parsed = extract_json_object(raw)
        report = validate(parsed)
        row["valid"] = report["ok"]
        status = "✓ OK" if report["ok"] else "✗ ISSUES"
        print(f"  Valid : {status}")
        if report["missing_keys"]:
            print(f"    missing keys      : {report['missing_keys']}")
        if report["type_errors"]:
            print(f"    type errors       : {report['type_errors']}")
        if report["null_high_priority"]:
            print(f"    null high-priority: {report['null_high_priority']}")
        print(
            f"  name={parsed.get('name')!r}  exp_years={parsed.get('experience_years')!r}  skills={str(parsed.get('skills',''))[:60]!r}"
        )
    except Exception as e:
        row["error"] = f"Parse failed: {e}"
        print(f"  Parse : FAILED — {e}")
        print(f"  --- RAW LLM OUTPUT ---\n{raw}\n  --- END RAW ---")

    summary.append(row)

# ── Summary table ─────────────────────────────────────────────────────────────
total_s = round(time.perf_counter() - total_start, 2)
passed = sum(1 for r in summary if r["valid"])
print(f"\n{'='*70}")
print(f"RESULTS  {passed}/{len(summary)} passed  |  total {total_s}s\n")
print(f"{'File':<44} {'Submit':>8} {'Wait':>8} {'LLM':>6} {'OK':>4}")
print("-" * 74)
for r in summary:
    submit = f"{r['submit_s']}s" if r["submit_s"] is not None else "ERR"
    wait = f"{r['wait_s']}s" if r["wait_s"] is not None else "ERR"
    llm = f"{r['llm_s']}s" if r["llm_s"] is not None else "ERR"
    ok = "✓" if r["valid"] else ("!" if not r["error"] else "✗")
    print(f"{r['file'][:43]:<44} {submit:>8} {wait:>8} {llm:>6} {ok:>4}")
