import io
import os
import time

import fitz  # PyMuPDF
import requests

url = "https://hieuailearning-resume-ocr-api.hf.space/ocr"
PDF_FOLDER = "pdfs/Clear"

total_start = time.perf_counter()

for pdf_name in os.listdir(PDF_FOLDER):
    if not pdf_name.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_FOLDER, pdf_name)
    pdf_start = time.perf_counter()
    print(f"\n=== Processing: {pdf_name} ===")

    doc = fitz.open(pdf_path)
    full_text = []

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")

        page_start = time.perf_counter()
        response = requests.post(
            url,
            files={
                "file": (
                    f"{pdf_name}_page{page_num + 1}.png",
                    io.BytesIO(img_bytes),
                    "image/png",
                )
            },
            timeout=60,
        )
        page_elapsed = time.perf_counter() - page_start
        response.raise_for_status()
        page_text = response.json()["text"]
        print(f"  Page {page_num + 1} ({page_elapsed:.2f}s): {page_text}...")
        full_text.append(page_text)

    pdf_elapsed = time.perf_counter() - pdf_start
    print(f"  >> PDF total: {pdf_elapsed:.2f}s")
    doc.close()

    combined_text = "\n".join(full_text)
    # Đưa thẳng vào LLM
    llm_prompt = f"Phân tích resume sau:\n\n{combined_text}"
    break

print(f"\n=== Total elapsed: {time.perf_counter() - total_start:.2f}s ===")
