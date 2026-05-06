"""Quick check: how many PDFs in /pdfs yield extractable text via PyMuPDF."""
import sys
from pathlib import Path

import fitz  # PyMuPDF

PDF_DIR = Path("/pdfs")


def check_pdf(path: Path) -> tuple[str, int, str]:
    try:
        with fitz.open(str(path)) as doc:
            text = "".join(page.get_text() for page in doc).strip()
        return ("ok", len(text), "")
    except Exception as exc:
        return ("error", 0, str(exc))


def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        sys.exit(1)

    ok = failed = empty = 0
    for pdf in pdfs:
        status, chars, err = check_pdf(pdf)
        if status == "error":
            failed += 1
            print(f"  ERROR   {pdf.name}: {err}")
        elif chars == 0:
            empty += 1
            print(f"  EMPTY   {pdf.name}  (image-based / no selectable text)")
        else:
            ok += 1
            print(f"  OK      {pdf.name}  ({chars} chars)")

    print(f"\n--- Summary: {len(pdfs)} files ---")
    print(f"  extractable : {ok}")
    print(f"  empty (scan): {empty}")
    print(f"  error       : {failed}")

    if empty or failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
