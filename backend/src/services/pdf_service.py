import fitz  # PyMuPDF

def extract_text_from_pdf(filepath: str) -> str:
    text = ""
    try:
        doc = fitz.open(filepath)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {filepath}: {e}")
        return ""
