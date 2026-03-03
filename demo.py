import fitz

name = "Khang.pdf"

doc = fitz.open(name)

all_text = ""

for page in doc:
    text = page.get_text()
    print(text)
    all_text += text + "\n"

doc.close()
with open(name.replace(".pdf", ".txt"), "w") as f:
    f.write(all_text)
