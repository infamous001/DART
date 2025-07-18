import pdfplumber

def text_extract(pdf_path: str) -> str:

    extracted_text = ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"

    except Exception as e:
        print(f"Error while extracting text: {e}")

    return extracted_text

# pdf_path = "/Users/rohitjindal/Desktop/narratives-epfl/reinvoke/notes/1|decisionTrees.pdf"
# extracted_text = text_extract(pdf_path)
# print(extracted_text)
