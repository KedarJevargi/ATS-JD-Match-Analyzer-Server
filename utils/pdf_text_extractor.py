import pdfplumber
from fastapi import UploadFile
from io import BytesIO

def extract_text_from_pdf(pdf: UploadFile) -> str:
    """
    Extracts text from ALL pages of an uploaded PDF file.
    Works by reading the file into memory first.
    Returns an empty string if extraction fails.
    """
    full_text = ""
    try:
        pdf_bytes = pdf.file.read()
        pdf_stream = BytesIO(pdf_bytes)
        pdf.file.seek(0)  # Reset pointer after reading

        with pdfplumber.open(pdf_stream) as pdf_doc:
            for page in pdf_doc.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        return full_text
    except Exception as e:
        return ""
    finally:
        pdf.file.seek(0)  