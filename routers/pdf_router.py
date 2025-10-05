from fastapi import APIRouter, File, UploadFile, HTTPException, status
from utils import pdf_text_extractor

router = APIRouter(
    prefix="/pdfs",
    tags=["PDFs"]
)

@router.post("/extracttext", status_code=status.HTTP_201_CREATED)
def extract_text(pdf: UploadFile = File(...)):
    result = pdf_text_extractor.extract_text_from_pdf(pdf)
    if not result.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the PDF")
    return {"extracted_text": result}
