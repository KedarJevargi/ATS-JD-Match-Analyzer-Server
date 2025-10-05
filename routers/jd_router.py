from fastapi import FastAPI, APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from utils import jd_keyword_extractor, sendGemini
import utils.sendGemini as sendGemini



router = APIRouter(prefix="/jds", tags=["JDs"])

# --- Pydantic model for JD input ---
class JDInput(BaseModel):
    jd: str = Field(..., description="Job description text")

# --- Single endpoint: Extract + Structure ---
@router.post("/parse_text", status_code=status.HTTP_201_CREATED)
async def parse_jd_text(jd_input: JDInput):
    jd_text = jd_input.jd.strip()
    if not jd_text:
        raise HTTPException(status_code=422, detail="Job description cannot be empty")

    # Step 1: Extract keywords (await because it's async)
    raw_keywords = await jd_keyword_extractor.extract_keywords_simple(jd_text)

    # Step 2: Structure keywords using Gemini
    structured_jd = sendGemini.parse_with_gemini(
        sendGemini.system_prompt, raw_keywords
    )

    return structured_jd



