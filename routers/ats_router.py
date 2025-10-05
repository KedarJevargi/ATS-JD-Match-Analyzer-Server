import os
import json
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Form
from fastapi.responses import JSONResponse
from utils import jd_keyword_extractor, pdf_text_extractor, atsAanalyzer, sendGemini
from io import BytesIO
import google.generativeai as genai

# --- Router Setup ---
router = APIRouter(
    prefix="/ats",
    tags=["ATS"]
)

@router.post("/analyse", status_code=status.HTTP_201_CREATED)
async def analyse_resume(pdf: UploadFile = File(...), jd: str = Form(...)):
    """
    Analyzes a resume PDF against a job description.
    """
    
    # === STEP 1: Initial Data Processing ===
    try:
        # Extract text from PDF
        pdf_text = pdf_text_extractor.extract_text_from_pdf(pdf)
        if not pdf_text.strip():
            raise HTTPException(status_code=422, detail="No text could be extracted from the PDF")
        
        # Parse JD text
        jd_text = jd.strip()
        if not jd_text:
            raise HTTPException(status_code=422, detail="Job description cannot be empty")
        
        # Extract keywords from JD
        raw_keywords = await jd_keyword_extractor.extract_keywords_simple(jd_text)
        
        # Structure keywords using a helper function
        parsed_jd = sendGemini.parse_with_gemini(sendGemini.system_prompt, raw_keywords)
        
        # Reset file pointer and get bytes for analysis
        await pdf.seek(0)
        pdf_bytes = await pdf.read()
        pdf_stream = BytesIO(pdf_bytes)
        
        # Perform the initial structural and keyword analysis
        analysis_result_json = await atsAanalyzer.analyze_resume(pdf_stream, parsed_jd, False)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during initial processing: {str(e)}")


    # === STEP 2: Use the Original Prompt for Formatted String Output ===
    final_prompt_template = """
You are an expert software developer resume analyst with 10+ years of experience in technical recruiting and ATS optimization. 

Given the following resume analysis data:
{json_analysis}

And the resume text:
{resume_text}

Provide a comprehensive analysis in the following JSON structure:

{{
  "resumeAnalysis": {{
    "overallAssessment": "A comprehensive 3-4 sentence summary of the resume's current state, highlighting major strengths and critical weaknesses",
    "detailedBreakdown": {{
      "atsCompatibilityScore": {{
        "score": <number between 0-100>,
        "analysis": "Detailed explanation of why this score was given and what it means for ATS parsing",
        "recommendation": "Specific actionable steps to improve ATS compatibility"
      }},
      "keywordAnalysis": {{
        "matchPercentage": <percentage of keywords matched>,
        "missingKeywords": ["keyword1", "keyword2", "keyword3"],
        "analysis": "Explanation of keyword gaps and their impact",
        "recommendation": "Specific guidance on where and how to add missing keywords with examples"
      }},
      "impactAndQuantification": {{
        "quantifiedResults": <number of metrics found in resume>,
        "analysis": "Assessment of how well achievements are quantified",
        "recommendation": "Guidance on adding metrics using STAR/XYZ method with examples"
      }},
      "formattingAndReadability": {{
        "issues": ["issue1", "issue2", "issue3"],
        "analysis": "Explanation of formatting problems and their impact",
        "recommendation": "Step-by-step formatting improvements"
      }},
      "grammarAndSpelling": {{
        "errorCount": <estimated number of errors>,
        "analysis": "Assessment of language quality",
        "recommendation": "Proofreading and correction guidance"
      }},
      "structureAndContent": {{
        "skillsSection": "Analysis of the skills section organization and completeness",
        "projectsSection": "Analysis of projects section and suggestions",
        "recommendations": ["recommendation1", "recommendation2"]
      }}
    }},
    "summaryOfKeyRecommendations": {{
      "rectifyFormattingAndProofread": {{
        "priority": "Critical|High|Medium",
        "action": "Detailed action to take"
      }},
      "aggressivelyOptimizeKeywords": {{
        "priority": "Critical|High|Medium",
        "action": "Detailed action to take"
      }},
      "quantifyAllAchievements": {{
        "priority": "Critical|High|Medium",
        "action": "Detailed action to take"
      }},
      "restructureSkillsAndProjects": {{
        "priority": "Critical|High|Medium",
        "action": "Detailed action to take"
      }}
    }}
  }}
}}

Important guidelines:
1. Base ATS score on the structure score from the JSON analysis
2. Use the "keyword missing" array to populate missingKeywords
3. Calculate keyword match percentage from keywords matched vs total keywords
4. Be specific and actionable in all recommendations
5. Use proper priority levels (Critical for urgent issues, High for important, Medium for nice-to-have)
6. Provide concrete examples in recommendations
7. Return ONLY valid JSON, no additional text
"""
    
    full_prompt_with_context = final_prompt_template.format(
        resume_text=pdf_text,
        json_analysis=json.dumps(analysis_result_json, indent=2)
    )

    # === STEP 3: Call Gemini and Return the Raw String Response ===
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not set.")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')

        gemini_response = await model.generate_content_async(
            full_prompt_with_context,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"  # Force JSON output
            )
        )
        
        # --- Return the raw text directly from the model ---
        return {
            "result": analysis_result_json,
            "response": gemini_response.text 
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred with the Gemini API: {str(e)}")