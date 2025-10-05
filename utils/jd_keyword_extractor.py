import spacy
import re
from collections import defaultdict

# You still need spaCy and its model. If you haven't installed it:
# pip install spacy
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

async def extract_keywords_simple(job_description: str) -> dict:
    doc = nlp(job_description)
    keywords = set()

    for chunk in doc.noun_chunks:
        keywords.add(chunk.text.lower())
    for token in doc:
        if token.pos_ == 'PROPN':
            keywords.add(token.text.lower())
    for ent in doc.ents:
        if ent.label_ in ['PRODUCT', 'ORG']:
            keywords.add(ent.text.lower())

    # Updated regex for experience
    experience_pattern = r'(\d+(?:\.\d+)?\s*\+?)\s*years?'
    matches = re.findall(experience_pattern, job_description.lower())
    experience = list({f"{m.strip()} years" for m in matches})

    # Filter generic keywords
    final_keywords = [
        keyword for keyword in keywords 
        if len(keyword) > 2 and keyword not in ['the job', 'experience', 'team', 'responsibilities']
    ]

    return {
        "keywords": sorted(list(final_keywords)),
        "experience_requirements": sorted(experience)
    }

