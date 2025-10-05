from pydantic import BaseModel, Field

class JDinput(BaseModel):
    jd: str = Field(..., description="Job description text")
