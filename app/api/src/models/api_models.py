from pydantic import BaseModel
from typing import List

class EmailRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    label: str
    confidence: float
    suggested_reply: str
