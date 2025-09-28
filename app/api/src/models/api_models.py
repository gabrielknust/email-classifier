from pydantic import BaseModel
from typing import List

class ClassificationRequest(BaseModel):
    text: str
    labels: List[str]

class ClassificationResponse(BaseModel):
    sequence: str
    labels: List[str]
    scores: List[float]