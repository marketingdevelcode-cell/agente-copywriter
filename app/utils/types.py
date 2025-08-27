from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class FilterQuery(BaseModel):
    persona: Optional[List[str]] = None
    funnel_stage: Optional[List[str]] = None
    objective: Optional[List[str]] = None
    tone: Optional[List[str]] = None
    emotions: Optional[List[str]] = None
    triggers: Optional[List[str]] = None
    format: Optional[List[str]] = None
    sector: Optional[List[str]] = None
    language: Optional[str] = "pt-BR"

class SearchIn(BaseModel):
    query: str
    filters: Optional[FilterQuery] = None
    top_k: Optional[int] = None

class Chunk(BaseModel):
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None

class SearchOut(BaseModel):
    query: str
    hits: List[Chunk]

class TemplateIn(BaseModel):
    format: str
    persona: Optional[str] = None
    funnel_stage: Optional[str] = None
