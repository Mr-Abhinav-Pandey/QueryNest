from pydantic import BaseModel, conint, validator
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    top_k: conint(ge=1, le=100) = 5

    @validator("query")
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("query must not be empty or whitespace only")
        return value

class SearchResult(BaseModel):
    filename: str
    snippet: str
    score: float
    url: Optional[str]
