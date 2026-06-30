from typing import Literal, Optional

from pydantic import BaseModel, conint, validator

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


class HealthResponse(BaseModel):
    status: Literal["healthy"]


class ReadinessResponse(BaseModel):
    status: Literal["ready"]
    documents_indexed: int
    embedding_model: str
    embedding_dimension: int
    faiss_vectors: int


class ReadinessFailureResponse(BaseModel):
    status: Literal["not_ready"]
    reason: str
    checks: dict[str, bool]


class VersionResponse(BaseModel):
    application: str
    version: str
    python_version: str
    fastapi_version: str
    startup_time: str
    environment: str
