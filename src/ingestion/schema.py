from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class Document(BaseModel):
    """One unit of unstructured text, normalized from any source."""
    source: str                                # e.g. "newsapi", "kaggle_tweets"
    source_type: Literal["news", "social"]
    timestamp: datetime
    text: str
    ticker: Optional[str] = None               # None if no company detected

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = " ".join(v.split())                # collapse whitespace/newlines
        if not v:
            raise ValueError("text must not be empty")
        return v
