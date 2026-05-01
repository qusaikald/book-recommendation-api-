from pydantic import BaseModel
from typing import Optional

class BookBase(BaseModel):
    title: str
    author: str
    category: str
    year: int
    rating: float
    description: str

class BookRecommendation(BookBase):
    similarity: float

class RecommendRequest(BaseModel):
    title: str
    top_k: Optional[int] = 5

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class UserReadAction(BaseModel):
    title: str

class UserProfile(BaseModel):
    user_id: str
    read_list: list[str]
