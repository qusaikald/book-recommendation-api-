from fastapi import APIRouter, HTTPException, Request
from typing import List
from app.models.book import BookRecommendation, RecommendRequest, SearchRequest, UserReadAction
from app.services.recommendation import RecommendationService

router = APIRouter()

@router.get("/")
def read_root():
    """Root endpoint to welcome users and point to the docs."""
    return {
        "message": "Welcome to the Book Recommendation API!",
        "documentation": "/docs",
        "status": "online"
    }

@router.get("/health")
def health_check():
    """Health check endpoint to ensure the API is running."""
    return {"status": "healthy", "service": "Book Recommendation API"}

@router.post("/recommend", response_model=List[BookRecommendation])
def get_recommendations(req: RecommendRequest, request: Request):
    """Get recommendations based on an existing book's title."""
    service: RecommendationService = request.app.state.reco_service
    try:
        results = service.search_similar_books(title=req.title, top_k=req.top_k)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/search", response_model=List[BookRecommendation])
def search_books(req: SearchRequest, request: Request):
    """Semantic search for books using any arbitrary text query."""
    service: RecommendationService = request.app.state.reco_service
    try:
        results = service.search_by_text(query=req.query, top_k=req.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/users/{user_id}/read")
def add_read_book(user_id: str, action: UserReadAction, request: Request):
    """Add a book to the user's read list."""
    service: RecommendationService = request.app.state.reco_service
    try:
        read_list = service.add_to_read_list(user_id, action.title)
        return {"user_id": user_id, "read_list": read_list}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/users/{user_id}/recommendations", response_model=List[BookRecommendation])
def get_user_recos(user_id: str, request: Request, top_k: int = 4):
    """Get personalized, blended recommendations based on the user's read history."""
    service: RecommendationService = request.app.state.reco_service
    try:
        results = service.get_user_recommendations(user_id, top_k=top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
