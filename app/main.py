import os
# Limit threads to reduce memory overhead
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
from fastapi import FastAPI
import uvicorn
import torch
import gc
from contextlib import asynccontextmanager

# Limit PyTorch memory footprint
torch.set_num_threads(1)

from app.services.recommendation import RecommendationService
from app.api.endpoints import router

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load FAISS index once
    logger.info("Loading Recommendation System...")
    service = RecommendationService()
    service.load_data()
    gc.collect()
    service.initialize_model()
    gc.collect()
    # We load from disk by default
    service.load_index(force_rebuild=False)
    gc.collect()
    
    app.state.reco_service = service
    logger.info("System ready!")
    yield
    # Shutdown logic if any
    logger.info("Shutting down...")

app = FastAPI(title="Book Recommendation API", lifespan=lifespan)

app.include_router(router)

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
