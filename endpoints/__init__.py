"""
Endpoints Package
-----------------
This package contains all API route handlers for the ML microservice.

Modules:
- mood_router: Mood prediction and analysis endpoints
- optimize_router: Playlist flow optimization endpoints  
- train_router: Model training, feedback, and personalization endpoints
"""

from .mood_router import router as mood_router
from .optimize_router import router as optimize_router
from .train_router import router as train_router
from .analytics_router import router as analytics_router
from .generate_router import router as generate_router

__all__ = [
    "mood_router",
    "optimize_router", 
    "train_router",
    "analytics_router",
    "generate_router"
]