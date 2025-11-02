"""
Services package for Moodify-AI ML Service.

This package provides core services:
- cache_service: Redis caching
- lyrics_service: Lyrics fetching and sentiment analysis
- model_service: ML model inference and optimization
- spotify_service: Spotify API interactions
"""

from . import cache_service
from . import lyrics_service
from . import model_service
from . import spotify_service

__all__ = [
    'cache_service',
    'lyrics_service',
    'model_service',
    'spotify_service'
]