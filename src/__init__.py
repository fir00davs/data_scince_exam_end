"""
Movie Recommendation System
A hybrid recommendation system using collaborative and content-based filtering
"""

__version__ = "1.0.0"
__author__ = "Data Science Student"

from src.data_loader import MovieLensDataLoader
from src.models import CollaborativeFilter, ContentBasedFilter, HybridRecommender

__all__ = [
    'MovieLensDataLoader',
    'CollaborativeFilter',
    'ContentBasedFilter',
    'HybridRecommender'
]
