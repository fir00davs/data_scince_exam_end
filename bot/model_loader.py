"""
Model loader for Telegram bot
"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

import config
from src.models import HybridRecommender, CollaborativeFilter, ContentBasedFilter
from bot.user_manager import UserManager


class ModelLoader:
    """Load and manage ML models for the bot"""
    
    def __init__(self):
        self.model = None
        self.movies = None
        self.user_manager = UserManager()
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        logger.info("Loading trained models...")
        
        try:
            self.model = HybridRecommender.load(
                config.COLLAB_FILTER_MODEL,
                config.CONTENT_FILTER_MODEL
            )
            logger.info("✓ Loaded hybrid recommendation model")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            logger.warning("Models not found. Please run training first!")
            return False
        
        try:
            import pandas as pd
            self.movies = pd.read_csv(config.MOVIES_DB)
            logger.info(f"✓ Loaded {len(self.movies)} movies from database")
        except Exception as e:
            logger.error(f"Failed to load movies database: {e}")
            return False
        
        return True
    
    def is_ready(self) -> bool:
        """Check if models are ready"""
        return self.model is not None and self.movies is not None
    
    def search_movies(self, query: str, limit: int = 10) -> List[Dict]:
        """Search movies by title
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of matching movies
        """
        if self.movies is None:
            return []
        
        query_lower = query.lower()
        results = self.movies[
            self.movies['title'].str.lower().str.contains(query_lower, na=False)
        ].head(limit)
        
        return [
            {
                'movie_id': int(row['movie_id']),
                'title': row['title'],
                'avg_rating': round(row['avg_rating'], 2),
                'num_ratings': int(row['num_ratings'])
            }
            for _, row in results.iterrows()
        ]
    
    def get_movie_info(self, movie_id: int) -> Optional[Dict]:
        """Get detailed info about a movie"""
        if self.movies is None:
            return None
        
        movie = self.movies[self.movies['movie_id'] == movie_id]
        if movie.empty:
            return None
        
        row = movie.iloc[0]
        
        # Get genres
        genre_cols = ['action', 'adventure', 'animation', 'children', 'comedy', 
                      'crime', 'documentary', 'drama', 'fantasy', 'horror', 
                      'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']
        genres = [col.replace('_', ' ').title() for col in genre_cols if row[col] == 1]
        
        return {
            'movie_id': int(row['movie_id']),
            'title': row['title'],
            'release_date': str(row['release_date']),
            'genres': genres,
            'avg_rating': round(row['avg_rating'], 2),
            'num_ratings': int(row['num_ratings'])
        }
    
    def get_recommendations(self, user_id: int, n: int = 5) -> List[Dict]:
        """Get movie recommendations for a user
        
        Args:
            user_id: User ID
            n: Number of recommendations
            
        Returns:
            List of recommended movies
        """
        if not self.is_ready():
            logger.warning("Models not loaded")
            return []
        
        try:
            # Get user ratings from session manager
            ratings = self.user_manager.get_ratings(user_id)
            
            if len(ratings) < config.MIN_RATINGS_FOR_CF:
                logger.info(f"User {user_id} has insufficient ratings")
                return self._get_popular_movies(n)
            
            # Get recommendations from model
            recommendations = self.model.get_recommendations(user_id, self.movies, n=n)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def _get_popular_movies(self, n: int = 5) -> List[Dict]:
        """Get popular movies (for cold start)"""
        if self.movies is None:
            return []
        
        # Get top rated movies
        popular = self.movies.nlargest(n, 'avg_rating')
        
        return [
            {
                'movie_id': int(row['movie_id']),
                'title': row['title'],
                'score': 1.0,
                'avg_rating': round(row['avg_rating'], 2),
                'num_ratings': int(row['num_ratings'])
            }
            for _, row in popular.iterrows()
        ]
    
    def add_user_rating(self, user_id: int, movie_id: int, rating: int) -> bool:
        """Add a rating from the user"""
        if rating < 1 or rating > 5:
            logger.warning(f"Invalid rating: {rating}")
            return False
        
        return self.user_manager.add_rating(user_id, movie_id, rating)
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get user statistics"""
        return self.user_manager.get_user_stats(user_id)
    
    def format_recommendation(self, rec: Dict) -> str:
        """Format recommendation for display"""
        return (
            f"🎬 {rec['title']}\n"
            f"⭐ Rating: {rec['avg_rating']}/5 ({int(rec['num_ratings'])} votes)\n"
            f"💯 Recommendation Score: {rec['score']:.2f}"
        )
    
    def format_movie_info(self, info: Dict) -> str:
        """Format movie info for display"""
        genres = ', '.join(info['genres']) if info['genres'] else 'N/A'
        return (
            f"🎬 {info['title']}\n"
            f"📅 Released: {info['release_date']}\n"
            f"🎭 Genres: {genres}\n"
            f"⭐ Rating: {info['avg_rating']}/5 ({int(info['num_ratings'])} votes)"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    loader = ModelLoader()
    
    if loader.is_ready():
        # Test search
        results = loader.search_movies("Star Wars", limit=5)
        print("Search results for 'Star Wars':")
        for movie in results:
            print(f"  - {movie['title']} (ID: {movie['movie_id']})")
        
        # Test popular movies
        popular = loader._get_popular_movies(5)
        print("\nTop popular movies:")
        for movie in popular:
            print(f"  - {movie['title']}")
    else:
        print("Models not available. Please run training first!")
