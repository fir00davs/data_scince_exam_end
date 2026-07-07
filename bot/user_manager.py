"""
User session management for Telegram bot
"""
import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

import config


class UserManager:
    """Manage user sessions and ratings"""
    
    def __init__(self, data_file=None):
        if data_file is None:
            data_file = config.MODELS_DIR / "user_sessions.json"
        
        self.data_file = Path(data_file)
        self.users = self._load_users()
    
    def _load_users(self) -> Dict:
        """Load user data from JSON"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load users: {e}")
                return {}
        return {}
    
    def _save_users(self):
        """Save user data to JSON"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.users, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save users: {e}")
    
    def get_user(self, user_id: int) -> Dict:
        """Get user data"""
        user_id_str = str(user_id)
        if user_id_str not in self.users:
            self.users[user_id_str] = {
                'user_id': user_id,
                'ratings': {},
                'created_at': str(Path.cwd())
            }
            self._save_users()
        return self.users[user_id_str]
    
    def add_rating(self, user_id: int, movie_id: int, rating: int) -> bool:
        """Add movie rating from user"""
        if rating < 1 or rating > 5:
            logger.warning(f"Invalid rating: {rating}")
            return False
        
        user = self.get_user(user_id)
        user['ratings'][str(movie_id)] = rating
        self._save_users()
        logger.info(f"User {user_id} rated movie {movie_id}: {rating}/5")
        return True
    
    def get_ratings(self, user_id: int) -> Dict[int, int]:
        """Get all ratings from user"""
        user = self.get_user(user_id)
        return {int(k): v for k, v in user['ratings'].items()}
    
    def get_rating_count(self, user_id: int) -> int:
        """Get number of ratings"""
        user = self.get_user(user_id)
        return len(user['ratings'])
    
    def get_average_rating(self, user_id: int) -> float:
        """Get average rating"""
        ratings = self.get_ratings(user_id)
        if not ratings:
            return 0.0
        return sum(ratings.values()) / len(ratings)
    
    def has_rated_movie(self, user_id: int, movie_id: int) -> bool:
        """Check if user has rated a movie"""
        ratings = self.get_ratings(user_id)
        return movie_id in ratings
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get user statistics"""
        rating_count = self.get_rating_count(user_id)
        avg_rating = self.get_average_rating(user_id)
        
        return {
            'rating_count': rating_count,
            'average_rating': round(avg_rating, 2),
            'can_get_recommendations': rating_count >= config.MIN_RATINGS_FOR_CF
        }
    
    def clear_user_ratings(self, user_id: int) -> bool:
        """Clear all ratings for a user"""
        user = self.get_user(user_id)
        user['ratings'] = {}
        self._save_users()
        logger.info(f"Cleared ratings for user {user_id}")
        return True


if __name__ == "__main__":
    # Test UserManager
    manager = UserManager()
    
    # Add some test ratings
    manager.add_rating(123, 1, 5)
    manager.add_rating(123, 2, 4)
    manager.add_rating(123, 3, 3)
    
    # Get stats
    stats = manager.get_user_stats(123)
    print(f"User 123 stats: {stats}")
    
    # Get ratings
    ratings = manager.get_ratings(123)
    print(f"User 123 ratings: {ratings}")
