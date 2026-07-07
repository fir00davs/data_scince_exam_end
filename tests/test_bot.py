"""
Unit tests for Movie Recommendation Bot
"""
import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from bot.user_manager import UserManager
from bot.model_loader import ModelLoader
from bot.handlers import BotHandlers


class TestUserManager(unittest.TestCase):
    """Test user manager"""
    
    def setUp(self):
        """Create temporary user manager"""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_file.close()
        self.manager = UserManager(self.temp_file.name)
    
    def tearDown(self):
        """Clean up"""
        Path(self.temp_file.name).unlink()
    
    def test_add_rating(self):
        """Test adding ratings"""
        result = self.manager.add_rating(123, 1, 5)
        self.assertTrue(result)
        
        ratings = self.manager.get_ratings(123)
        self.assertEqual(ratings[1], 5)
    
    def test_invalid_rating(self):
        """Test invalid rating"""
        result = self.manager.add_rating(123, 1, 10)
        self.assertFalse(result)
        
        result = self.manager.add_rating(123, 1, 0)
        self.assertFalse(result)
    
    def test_get_stats(self):
        """Test user statistics"""
        self.manager.add_rating(123, 1, 5)
        self.manager.add_rating(123, 2, 4)
        self.manager.add_rating(123, 3, 3)
        
        stats = self.manager.get_user_stats(123)
        
        self.assertEqual(stats['rating_count'], 3)
        self.assertAlmostEqual(stats['average_rating'], 4.0, places=1)
    
    def test_has_rated_movie(self):
        """Test checking if movie is rated"""
        self.manager.add_rating(123, 1, 5)
        
        self.assertTrue(self.manager.has_rated_movie(123, 1))
        self.assertFalse(self.manager.has_rated_movie(123, 2))
    
    def test_clear_ratings(self):
        """Test clearing ratings"""
        self.manager.add_rating(123, 1, 5)
        self.manager.add_rating(123, 2, 4)
        
        self.manager.clear_user_ratings(123)
        ratings = self.manager.get_ratings(123)
        
        self.assertEqual(len(ratings), 0)


class TestBotHandlers(unittest.TestCase):
    """Test bot command handlers"""
    
    def setUp(self):
        """Setup handlers"""
        # Create mock model loader
        class MockModelLoader:
            def get_user_stats(self, user_id):
                return {
                    'rating_count': 2,
                    'average_rating': 4.5,
                    'can_get_recommendations': True
                }
            
            def search_movies(self, query, limit=10):
                return [
                    {
                        'movie_id': 1,
                        'title': 'Inception',
                        'avg_rating': 8.8,
                        'num_ratings': 1000
                    }
                ]
            
            def add_user_rating(self, user_id, movie_id, rating):
                return True
            
            def get_movie_info(self, movie_id):
                return {
                    'movie_id': movie_id,
                    'title': 'Inception',
                    'release_date': '2010-07-16',
                    'genres': ['Action', 'Sci-Fi'],
                    'avg_rating': 8.8,
                    'num_ratings': 1000
                }
            
            def get_recommendations(self, user_id, n):
                return [
                    {
                        'movie_id': 1,
                        'title': 'Inception',
                        'score': 0.9,
                        'avg_rating': 8.8,
                        'num_ratings': 1000
                    }
                ]
            
            def _get_popular_movies(self, n):
                return [
                    {
                        'movie_id': 1,
                        'title': 'Inception',
                        'score': 1.0,
                        'avg_rating': 8.8,
                        'num_ratings': 1000
                    }
                ]
            
            def is_ready(self):
                return True
        
        self.mock_loader = MockModelLoader()
        self.handlers = BotHandlers(self.mock_loader)
    
    def test_handle_start(self):
        """Test /start command"""
        response = self.handlers.handle_start(123)
        self.assertIn("Welcome", response)
        self.assertIn("Commands", response)
    
    def test_handle_help(self):
        """Test /help command"""
        response = self.handlers.handle_help()
        self.assertIn("search", response)
        self.assertIn("rate", response)
        self.assertIn("recommend", response)
    
    def test_handle_search(self):
        """Test /search command"""
        response = self.handlers.handle_search(123, "Inception")
        self.assertIn("Inception", response)
        self.assertIn("search results", response.lower())
    
    def test_handle_rate(self):
        """Test /rate command"""
        response = self.handlers.handle_rate(123, "1", "5")
        self.assertIn("rated", response.lower())
    
    def test_handle_stats(self):
        """Test /stats command"""
        response = self.handlers.handle_stats(123)
        self.assertIn("Statistics", response)
        self.assertIn("2", response)  # rating_count


class TestDataIntegration(unittest.TestCase):
    """Test data loading and processing"""
    
    def test_data_loader_import(self):
        """Test that data loader can be imported"""
        from src.data_loader import MovieLensDataLoader
        self.assertIsNotNone(MovieLensDataLoader)
    
    def test_models_import(self):
        """Test that models can be imported"""
        from src.models import (
            CollaborativeFilter,
            ContentBasedFilter,
            HybridRecommender
        )
        self.assertIsNotNone(CollaborativeFilter)
        self.assertIsNotNone(ContentBasedFilter)
        self.assertIsNotNone(HybridRecommender)


if __name__ == '__main__':
    unittest.main()
