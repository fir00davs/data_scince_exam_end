"""
Telegram bot command handlers
"""
import logging
import os
import sys
from typing import Optional

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

import config
from bot.model_loader import ModelLoader


class BotHandlers:
    """Handle Telegram bot commands"""
    
    def __init__(self, model_loader: ModelLoader):
        self.model = model_loader
    
    # Helper methods
    def _format_rating_message(self, rating: int) -> str:
        """Convert rating to emoji"""
        stars = "⭐" * rating + "☆" * (5 - rating)
        return f"{stars} ({rating}/5)"
    
    def _escape_markdown(self, text: str) -> str:
        """Escape markdown special characters"""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text
    
    # Command handlers
    def handle_start(self, user_id: int) -> str:
        """Handle /start command"""
        stats = self.model.get_user_stats(user_id)
        
        welcome = (
            "🎬 *Welcome to Movie Recommendation Bot\\!*\n\n"
            "I can help you discover movies you'll love\\!\n\n"
            "*Available Commands:*\n"
            "📌 `/search [movie name]` \\- Search for a movie\n"
            "⭐ `/rate [movie id] [1\\-5]` \\- Rate a movie\n"
            "🎯 `/recommend` \\- Get personalized recommendations\n"
            "📊 `/stats` \\- View your stats\n"
            "❓ `/help` \\- Show help message\n"
            "🗑️ `/clear` \\- Clear your ratings\n\n"
            f"*Your Stats:*\n"
            f"Movies rated: {stats['rating_count']}\n"
            f"Average rating: {stats['average_rating']}/5\n"
        )
        
        if stats['can_get_recommendations']:
            welcome += "\n✅ You can now get personalized recommendations\\!"
        else:
            welcome += f"\n❌ Rate at least {config.MIN_RATINGS_FOR_CF} movies to get recommendations"
        
        return welcome
    
    def handle_help(self) -> str:
        """Handle /help command"""
        return (
            "🎬 *Movie Recommendation Bot Help*\n\n"
            "*How to use:*\n"
            "1️⃣ Search for movies: `/search Inception`\n"
            "2️⃣ Rate movies: `/rate 1 5` (rate movie ID 1 with 5 stars)\n"
            "3️⃣ Get recommendations: `/recommend`\n\n"
            "*Commands:*\n"
            "• `/search [query]` \\- Search movies\n"
            "• `/rate [id] [rating]` \\- Rate a movie (1\\-5)\n"
            "• `/recommend` \\- Get 5 personalized recommendations\n"
            "• `/stats` \\- Your personal statistics\n"
            "• `/clear` \\- Clear all your ratings\n"
            "• `/help` \\- This message\n\n"
            "*Tips:*\n"
            "💡 Rate more movies to get better recommendations\n"
            "💡 The bot learns from your taste\n"
            "💡 Use `/stats` to track your activity"
        )
    
    def handle_search(self, user_id: int, query: str) -> str:
        """Handle /search command"""
        if not query or len(query.strip()) < 2:
            return "❌ Please provide a movie name to search\\. Example: `/search Inception`"
        
        results = self.model.search_movies(query.strip(), limit=10)
        
        if not results:
            return f"❌ No movies found for: *{query}*"
        
        message = f"🔍 *Search results for: {query}*\n\n"
        
        for movie in results:
            movie_id = movie['movie_id']
            title = self._escape_markdown(movie['title'])
            rating = movie['avg_rating']
            votes = movie['num_ratings']
            
            message += (
                f"*ID: {movie_id}*\n"
                f"📽️ {title}\n"
                f"⭐ {rating}/5 \\({votes} votes\\)\n"
                f"→ Rate with: `/rate {movie_id} 1-5`\n\n"
            )
        
        return message
    
    def handle_rate(self, user_id: int, movie_id_str: str, rating_str: str) -> str:
        """Handle /rate command"""
        try:
            movie_id = int(movie_id_str)
            rating = int(rating_str)
        except ValueError:
            return "❌ Invalid format\\. Use: `/rate [movie_id] [1-5]`"
        
        if rating < 1 or rating > 5:
            return "❌ Rating must be between 1 and 5\\!"
        
        # Get movie info
        movie_info = self.model.get_movie_info(movie_id)
        if movie_info is None:
            return f"❌ Movie with ID {movie_id} not found\\!"
        
        # Add rating
        if self.model.add_user_rating(user_id, movie_id, rating):
            title = self._escape_markdown(movie_info['title'])
            stars = "⭐" * rating + "☆" * (5 - rating)
            
            return (
                f"✅ *You rated:* {title}\n"
                f"{stars}\n\n"
                f"💡 Rate more movies to get better recommendations\\!\n"
                f"Use `/recommend` to get personalized suggestions"
            )
        else:
            return "❌ Failed to save rating\\. Please try again\\."
    
    def handle_recommend(self, user_id: int) -> str:
        """Handle /recommend command"""
        stats = self.model.get_user_stats(user_id)
        
        if not stats['can_get_recommendations']:
            popular = self.model._get_popular_movies(config.N_RECOMMENDATIONS)
            
            message = (
                f"⚠️ *Rate at least {config.MIN_RATINGS_FOR_CF} movies first\\!*\n\n"
                f"*Popular movies right now:*\n\n"
            )
            
            for i, movie in enumerate(popular, 1):
                title = self._escape_markdown(movie['title'])
                rating = movie['avg_rating']
                votes = movie['num_ratings']
                
                message += (
                    f"{i}\\. *{title}*\n"
                    f"   ⭐ {rating}/5 \\({votes} votes\\)\n"
                    f"   ID: {movie['movie_id']}\n\n"
                )
            
            return message
        
        recommendations = self.model.get_recommendations(user_id, config.N_RECOMMENDATIONS)
        
        if not recommendations:
            return "❌ Could not generate recommendations\\. Please try again\\."
        
        message = "🎯 *Your Personalized Recommendations:*\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            title = self._escape_markdown(rec['title'])
            rating = rec['avg_rating']
            votes = rec['num_ratings']
            movie_id = rec['movie_id']
            
            message += (
                f"{i}\\. *{title}*\n"
                f"   ⭐ {rating}/5 \\({votes} votes\\)\n"
                f"   ID: {movie_id}\n"
                f"   💡 To rate: `/rate {movie_id} 1-5`\n\n"
            )
        
        message += "💬 Rate these movies and run `/recommend` again to improve suggestions\\!"
        
        return message
    
    def handle_stats(self, user_id: int) -> str:
        """Handle /stats command"""
        stats = self.model.get_user_stats(user_id)
        
        message = (
            "📊 *Your Statistics:*\n\n"
            f"🎬 Movies rated: *{stats['rating_count']}*\n"
            f"⭐ Average rating: *{stats['average_rating']}/5*\n\n"
        )
        
        if stats['can_get_recommendations']:
            message += "✅ You can get personalized recommendations"
        else:
            needed = config.MIN_RATINGS_FOR_CF - stats['rating_count']
            message += f"❌ Rate {needed} more movie(s) to unlock recommendations"
        
        return message
    
    def handle_clear(self, user_id: int) -> str:
        """Handle /clear command"""
        self.model.user_manager.clear_user_ratings(user_id)
        
        return (
            "🗑️ *All your ratings have been cleared\\!*\n\n"
            "Start fresh with `/search` and `/rate`"
        )
    
    def handle_error(self, command: str) -> str:
        """Handle unknown commands"""
        return (
            f"❌ Unknown command: `{command}`\n\n"
            "Use `/help` to see available commands"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    loader = ModelLoader()
    handlers = BotHandlers(loader)
    
    # Test handlers
    print("=== Start Message ===")
    print(handlers.handle_start(123))
    
    print("\n=== Help Message ===")
    print(handlers.handle_help())
