"""
Movie Recommendation Models - Collaborative and Content-Based Filtering
"""
import logging
import pickle
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

import config
from src.data_loader import get_movie_genres


class UserItemMatrix:
    """Create and manage user-item rating matrix"""
    
    def __init__(self, ratings_df):
        self.ratings = ratings_df
        self.user_item_matrix = None
        self.user_id_map = None
        self.item_id_map = None
        self.build_matrix()
    
    def build_matrix(self):
        """Build user-item rating matrix"""
        logger.info("Building user-item matrix...")
        
        self.user_item_matrix = self.ratings.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(self.user_item_matrix.index)}
        self.item_id_map = {mid: idx for idx, mid in enumerate(self.user_item_matrix.columns)}
        
        logger.info(f"Matrix shape: {self.user_item_matrix.shape}")
        return self.user_item_matrix
    
    def get_matrix(self):
        """Return the user-item matrix"""
        return self.user_item_matrix
    
    def save(self, path):
        """Save matrix to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'matrix': self.user_item_matrix,
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map
            }, f)
        logger.info(f"Saved user-item matrix to {path}")
    
    @staticmethod
    def load(path):
        """Load matrix from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        matrix_obj = UserItemMatrix.__new__(UserItemMatrix)
        matrix_obj.user_item_matrix = data['matrix']
        matrix_obj.user_id_map = data['user_id_map']
        matrix_obj.item_id_map = data['item_id_map']
        logger.info(f"Loaded user-item matrix from {path}")
        return matrix_obj


class CollaborativeFilter:
    """User-User Collaborative Filtering"""
    
    def __init__(self, ratings_df):
        self.ratings = ratings_df
        self.matrix = UserItemMatrix(ratings_df)
        self.user_similarity = None
        self.compute_similarity()
    
    def compute_similarity(self):
        """Compute user-user similarity matrix using cosine similarity"""
        logger.info("Computing user similarity matrix...")
        
        # Fill NaN with 0 for similarity computation
        matrix_filled = self.matrix.user_item_matrix.fillna(0)
        
        # Compute cosine similarity
        self.user_similarity = pd.DataFrame(
            cosine_similarity(matrix_filled),
            index=matrix_filled.index,
            columns=matrix_filled.index
        )
        
        logger.info(f"Similarity matrix shape: {self.user_similarity.shape}")
    
    def get_recommendations(self, user_id, n=5, top_k_similar=10):
        """Get recommendations for a user using collaborative filtering
        
        Args:
            user_id: Target user ID
            n: Number of recommendations to return
            top_k_similar: Number of similar users to consider
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.matrix.user_item_matrix.index:
            logger.warning(f"User {user_id} not in training data")
            return []
        
        # Get similar users
        similar_users = self.user_similarity[user_id].nlargest(top_k_similar + 1)[1:]  # Exclude self
        
        # Get movies rated by similar users
        user_rated = set(self.matrix.user_item_matrix.loc[user_id][self.matrix.user_item_matrix.loc[user_id] > 0].index)
        
        recommendations = {}
        for similar_user_id in similar_users.index:
            similar_user_ratings = self.matrix.user_item_matrix.loc[similar_user_id]
            unrated_movies = similar_user_ratings[similar_user_ratings > 0].index
            unrated_movies = [m for m in unrated_movies if m not in user_rated]
            
            for movie_id in unrated_movies:
                if movie_id not in recommendations:
                    recommendations[movie_id] = []
                # Weight by similarity
                recommendations[movie_id].append(
                    similar_users[similar_user_id] * similar_user_ratings[movie_id]
                )
        
        # Average weighted scores
        final_scores = {
            movie_id: np.mean(scores)
            for movie_id, scores in recommendations.items()
        }
        
        # Sort and return top N
        top_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        return top_recommendations
    
    def save(self, path):
        """Save model to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'ratings': self.ratings,
                'user_similarity': self.user_similarity,
                'matrix': self.matrix
            }, f)
        logger.info(f"Saved collaborative filter to {path}")
        self.matrix.save(config.USER_ITEM_MATRIX)
    
    @staticmethod
    def load(path):
        """Load model from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = CollaborativeFilter.__new__(CollaborativeFilter)
        model.ratings = data['ratings']
        model.user_similarity = data['user_similarity']
        model.matrix = data['matrix']
        logger.info(f"Loaded collaborative filter from {path}")
        return model


class ContentBasedFilter:
    """Content-Based Filtering using movie genres and ratings"""
    
    def __init__(self, movies_df, ratings_df):
        self.movies = movies_df.copy()
        self.ratings = ratings_df
        self.movie_features = None
        self.movie_similarity = None
        self.build_features()
    
    def build_features(self):
        """Build movie feature matrix (genres + ratings)"""
        logger.info("Building movie content features...")
        
        genre_cols = ['action', 'adventure', 'animation', 'children', 'comedy', 
                      'crime', 'documentary', 'drama', 'fantasy', 'horror', 
                      'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']
        
        # Prepare features
        self.movie_features = self.movies[genre_cols + ['avg_rating']].fillna(0)
        
        # Normalize rating to 0-1
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.movie_features)
        
        # Compute item-item similarity
        self.movie_similarity = pd.DataFrame(
            cosine_similarity(features_scaled),
            index=self.movies['movie_id'],
            columns=self.movies['movie_id']
        )
        
        logger.info(f"Movie feature matrix shape: {self.movie_features.shape}")
    
    def get_recommendations(self, user_id, movies_df, n=5):
        """Get recommendations based on movies user liked
        
        Args:
            user_id: Target user ID
            movies_df: Movies dataframe
            n: Number of recommendations to return
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        # Get user's rated movies (rated >= 4)
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        if len(user_ratings) == 0:
            return []
        
        liked_movies = user_ratings[user_ratings['rating'] >= 4]['movie_id'].values
        if len(liked_movies) == 0:
            return []
        
        # Get similar movies
        recommendations = {}
        for movie_id in liked_movies:
            if movie_id in self.movie_similarity.index:
                similar_movies = self.movie_similarity[movie_id]
                
                for similar_movie_id, similarity in similar_movies.items():
                    # Don't recommend already rated movies
                    if similar_movie_id not in user_ratings['movie_id'].values:
                        if similar_movie_id not in recommendations:
                            recommendations[similar_movie_id] = []
                        recommendations[similar_movie_id].append(similarity)
        
        # Average similarity scores
        final_scores = {
            movie_id: np.mean(scores)
            for movie_id, scores in recommendations.items()
        }
        
        top_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        return top_recommendations
    
    def save(self, path):
        """Save model to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'movies': self.movies,
                'ratings': self.ratings,
                'movie_features': self.movie_features,
                'movie_similarity': self.movie_similarity
            }, f)
        logger.info(f"Saved content-based filter to {path}")
    
    @staticmethod
    def load(path):
        """Load model from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = ContentBasedFilter.__new__(ContentBasedFilter)
        model.movies = data['movies']
        model.ratings = data['ratings']
        model.movie_features = data['movie_features']
        model.movie_similarity = data['movie_similarity']
        logger.info(f"Loaded content-based filter from {path}")
        return model


class HybridRecommender:
    """Hybrid recommendation system combining collaborative and content-based filtering"""
    
    def __init__(self, collab_filter, content_filter, collab_weight=0.6):
        self.collab_filter = collab_filter
        self.content_filter = content_filter
        self.collab_weight = collab_weight
        self.content_weight = 1 - collab_weight
    
    def get_recommendations(self, user_id, movies_df, n=5):
        """Get hybrid recommendations
        
        Args:
            user_id: Target user ID
            movies_df: Movies dataframe
            n: Number of recommendations
            
        Returns:
            List of movie dicts with recommendations
        """
        recommendations = {}
        
        # Get collaborative filtering recommendations
        collab_recs = self.collab_filter.get_recommendations(user_id, n=n*2)
        for movie_id, score in collab_recs:
            recommendations[movie_id] = self.collab_weight * (score / 5.0)  # Normalize
        
        # Get content-based recommendations
        content_recs = self.content_filter.get_recommendations(user_id, movies_df, n=n*2)
        for movie_id, score in content_recs:
            if movie_id in recommendations:
                recommendations[movie_id] += self.content_weight * score
            else:
                recommendations[movie_id] = self.content_weight * score
        
        # Sort and return top N
        top_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Enrich with movie info
        result = []
        for movie_id, score in top_recs:
            movie_info = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
            result.append({
                'movie_id': movie_id,
                'title': movie_info['title'],
                'score': score,
                'avg_rating': movie_info['avg_rating'],
                'num_ratings': movie_info['num_ratings']
            })
        
        return result
    
    def save(self, collab_path, content_path):
        """Save both models"""
        self.collab_filter.save(collab_path)
        self.content_filter.save(content_path)
        logger.info("Saved hybrid recommender models")
    
    @staticmethod
    def load(collab_path, content_path, collab_weight=0.6):
        """Load both models"""
        collab_filter = CollaborativeFilter.load(collab_path)
        content_filter = ContentBasedFilter.load(content_path)
        return HybridRecommender(collab_filter, content_filter, collab_weight)
