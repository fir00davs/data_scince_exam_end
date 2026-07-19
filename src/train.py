"""
Train recommendation models
"""
import logging
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

import config
from src.data_loader import MovieLensDataLoader
from src.models import CollaborativeFilter, ContentBasedFilter, HybridRecommender, UserItemMatrix


def evaluate_model(model, test_ratings, movies_df, n_samples=100):
    """Evaluate model on test set"""
    logger.info(f"Evaluating model on {n_samples} test samples...")
    
    test_users = test_ratings['user_id'].unique()[:n_samples]
    
    mae_scores = []
    precision_scores = []
    
    for user_id in test_users:
        user_test_ratings = test_ratings[test_ratings['user_id'] == user_id]
        
        if len(user_test_ratings) < 2:
            continue
        
        # Get recommendations
        recs = model.get_recommendations(user_id, movies_df, n=5)
        if not recs:
            continue
        
        rec_movie_ids = [r['movie_id'] for r in recs]
        test_movie_ids = user_test_ratings['movie_id'].values
        
        # Precision@5
        hits = len(set(rec_movie_ids) & set(test_movie_ids))
        precision = hits / min(5, len(test_movie_ids))
        precision_scores.append(precision)
        
        # Calculate MAE for predicted ratings
        for movie_id in rec_movie_ids:
            actual = user_test_ratings[user_test_ratings['movie_id'] == movie_id]['rating']
            if len(actual) > 0:
                predicted = next((r['score'] for r in recs if r['movie_id'] == movie_id), 3.0)
                mae_scores.append(abs(actual.values[0] - predicted))
    
    avg_mae = sum(mae_scores) / len(mae_scores) if mae_scores else 0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    
    logger.info(f"MAE: {avg_mae:.4f}")
    logger.info(f"Precision@5: {avg_precision:.4f}")
    
    return {
        'mae': avg_mae,
        'precision': avg_precision
    }


def train_models(ratings_df, movies_df, test_size=0.2):
    """Train all recommendation models"""
    
    logger.info("=" * 60)
    logger.info("TRAINING RECOMMENDATION MODELS")
    logger.info("=" * 60)
    
    # Split data
    train_ratings, test_ratings = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=config.RANDOM_STATE
    )
    
    logger.info(f"Train set: {len(train_ratings)} ratings")
    logger.info(f"Test set: {len(test_ratings)} ratings")
    
    # Train Collaborative Filtering
    logger.info("\n--- Training Collaborative Filtering ---")
    collab_filter = CollaborativeFilter(train_ratings)
    
    # Train Content-Based Filtering
    logger.info("\n--- Training Content-Based Filtering ---")
    content_filter = ContentBasedFilter(movies_df, train_ratings)
    
    # Create Hybrid Recommender
    logger.info("\n--- Creating Hybrid Recommender ---")
    hybrid = HybridRecommender(collab_filter, content_filter, collab_weight=0.6)
    
    # Evaluate on test set
    logger.info("\n--- Evaluating Model ---")
    metrics = evaluate_model(hybrid, test_ratings, movies_df, n_samples=50)
    
    # Save models
    logger.info("\n--- Saving Models ---")
    hybrid.save(config.COLLAB_FILTER_MODEL, config.CONTENT_FILTER_MODEL)
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    return hybrid, metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT
    )
    
    # Load data
    loader = MovieLensDataLoader()
    ratings, movies = loader.get_data()
    
    # Train models
    model, metrics = train_models(ratings, movies)
    
    print("\n" + "="*60)
    print("EVALUATION METRICS:")
    print("="*60)
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Precision@5: {metrics['precision']:.4f}")
    print("="*60)
