"""
Data collection and preprocessing module for MovieLens dataset
"""
import os
import zipfile
import urllib.request
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import config


class MovieLensDataLoader:
    """Load and preprocess MovieLens dataset"""
    
    def __init__(self, dataset_url=config.MOVIELENS_URL):
        self.dataset_url = dataset_url
        self.dataset_name = config.DATASET_NAME
        self.raw_dir = config.DATA_RAW_DIR
        self.processed_dir = config.DATA_PROCESSED_DIR
        
    def download_dataset(self):
        """Download MovieLens dataset"""
        logger.info(f"Downloading MovieLens dataset from {self.dataset_url}...")
        
        zip_path = self.raw_dir / f"{self.dataset_name}.zip"
        
        # Check if already downloaded
        if zip_path.exists():
            logger.info("Dataset already downloaded!")
            return zip_path
        
        try:
            urllib.request.urlretrieve(self.dataset_url, zip_path)
            logger.info(f"Downloaded to {zip_path}")
            return zip_path
        except Exception as e:
            logger.error(f"Failed to download: {e}")
            raise
    
    def extract_dataset(self, zip_path):
        """Extract ZIP file"""
        logger.info(f"Extracting {zip_path}...")
        
        extract_path = self.raw_dir / self.dataset_name
        if extract_path.exists():
            logger.info("Dataset already extracted!")
            return extract_path
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir)
            logger.info(f"Extracted to {extract_path}")
            return extract_path
        except Exception as e:
            logger.error(f"Failed to extract: {e}")
            raise
    
    def load_raw_data(self, extract_path):
        """Load raw MovieLens files"""
        logger.info("Loading raw data files...")
        
        data_path = extract_path / self.dataset_name
        
        # Load ratings
        ratings_file = data_path / "u.data"
        ratings = pd.read_csv(
            ratings_file,
            sep='\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python'
        )
        logger.info(f"Loaded {len(ratings)} ratings")
        
        # Load movies
        movies_file = data_path / "u.item"
        movies = pd.read_csv(
            movies_file,
            sep='|',
            names=[
                'movie_id', 'title', 'release_date', 'video_release_date',
                'imdb_url', 'unknown', 'action', 'adventure', 'animation',
                'children', 'comedy', 'crime', 'documentary', 'drama',
                'fantasy', 'film_noir', 'horror', 'musical', 'mystery',
                'romance', 'sci_fi', 'thriller', 'war', 'western'
            ],
            engine='python'
        )
        logger.info(f"Loaded {len(movies)} movies")
        
        # Load users
        users_file = data_path / "u.user"
        users = pd.read_csv(
            users_file,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            engine='python'
        )
        logger.info(f"Loaded {len(users)} users")
        
        return ratings, movies, users
    
    def preprocess_data(self, ratings, movies, users):
        """Clean and preprocess data"""
        logger.info("Preprocessing data...")
        
        # Clean ratings
        ratings = ratings.dropna()
        
        # Convert timestamp to datetime
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        
        # Clean movies - keep only relevant columns
        movies = movies[['movie_id', 'title', 'release_date', 'action', 'adventure', 
                         'animation', 'children', 'comedy', 'crime', 'documentary', 
                         'drama', 'fantasy', 'horror', 'musical', 'mystery', 
                         'romance', 'sci_fi', 'thriller', 'war', 'western']]
        
        # Convert release_date to datetime
        movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
        
        # Add rating statistics to movies
        movie_stats = ratings.groupby('movie_id').agg({
            'rating': ['mean', 'count', 'std']
        }).reset_index()
        movie_stats.columns = ['movie_id', 'avg_rating', 'num_ratings', 'rating_std']
        
        movies = movies.merge(movie_stats, on='movie_id', how='left')
        movies['rating_std'] = movies['rating_std'].fillna(0)
        
        logger.info(f"Preprocessed: {len(ratings)} ratings, {len(movies)} movies")
        
        return ratings, movies, users
    
    def save_processed_data(self, ratings, movies, users):
        """Save preprocessed data to CSV"""
        logger.info("Saving processed data...")
        
        ratings.to_csv(config.RATINGS_DB, index=False)
        movies.to_csv(config.MOVIES_DB, index=False)
        
        logger.info(f"Saved to {config.RATINGS_DB} and {config.MOVIES_DB}")
    
    def load_processed_data(self):
        """Load previously processed data"""
        if config.RATINGS_DB.exists() and config.MOVIES_DB.exists():
            logger.info("Loading previously processed data...")
            ratings = pd.read_csv(config.RATINGS_DB)
            movies = pd.read_csv(config.MOVIES_DB)
            return ratings, movies
        return None, None
    
    def get_data(self, force_reprocess=False):
        """Get data (download, extract, preprocess if needed)"""
        # Try to load existing processed data
        if not force_reprocess:
            ratings, movies = self.load_processed_data()
            if ratings is not None and movies is not None:
                return ratings, movies
        
        # Download and process
        zip_path = self.download_dataset()
        extract_path = self.extract_dataset(zip_path)
        ratings, movies, users = self.load_raw_data(extract_path)
        ratings, movies, users = self.preprocess_data(ratings, movies, users)
        self.save_processed_data(ratings, movies, users)
        
        return ratings, movies


def get_movie_genres(movie_row):
    """Extract genre list from movie row"""
    genre_cols = ['action', 'adventure', 'animation', 'children', 'comedy', 
                  'crime', 'documentary', 'drama', 'fantasy', 'horror', 
                  'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']
    return [col for col in genre_cols if movie_row[col] == 1]


if __name__ == "__main__":
    loader = MovieLensDataLoader()
    ratings, movies = loader.get_data()
    
    print("\n=== DATA SUMMARY ===")
    print(f"Ratings shape: {ratings.shape}")
    print(f"Movies shape: {movies.shape}")
    print(f"\nRatings statistics:\n{ratings['rating'].describe()}")
    print(f"\nMovies with highest avg rating:\n{movies.nlargest(5, 'avg_rating')[['title', 'avg_rating', 'num_ratings']]}")
