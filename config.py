"""
Configuration file for Movie Recommendation Bot
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
BOT_DIR = PROJECT_ROOT / "bot"
SRC_DIR = PROJECT_ROOT / "src"

# Create directories if they don't exist
for dir_path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# MovieLens Dataset URLs
MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATASET_NAME = "ml-100k"

# Model file paths
COLLAB_FILTER_MODEL = MODELS_DIR / "collaborative_filter.pkl"
CONTENT_FILTER_MODEL = MODELS_DIR / "content_filter.pkl"
USER_ITEM_MATRIX = MODELS_DIR / "user_item_matrix.pkl"
MOVIES_DB = DATA_PROCESSED_DIR / "movies.csv"
RATINGS_DB = DATA_PROCESSED_DIR / "ratings.csv"

# Telegram Bot Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")

# Model Parameters
N_RECOMMENDATIONS = 5
MIN_RATINGS_FOR_CF = 2  # Minimum ratings user needs for collab filtering
COLLAB_SIMILARITY_THRESHOLD = 0.1
CONTENT_SIMILARITY_THRESHOLD = 0.2

# Data Parameters
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
