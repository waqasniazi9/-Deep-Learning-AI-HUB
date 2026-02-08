"""
Configuration settings for the AI & Deep Learning Hub application.
Centralized configuration management for development, testing, and production.
"""

import os
from pathlib import Path

# Application paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
CACHE_DIR = BASE_DIR / '.cache'

# Flask configuration
DEBUG = os.getenv('FLASK_ENV') == 'development'
TESTING = os.getenv('FLASK_ENV') == 'testing'
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Server configuration
HOST = os.getenv('HOST', '127.0.0.1')
PORT = int(os.getenv('PORT', 8501))
WORKERS = int(os.getenv('WORKERS', 4))
THREADED = True

# PyTorch/ML configuration
DEVICE = 'cuda' if os.getenv('USE_GPU', 'false').lower() == 'true' else 'cpu'
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 10))

# Caching configuration
CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
CACHE_DEFAULT_TIMEOUT = 3600  # 1 hour
CACHE_TYPE = 'simple'  # Use 'redis' for production

# API configuration
API_VERSION = 'v1'
API_PREFIX = f'/api/{API_VERSION}'
JSON_RESPONSE_TIMEOUT = 30  # seconds
MAX_SEARCH_RESULTS = 100

# Database configuration (for future SQLite/PostgreSQL integration)
DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{BASE_DIR}/app.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = BASE_DIR / 'logs' / 'app.log'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Security
CORS_ENABLED = True
CORS_ORIGINS = ['http://localhost:8501', 'http://127.0.0.1:8501']

# Pagination
ITEMS_PER_PAGE = 20

# Topics configuration
TOPICS_CACHE_TTL = 3600  # Cache topics for 1 hour
MAX_TOPICS = 500

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
LOG_FILE.parent.mkdir(exist_ok=True)

__all__ = [
    'BASE_DIR', 'DATA_DIR', 'CACHE_DIR', 'DEBUG', 'TESTING', 'SECRET_KEY',
    'HOST', 'PORT', 'WORKERS', 'THREADED', 'DEVICE', 'BATCH_SIZE',
    'LEARNING_RATE', 'NUM_EPOCHS', 'CACHE_ENABLED', 'CACHE_DEFAULT_TIMEOUT',
    'DATABASE_URL', 'LOG_LEVEL', 'LOG_FILE', 'CORS_ENABLED', 'TOPICS_CACHE_TTL'
]
