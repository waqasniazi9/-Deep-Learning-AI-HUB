"""
Utility functions and decorators for efficient data processing and caching.
"""

import logging
import functools
import time
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

# Simple in-memory cache with TTL
_cache: Dict[str, Dict[str, Any]] = {}


def setup_logging(config):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def memoize_with_ttl(ttl_seconds: int = 3600):
    """
    Decorator to cache function results with TTL (Time To Live).
    Automatically expires cached values after the specified duration.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create a cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Check if cached value exists and hasn't expired
            if cache_key in _cache:
                cached_data = _cache[cache_key]
                if datetime.now() < cached_data['expires']:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_data['value']
                else:
                    del _cache[cache_key]

            # Execute function and cache result
            result = func(*args, **kwargs)
            _cache[cache_key] = {
                'value': result,
                'expires': datetime.now() + timedelta(seconds=ttl_seconds),
                'created': datetime.now()
            }
            logger.debug(f"Cached result for {func.__name__}")
            return result

        return wrapper
    return decorator


def clear_cache(pattern: Optional[str] = None):
    """Clear cached values, optionally by pattern."""
    global _cache
    if pattern is None:
        _cache.clear()
        logger.info("Cache cleared")
    else:
        keys_to_delete = [k for k in _cache.keys() if pattern in k]
        for key in keys_to_delete:
            del _cache[key]
        logger.info(
            f"Cleared {len(keys_to_delete)} cache entries matching pattern: {pattern}")


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the cache."""
    return {
        'total_entries': len(_cache),
        'cache_keys': list(_cache.keys()),
        'timestamp': datetime.now().isoformat()
    }


def time_operation(func: Callable) -> Callable:
    """Decorator to log execution time of function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper


def handle_errors(default_return: Any = None):
    """Decorator for consistent error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {str(e)}", exc_info=True)
                return default_return or {'error': str(e), 'status': 'error'}
        return wrapper
    return decorator


def paginate(items: List[Any], page: int = 1, per_page: int = 20) -> Dict[str, Any]:
    """Paginate a list of items."""
    total = len(items)
    pages = (total + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page

    return {
        'items': items[start:end],
        'current_page': page,
        'total_pages': pages,
        'total_items': total,
        'per_page': per_page,
        'has_next': page < pages,
        'has_prev': page > 1
    }


def format_response(data: Any, status: str = 'success', message: str = '') -> Dict[str, Any]:
    """Format API response consistently."""
    return {
        'status': status,
        'data': data,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }


def validate_input(data: Dict[str, Any], required_fields: List[str]) -> tuple[bool, Optional[str]]:
    """Validate that required fields are present in data."""
    for field in required_fields:
        if field not in data or data[field] is None:
            return False, f"Missing required field: {field}"
    return True, None


def convert_to_json_serializable(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to JSON-serializable format."""
    import numpy as np
    import torch

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj


__all__ = [
    'setup_logging', 'memoize_with_ttl', 'clear_cache', 'get_cache_stats',
    'time_operation', 'handle_errors', 'paginate', 'format_response',
    'validate_input', 'convert_to_json_serializable'
]
