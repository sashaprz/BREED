import os
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List
import torch
import numpy as np
from functools import wraps
import time

logger = logging.getLogger(__name__)


class DataCache:
    """
    Enhanced caching system for CDVAE training data with memory management
    and persistent storage capabilities.
    """
    
    def __init__(self, cache_dir: str = "cache", max_memory_items: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_memory_items = max_memory_items
        
        # In-memory cache for frequently accessed items
        self.memory_cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        
        # Metadata cache
        self.metadata_file = self.cache_dir / "cache_metadata.pkl"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {"file_hashes": {}, "cache_stats": {}}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_file_hash(self, filepath: str) -> str:
        """Generate hash for file to detect changes."""
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash file {filepath}: {e}")
            return str(time.time())  # Fallback to timestamp
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _manage_memory_cache(self):
        """Remove least recently used items from memory cache."""
        if len(self.memory_cache) <= self.max_memory_items:
            return
            
        # Sort by access time and remove oldest items
        sorted_items = sorted(
            self.access_times.items(), 
            key=lambda x: x[1]
        )
        
        items_to_remove = len(self.memory_cache) - self.max_memory_items + 100
        for key, _ in sorted_items[:items_to_remove]:
            if key in self.memory_cache:
                del self.memory_cache[key]
                del self.access_times[key]
                if key in self.access_counts:
                    del self.access_counts[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache."""
        current_time = time.time()
        
        # Check memory cache first
        if key in self.memory_cache:
            self.access_times[key] = current_time
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Add to memory cache
                self.memory_cache[key] = data
                self.access_times[key] = current_time
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                
                self._manage_memory_cache()
                return data
            except Exception as e:
                logger.warning(f"Failed to load cached data for key {key}: {e}")
        
        return default
    
    def set(self, key: str, value: Any, save_to_disk: bool = True):
        """Set item in cache."""
        current_time = time.time()
        
        # Add to memory cache
        self.memory_cache[key] = value
        self.access_times[key] = current_time
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        
        # Save to disk if requested
        if save_to_disk:
            cache_file = self.cache_dir / f"{key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                logger.warning(f"Failed to save cached data for key {key}: {e}")
        
        self._manage_memory_cache()
    
    def invalidate(self, key: str):
        """Remove item from cache."""
        if key in self.memory_cache:
            del self.memory_cache[key]
            del self.access_times[key]
            if key in self.access_counts:
                del self.access_counts[key]
        
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            cache_file.unlink()
    
    def clear(self):
        """Clear all cached data."""
        self.memory_cache.clear()
        self.access_times.clear()
        self.access_counts.clear()
        
        # Remove all cache files
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.name != "cache_metadata.pkl":
                cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_files": len(list(self.cache_dir.glob("*.pkl"))) - 1,  # Exclude metadata
            "total_access_counts": sum(self.access_counts.values()),
            "most_accessed": max(self.access_counts.items(), key=lambda x: x[1]) if self.access_counts else None,
        }


def cached_preprocessing(cache_dir: str = "preprocessing_cache"):
    """
    Decorator for caching preprocessing results.
    """
    cache = DataCache(cache_dir)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache._get_cache_key(func.__name__, *args, **kwargs)
            
            # Check if we have cached result
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached result for {func.__name__}")
                return cached_result
            
            # Compute result and cache it
            logger.info(f"Computing and caching result for {func.__name__}")
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            
            return result
        
        wrapper.cache = cache
        return wrapper
    
    return decorator


class BatchCache:
    """
    Cache for batched data to speed up training.
    """
    
    def __init__(self, max_batches: int = 100):
        self.max_batches = max_batches
        self.batches: Dict[str, torch.Tensor] = {}
        self.batch_order: List[str] = []
    
    def get_batch(self, batch_key: str) -> Optional[torch.Tensor]:
        """Get cached batch."""
        if batch_key in self.batches:
            # Move to end (most recently used)
            self.batch_order.remove(batch_key)
            self.batch_order.append(batch_key)
            return self.batches[batch_key]
        return None
    
    def cache_batch(self, batch_key: str, batch_data: torch.Tensor):
        """Cache batch data."""
        if batch_key in self.batches:
            # Update existing
            self.batches[batch_key] = batch_data
            self.batch_order.remove(batch_key)
            self.batch_order.append(batch_key)
        else:
            # Add new batch
            if len(self.batches) >= self.max_batches:
                # Remove oldest batch
                oldest_key = self.batch_order.pop(0)
                del self.batches[oldest_key]
            
            self.batches[batch_key] = batch_data
            self.batch_order.append(batch_key)
    
    def clear(self):
        """Clear all cached batches."""
        self.batches.clear()
        self.batch_order.clear()


# Global cache instances
_data_cache = None
_batch_cache = None


def get_data_cache(cache_dir: str = "data_cache") -> DataCache:
    """Get global data cache instance."""
    global _data_cache
    if _data_cache is None:
        _data_cache = DataCache(cache_dir)
    return _data_cache


def get_batch_cache(max_batches: int = 100) -> BatchCache:
    """Get global batch cache instance."""
    global _batch_cache
    if _batch_cache is None:
        _batch_cache = BatchCache(max_batches)
    return _batch_cache