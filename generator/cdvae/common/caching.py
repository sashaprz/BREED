"""
Caching utilities for CDVAE data preprocessing.
"""
import os
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Dict


class DataCache:
    """Simple file-based cache for preprocessed data."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a given key."""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data by key."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"Cache hit: {key}")
                return data
            except Exception as e:
                print(f"Cache read error for {key}: {e}")
                # Remove corrupted cache file
                try:
                    cache_path.unlink()
                except:
                    pass
        return None
    
    def set(self, key: str, data: Any) -> None:
        """Set cached data by key."""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cache stored: {key}")
        except Exception as e:
            print(f"Cache write error for {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass
        print(f"Cache cleared: {self.cache_dir}")
    
    def size(self) -> int:
        """Get number of cached items."""
        return len(list(self.cache_dir.glob("*.pkl")))
    
    def info(self) -> Dict[str, Any]:
        """Get cache information."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            "cache_dir": str(self.cache_dir),
            "num_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "files": [f.name for f in cache_files]
        }


def get_data_cache(cache_dir: str = "./data_cache") -> DataCache:
    """Get a data cache instance."""
    return DataCache(cache_dir)


def cached_preprocessing(cache_key: str, preprocessing_func, *args, **kwargs):
    """
    Decorator-like function for caching preprocessing results.
    
    Args:
        cache_key: Unique key for this preprocessing operation
        preprocessing_func: Function to call if cache miss
        *args, **kwargs: Arguments to pass to preprocessing_func
    
    Returns:
        Cached or newly computed preprocessing results
    """
    cache = get_data_cache()
    
    # Try to get from cache first
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Cache miss - compute and store
    print(f"Computing preprocessing for cache key: {cache_key}")
    start_time = time.time()
    result = preprocessing_func(*args, **kwargs)
    end_time = time.time()
    
    print(f"Preprocessing completed in {end_time - start_time:.2f} seconds")
    cache.set(cache_key, result)
    
    return result


def create_cache_key(data_path: str, **params) -> str:
    """
    Create a unique cache key based on data path and parameters.
    
    Args:
        data_path: Path to the data file
        **params: Additional parameters that affect preprocessing
    
    Returns:
        MD5 hash string to use as cache key
    """
    # Include file modification time and size for cache invalidation
    try:
        stat = os.stat(data_path)
        file_info = f"{data_path}_{stat.st_size}_{stat.st_mtime}"
    except:
        file_info = str(data_path)
    
    # Create string from all parameters
    param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    cache_data = f"{file_info}_{param_str}"
    
    # Return MD5 hash
    return hashlib.md5(cache_data.encode()).hexdigest()