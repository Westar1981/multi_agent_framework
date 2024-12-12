"""Cache manager for optimizing resource usage through intelligent caching."""

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
from collections import OrderedDict
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    max_size: int = 1000
    ttl_seconds: int = 300  # 5 minutes
    cleanup_interval: int = 60  # 1 minute
    hit_threshold: int = 5  # Minimum hits to keep in cache
    memory_threshold: float = 0.8  # 80% memory threshold

@dataclass
class CacheStats:
    """Statistics for cache performance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage: float = 0.0
    avg_access_time: float = 0.0

class LRUCache:
    """Least Recently Used (LRU) cache implementation."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.access_count: Dict[Any, int] = {}
        self._lock = threading.Lock()
        
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return value
            return None
            
    def put(self, key: Any, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Evict least recently used
                self.cache.popitem(last=False)
            self.cache[key] = value
            self.access_count[key] = self.access_count.get(key, 0) + 1
            
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.access_count.clear()

class TimedCache:
    """Cache with time-based expiration."""
    
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self.cache: Dict[Any, Tuple[Any, float]] = {}
        self._lock = threading.Lock()
        
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache if not expired."""
        with self._lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if time.time() < expiry:
                    return value
                # Remove expired
                del self.cache[key]
            return None
            
    def put(self, key: Any, value: Any) -> None:
        """Put item in cache with expiration."""
        with self._lock:
            expiry = time.time() + self.ttl
            self.cache[key] = (value, expiry)
            
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            
    def cleanup(self) -> None:
        """Remove expired entries."""
        with self._lock:
            now = time.time()
            expired = [
                k for k, (_, exp) in self.cache.items()
                if exp < now
            ]
            for k in expired:
                del self.cache[k]

class AdaptiveCache:
    """Cache that adapts size based on hit rate and memory usage."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.lru = LRUCache(config.max_size)
        self.timed = TimedCache(config.ttl_seconds)
        self.stats = CacheStats()
        self._setup_cleanup()
        
    def _setup_cleanup(self) -> None:
        """Setup periodic cleanup."""
        def cleanup():
            while True:
                try:
                    self.cleanup()
                except Exception as e:
                    logger.error(f"Error in cache cleanup: {str(e)}")
                time.sleep(self.config.cleanup_interval)
                
        self._cleanup_thread = threading.Thread(
            target=cleanup,
            daemon=True
        )
        self._cleanup_thread.start()
        
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache with stats tracking."""
        start = time.time()
        
        # Try LRU first
        value = self.lru.get(key)
        if value is not None:
            self.stats.hits += 1
            self._update_stats(time.time() - start)
            return value
            
        # Try timed cache
        value = self.timed.get(key)
        if value is not None:
            # Promote to LRU if frequently accessed
            if self.lru.access_count.get(key, 0) >= self.config.hit_threshold:
                self.lru.put(key, value)
            self.stats.hits += 1
            self._update_stats(time.time() - start)
            return value
            
        self.stats.misses += 1
        self._update_stats(time.time() - start)
        return None
        
    def put(self, key: Any, value: Any) -> None:
        """Put item in cache with adaptive storage."""
        # Store in both caches initially
        self.lru.put(key, value)
        self.timed.put(key, value)
        self.stats.size = len(self.lru.cache) + len(self.timed.cache)
        
    def _update_stats(self, access_time: float) -> None:
        """Update cache statistics."""
        self.stats.avg_access_time = (
            (self.stats.avg_access_time * (self.stats.hits + self.stats.misses - 1) +
             access_time) / (self.stats.hits + self.stats.misses)
        )
        
    def cleanup(self) -> None:
        """Perform cache cleanup and optimization."""
        # Remove expired entries
        self.timed.cleanup()
        
        # Adjust LRU size based on hit rate
        hit_rate = self.stats.hits / (self.stats.hits + self.stats.misses + 1)
        if hit_rate < 0.5 and self.lru.max_size > 100:
            self.lru.max_size = max(100, self.lru.max_size // 2)
            self.stats.evictions += 1
        elif hit_rate > 0.8 and self.stats.memory_usage < self.config.memory_threshold:
            self.lru.max_size = min(
                self.config.max_size,
                self.lru.max_size * 2
            )
            
        # Update stats
        self.stats.size = len(self.lru.cache) + len(self.timed.cache)
        
    def clear(self) -> None:
        """Clear all caches."""
        self.lru.clear()
        self.timed.clear()
        self.stats = CacheStats()

def cached(ttl: Optional[int] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable):
        # Create cache for this function
        cache = AdaptiveCache(
            CacheConfig(
                ttl_seconds=ttl if ttl is not None else 300
            )
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = (
                func.__name__,
                args,
                tuple(sorted(kwargs.items()))
            )
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
                
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
            
        # Attach cache to function for inspection
        wrapper.cache = cache
        return wrapper
    return decorator 