"""Tests for cache manager."""

import time
import pytest
import threading
from unittest.mock import patch, MagicMock

from ..core.cache_manager import (
    CacheConfig,
    CacheStats,
    LRUCache,
    TimedCache,
    AdaptiveCache,
    cached
)

def test_lru_cache():
    """Test LRU cache functionality."""
    cache = LRUCache(max_size=2)
    
    # Test basic operations
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    
    # Test eviction
    cache.put("key3", "value3")
    assert cache.get("key1") is None  # Should be evicted
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"
    
    # Test access count
    assert cache.access_count["key2"] == 2
    assert cache.access_count["key3"] == 1
    
def test_timed_cache():
    """Test timed cache functionality."""
    cache = TimedCache(ttl_seconds=1)
    
    # Test basic operations
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    
    # Test expiration
    time.sleep(1.1)
    assert cache.get("key1") is None
    
    # Test cleanup
    cache.put("key2", "value2")
    time.sleep(1.1)
    cache.cleanup()
    assert len(cache.cache) == 0
    
def test_adaptive_cache():
    """Test adaptive cache functionality."""
    config = CacheConfig(
        max_size=100,
        ttl_seconds=300,
        cleanup_interval=1,
        hit_threshold=2
    )
    cache = AdaptiveCache(config)
    
    # Test basic operations
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.stats.hits == 1
    assert cache.stats.misses == 0
    
    # Test miss
    assert cache.get("key2") is None
    assert cache.stats.misses == 1
    
    # Test promotion to LRU
    cache.put("key3", "value3")
    cache.get("key3")  # First access
    cache.get("key3")  # Second access
    assert "key3" in cache.lru.cache
    
def test_cache_decorator():
    """Test cache decorator."""
    call_count = 0
    
    @cached(ttl=1)
    def expensive_function(x):
        nonlocal call_count
        call_count += 1
        return x * 2
        
    # First call should compute
    result1 = expensive_function(5)
    assert result1 == 10
    assert call_count == 1
    
    # Second call should use cache
    result2 = expensive_function(5)
    assert result2 == 10
    assert call_count == 1
    
    # Different argument should compute
    result3 = expensive_function(6)
    assert result3 == 12
    assert call_count == 2
    
    # Wait for TTL expiration
    time.sleep(1.1)
    result4 = expensive_function(5)
    assert result4 == 10
    assert call_count == 3
    
def test_cache_stats():
    """Test cache statistics."""
    cache = AdaptiveCache(CacheConfig())
    
    # Test hit/miss counting
    cache.put("key1", "value1")
    cache.get("key1")  # Hit
    cache.get("key2")  # Miss
    
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.size == 1
    assert cache.stats.avg_access_time > 0
    
def test_cache_cleanup():
    """Test cache cleanup."""
    config = CacheConfig(
        max_size=10,
        ttl_seconds=1,
        cleanup_interval=1
    )
    cache = AdaptiveCache(config)
    
    # Add items
    for i in range(5):
        cache.put(f"key{i}", f"value{i}")
        
    # Wait for expiration
    time.sleep(1.1)
    
    # Trigger cleanup
    cache.cleanup()
    
    # Timed cache should be empty
    assert len(cache.timed.cache) == 0
    
def test_thread_safety():
    """Test thread safety of caches."""
    cache = AdaptiveCache(CacheConfig())
    
    def worker():
        for i in range(100):
            cache.put(f"key{i}", f"value{i}")
            cache.get(f"key{i}")
            
    threads = [
        threading.Thread(target=worker)
        for _ in range(4)
    ]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    # No exceptions should have been raised
    
def test_cache_memory_adaptation():
    """Test cache size adaptation based on memory usage."""
    config = CacheConfig(
        max_size=1000,
        memory_threshold=0.8
    )
    cache = AdaptiveCache(config)
    
    # Simulate high hit rate but high memory usage
    cache.stats.hits = 800
    cache.stats.misses = 100
    cache.stats.memory_usage = 0.9
    
    # Should not increase size due to memory
    original_size = cache.lru.max_size
    cache.cleanup()
    assert cache.lru.max_size == original_size
    
def test_cache_hit_rate_adaptation():
    """Test cache size adaptation based on hit rate."""
    config = CacheConfig(max_size=1000)
    cache = AdaptiveCache(config)
    
    # Simulate low hit rate
    cache.stats.hits = 100
    cache.stats.misses = 900
    
    # Should decrease size
    original_size = cache.lru.max_size
    cache.cleanup()
    assert cache.lru.max_size < original_size
    
@pytest.mark.asyncio
async def test_async_cache_access():
    """Test cache access from async code."""
    cache = AdaptiveCache(CacheConfig())
    
    async def worker():
        for i in range(100):
            cache.put(f"key{i}", f"value{i}")
            assert cache.get(f"key{i}") == f"value{i}"
            
    await worker() 