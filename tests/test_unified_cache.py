"""Tests for unified cache manager."""

import os
import time
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import aioredis

from ..core.unified_cache import (
    UnifiedCache,
    UnifiedCacheConfig
)
from ..core.distributed_cache import DistributedCacheConfig
from ..core.cache_persistence import PersistenceConfig
from ..core.cache_preloader import PreloaderConfig

@pytest.fixture
def config(tmp_path):
    """Create test configuration."""
    return UnifiedCacheConfig(
        distributed=DistributedCacheConfig(
            redis_url="redis://localhost",
            namespace="test"
        ),
        persistence=PersistenceConfig(
            db_path=str(tmp_path / "test_cache.db"),
            auto_persist=False
        ),
        preloader=PreloaderConfig(
            window_size=3,
            update_interval=0.1
        ),
        sync_interval=0.1
    )

@pytest.fixture
async def redis_mock():
    """Create Redis mock."""
    mock = AsyncMock()
    mock.get = AsyncMock()
    mock.set = AsyncMock()
    mock.delete = AsyncMock()
    mock.keys = AsyncMock(return_value=[])
    return mock

@pytest.fixture
async def unified_cache(config, redis_mock):
    """Create unified cache with mocks."""
    with patch('aioredis.create_redis_pool', return_value=redis_mock):
        cache = UnifiedCache(config)
        yield cache
        await cache.shutdown()

@pytest.mark.asyncio
async def test_get_from_distributed(unified_cache, redis_mock):
    """Test getting value from distributed cache."""
    redis_mock.get.return_value = b'"value1"'
    
    value = await unified_cache.get("key1")
    assert value == "value1"
    assert redis_mock.get.called
    
@pytest.mark.asyncio
async def test_get_from_persistence(unified_cache, redis_mock):
    """Test getting value from persistence."""
    # Miss distributed cache
    redis_mock.get.return_value = None
    
    # Add to persistence
    unified_cache.persistence.save("key1", "value1")
    
    value = await unified_cache.get("key1")
    assert value == "value1"
    
    # Should be added to distributed cache
    assert redis_mock.set.called
    
@pytest.mark.asyncio
async def test_put(unified_cache):
    """Test putting value in cache."""
    await unified_cache.put("key1", "value1")
    
    # Should be in both caches
    value = await unified_cache.distributed.get("key1")
    assert value == "value1"
    
    value = unified_cache.persistence.load("key1")
    assert value == "value1"
    
@pytest.mark.asyncio
async def test_delete(unified_cache):
    """Test deleting value from cache."""
    # Add value
    await unified_cache.put("key1", "value1")
    
    # Delete
    await unified_cache.delete("key1")
    
    # Should be gone from both caches
    value = await unified_cache.distributed.get("key1")
    assert value is None
    
    value = unified_cache.persistence.load("key1")
    assert value is None
    
@pytest.mark.asyncio
async def test_clear(unified_cache):
    """Test clearing all caches."""
    # Add values
    await unified_cache.put("key1", "value1")
    await unified_cache.put("key2", "value2")
    
    # Clear
    await unified_cache.clear()
    
    # Should be empty
    value = await unified_cache.get("key1")
    assert value is None
    value = await unified_cache.get("key2")
    assert value is None
    
@pytest.mark.asyncio
async def test_preload_integration(unified_cache):
    """Test preloader integration."""
    # Create access pattern
    for _ in range(5):
        await unified_cache.put("key1", "value1")
        await unified_cache.put("key2", "value2")
        
    # Wait for preloader update
    await asyncio.sleep(0.2)
    
    # Should have predictions
    predictions = unified_cache.preloader.get_predictions(["key1"])
    assert predictions
    assert predictions[0][0] == "key2"
    
@pytest.mark.asyncio
async def test_sync_to_persistence(unified_cache, redis_mock):
    """Test syncing distributed cache to persistence."""
    # Setup mock data
    redis_mock.keys.return_value = ["test:key1", "test:key2"]
    redis_mock.get.side_effect = [b'"value1"', b'"value2"']
    
    # Run sync
    await unified_cache._sync_to_persistence()
    
    # Check persistence
    value = unified_cache.persistence.load("test:key1")
    assert value == "value1"
    value = unified_cache.persistence.load("test:key2")
    assert value == "value2"
    
@pytest.mark.asyncio
async def test_pattern_update(unified_cache):
    """Test access pattern updates."""
    # Add some accesses
    for key in ["key1", "key2", "key1", "key2", "key3"]:
        await unified_cache.put(key, f"value_{key}")
        
    # Update patterns
    unified_cache._update_patterns()
    
    # Should have patterns
    assert unified_cache.preloader.pattern.sequences
    
@pytest.mark.asyncio
async def test_error_handling(unified_cache, redis_mock):
    """Test error handling."""
    # Simulate Redis error
    redis_mock.get.side_effect = Exception("Redis error")
    
    # Should not raise
    value = await unified_cache.get("key1")
    assert value is None
    
@pytest.mark.asyncio
async def test_stats(unified_cache):
    """Test statistics collection."""
    # Add some data
    await unified_cache.put("key1", "value1")
    await unified_cache.get("key1")
    
    stats = unified_cache.get_stats()
    
    assert 'persistence' in stats
    assert 'preloader' in stats
    assert stats['persistence']['total_entries'] > 0
    
@pytest.mark.asyncio
async def test_context_manager(config, redis_mock):
    """Test async context manager."""
    with patch('aioredis.create_redis_pool', return_value=redis_mock):
        async with UnifiedCache(config) as cache:
            await cache.put("key1", "value1")
            value = await cache.get("key1")
            assert value == "value1"
            
        # Should be shutdown
        assert cache._sync_task.cancelled()
        
@pytest.mark.asyncio
async def test_concurrent_access(unified_cache):
    """Test concurrent access."""
    async def worker(start, end):
        for i in range(start, end):
            key = f"key{i}"
            await unified_cache.put(key, f"value{i}")
            value = await unified_cache.get(key)
            assert value == f"value{i}"
            
    # Run concurrent workers
    workers = [
        worker(i*10, (i+1)*10)
        for i in range(5)
    ]
    await asyncio.gather(*workers)
    
    # Check all values
    for i in range(50):
        value = await unified_cache.get(f"key{i}")
        assert value == f"value{i}" 