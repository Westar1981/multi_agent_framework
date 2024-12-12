"""Tests for distributed cache."""

import json
import time
import pytest
import asyncio
from unittest.mock import patch, MagicMock
import aioredis

from ..core.distributed_cache import (
    DistributedCache,
    DistributedCacheConfig
)

@pytest.fixture
async def redis_mock():
    """Create a mock Redis connection."""
    mock = MagicMock()
    mock.get = AsyncMock()
    mock.set = AsyncMock()
    mock.delete = AsyncMock()
    mock.keys = AsyncMock()
    return mock

@pytest.fixture
async def distributed_cache(redis_mock):
    """Create a distributed cache with mock Redis."""
    config = DistributedCacheConfig(
        redis_url="redis://localhost",
        namespace="test",
        replication_factor=2
    )
    
    with patch('aioredis.create_redis_pool', return_value=redis_mock):
        cache = DistributedCache(config)
        await cache._setup_tasks()
        yield cache
        await cache.shutdown()

class AsyncMock(MagicMock):
    """Mock for async functions."""
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)

@pytest.mark.asyncio
async def test_get_from_local_cache(distributed_cache):
    """Test getting value from local cache."""
    # Add to local cache
    distributed_cache.local_cache["key1"] = "value1"
    
    # Should return from local cache without Redis call
    value = await distributed_cache.get("key1")
    assert value == "value1"
    assert not distributed_cache.redis.get.called
    
@pytest.mark.asyncio
async def test_get_from_redis(distributed_cache, redis_mock):
    """Test getting value from Redis."""
    redis_mock.get.return_value = json.dumps("value1")
    
    value = await distributed_cache.get("key1")
    assert value == "value1"
    assert "key1" in distributed_cache.local_cache
    
@pytest.mark.asyncio
async def test_put_with_replication(distributed_cache):
    """Test putting value with replication."""
    await distributed_cache.put("key1", "value1")
    
    # Check Redis calls
    assert distributed_cache.redis.set.call_count == 2  # Original + 1 replica
    assert "key1" in distributed_cache.local_cache
    
@pytest.mark.asyncio
async def test_delete_with_replication(distributed_cache):
    """Test deleting value with replication."""
    # Add to local cache
    distributed_cache.local_cache["key1"] = "value1"
    
    await distributed_cache.delete("key1")
    
    # Check Redis calls
    assert distributed_cache.redis.delete.call_count == 2  # Original + 1 replica
    assert "key1" not in distributed_cache.local_cache
    
@pytest.mark.asyncio
async def test_sync_task(distributed_cache, redis_mock):
    """Test cache synchronization."""
    # Setup Redis mock
    redis_mock.keys.return_value = ["test:key1", "test:key2"]
    redis_mock.get.side_effect = [
        json.dumps("value1"),
        json.dumps("value2")
    ]
    
    # Run sync task once
    await distributed_cache._sync_task()
    
    # Check local cache updates
    assert distributed_cache.local_cache["test:key1"] == "value1"
    assert distributed_cache.local_cache["test:key2"] == "value2"
    
@pytest.mark.asyncio
async def test_backup_and_restore(distributed_cache, redis_mock):
    """Test cache backup and restore."""
    # Add some data
    distributed_cache.local_cache["key1"] = "value1"
    
    # Perform backup
    await distributed_cache._backup_task()
    
    # Check backup was saved
    backup_key = "test:backup"
    assert distributed_cache.redis.set.called_with(
        backup_key,
        json.dumps({
            "local_cache": {"key1": "value1"},
            "timestamp": pytest.approx(time.time(), abs=1)
        })
    )
    
    # Setup restore mock
    redis_mock.get.return_value = json.dumps({
        "local_cache": {"key2": "value2"},
        "timestamp": time.time()
    })
    
    # Clear and restore
    distributed_cache.local_cache.clear()
    await distributed_cache.restore_backup()
    
    assert distributed_cache.local_cache["key2"] == "value2"
    
@pytest.mark.asyncio
async def test_preload_task(distributed_cache, redis_mock):
    """Test cache preloading."""
    redis_mock.get.return_value = json.dumps("value1")
    
    # Schedule preload
    distributed_cache.schedule_preload("key1")
    
    # Wait for preload task
    await asyncio.sleep(0.1)
    
    assert "key1" in distributed_cache.local_cache
    assert distributed_cache.local_cache["key1"] == "value1"
    
@pytest.mark.asyncio
async def test_clear_cache(distributed_cache, redis_mock):
    """Test clearing cache."""
    # Add some data
    distributed_cache.local_cache["key1"] = "value1"
    redis_mock.keys.return_value = ["test:key1"]
    
    await distributed_cache.clear()
    
    assert len(distributed_cache.local_cache) == 0
    assert distributed_cache.redis.delete.called
    
@pytest.mark.asyncio
async def test_error_handling(distributed_cache, redis_mock):
    """Test error handling."""
    # Simulate Redis error
    redis_mock.get.side_effect = Exception("Redis error")
    
    # Should not raise exception
    value = await distributed_cache.get("key1")
    assert value is None
    
@pytest.mark.asyncio
async def test_concurrent_access(distributed_cache):
    """Test concurrent cache access."""
    async def worker(i):
        await distributed_cache.put(f"key{i}", f"value{i}")
        value = await distributed_cache.get(f"key{i}")
        assert value == f"value{i}"
        
    # Run multiple workers
    workers = [worker(i) for i in range(10)]
    await asyncio.gather(*workers)
    
    assert len(distributed_cache.local_cache) == 10 