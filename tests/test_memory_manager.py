"""
Tests for the enhanced memory management system.
"""

import pytest
import asyncio
import time
from typing import Dict, Any
import psutil
import os
import sys
from unittest.mock import patch, MagicMock

from ..core.memory_manager import MemoryManager, MemoryConfig, CacheItem

@pytest.fixture
def memory_manager():
    """Create memory manager instance."""
    config = MemoryConfig(
        max_cache_size=1000,
        cache_ttl=60,  # 60 seconds for testing
        memory_threshold=0.8,
        cleanup_interval=5,  # 5 seconds for testing
        min_hit_rate=0.2,
        max_item_size=1024  # 1KB for testing
    )
    return MemoryManager(config)

@pytest.mark.asyncio
async def test_basic_cache_operations(memory_manager):
    """Test basic cache operations."""
    # Set item
    success = await memory_manager.set("test_key", "test_value")
    assert success is True
    
    # Get item
    value = memory_manager.get("test_key")
    assert value == "test_value"
    
    # Get metrics
    metrics = memory_manager.get_metrics()
    assert metrics['cache_hits'] == 1
    assert metrics['cache_misses'] == 0
    assert metrics['cache_size'] == 1

@pytest.mark.asyncio
async def test_ttl_expiration(memory_manager):
    """Test cache item TTL expiration."""
    # Set item
    await memory_manager.set("expire_key", "test_value")
    
    # Verify initial state
    assert memory_manager.get("expire_key") == "test_value"
    
    # Wait for TTL
    await asyncio.sleep(memory_manager.config.cache_ttl + 1)
    
    # Item should be expired
    assert memory_manager.get("expire_key") is None
    
    # Check metrics
    metrics = memory_manager.get_metrics()
    assert metrics['evictions'] == 1

@pytest.mark.asyncio
async def test_memory_pressure(memory_manager):
    """Test memory pressure handling."""
    # Create large items
    large_data = "x" * 500  # 500 bytes
    
    # Add items until memory pressure
    items_added = 0
    while psutil.Process().memory_percent() / 100.0 < memory_manager.config.memory_threshold:
        success = await memory_manager.set(f"key_{items_added}", large_data)
        if not success:
            break
        items_added += 1
        if items_added > 1000:  # Safety limit
            break
            
    # Trigger cleanup
    await memory_manager._monitor_memory()
    
    # Verify cleanup occurred
    metrics = memory_manager.get_metrics()
    assert metrics['evictions'] > 0
    assert len(memory_manager.cache) < items_added

@pytest.mark.asyncio
async def test_access_patterns(memory_manager):
    """Test cache behavior with different access patterns."""
    # Add items
    for i in range(10):
        await memory_manager.set(f"key_{i}", f"value_{i}")
        
    # Access some items frequently
    for _ in range(5):
        memory_manager.get("key_0")  # Hot item
        memory_manager.get("key_1")  # Hot item
        
    # Force cleanup
    await memory_manager.cleanup()
    
    # Hot items should be retained
    assert memory_manager.get("key_0") is not None
    assert memory_manager.get("key_1") is not None
    
    # Check hit rates
    metrics = memory_manager.get_metrics()
    assert metrics['hit_rate'] > 0.5

@pytest.mark.asyncio
async def test_size_limits(memory_manager):
    """Test handling of size limits."""
    # Try to cache too large item
    large_data = "x" * (memory_manager.config.max_item_size + 1)
    success = await memory_manager.set("large_key", large_data)
    assert success is False
    
    # Try normal sized item
    normal_data = "x" * (memory_manager.config.max_item_size // 2)
    success = await memory_manager.set("normal_key", normal_data)
    assert success is True

@pytest.mark.asyncio
async def test_cleanup_strategies(memory_manager):
    """Test different cleanup strategies."""
    # Add mix of items
    await memory_manager.set("old_unused", "value")  # Old, unused
    await memory_manager.set("new_unused", "value")  # New, unused
    await memory_manager.set("old_used", "value")    # Old, frequently used
    
    # Age some items
    memory_manager.cache["old_unused"].created_at -= memory_manager.config.cache_ttl * 0.8
    memory_manager.cache["old_used"].created_at -= memory_manager.config.cache_ttl * 0.8
    
    # Use some items
    for _ in range(5):
        memory_manager.get("old_used")
        
    # Test normal cleanup
    await memory_manager.cleanup(aggressive=False)
    assert "old_used" in memory_manager.cache  # Should keep frequently used
    assert "new_unused" in memory_manager.cache  # Should keep new
    
    # Test aggressive cleanup
    await memory_manager.cleanup(aggressive=True)
    assert len(memory_manager.cache) < 3  # Should remove more items

@pytest.mark.asyncio
async def test_optimization(memory_manager):
    """Test cache optimization."""
    # Create poor hit rate scenario
    for i in range(10):
        await memory_manager.set(f"key_{i}", f"value_{i}")
        memory_manager.get(f"key_{i}")  # One hit each
        memory_manager.get(f"missing_{i}")  # One miss each
        
    # Run optimization
    metrics = await memory_manager.optimize()
    
    # Cache size should be reduced
    assert memory_manager.config.max_cache_size < 1000
    assert metrics['hit_rate'] == 0.5

@pytest.mark.asyncio
async def test_concurrent_access(memory_manager):
    """Test concurrent cache access."""
    async def access_cache(key: str, value: str, read_count: int):
        await memory_manager.set(key, value)
        for _ in range(read_count):
            memory_manager.get(key)
            await asyncio.sleep(0.01)
            
    # Run concurrent access
    tasks = [
        access_cache(f"key_{i}", f"value_{i}", i)
        for i in range(5)
    ]
    
    await asyncio.gather(*tasks)
    
    # Verify cache consistency
    metrics = memory_manager.get_metrics()
    assert metrics['cache_size'] == 5
    assert metrics['cache_hits'] >= 10  # Sum of read_counts
    assert len(memory_manager.access_history) == 5 

@pytest.mark.asyncio
async def test_monitoring_timeout(memory_manager):
    """Test monitoring timeout handling."""
    # Mock _monitor_memory to simulate timeout
    async def slow_monitor():
        await asyncio.sleep(memory_manager.config.monitoring_timeout + 1)
        
    with patch.object(memory_manager, '_monitor_memory', side_effect=slow_monitor):
        # Start monitoring and wait briefly
        monitor_task = asyncio.create_task(memory_manager.start_monitoring())
        await asyncio.sleep(memory_manager.config.monitoring_timeout + 2)
        
        # Should still be running despite timeout
        assert not monitor_task.done()
        monitor_task.cancel()
        
@pytest.mark.asyncio
async def test_error_handling(memory_manager):
    """Test error handling in various operations."""
    # Test memory monitoring error
    with patch('psutil.Process', side_effect=Exception("Test error")):
        await memory_manager._monitor_memory()
        # Should not raise exception
        
    # Test size estimation error
    class BadObject:
        def __sizeof__(self):
            raise Exception("Size error")
            
    # Should fall back to sys.getsizeof
    size = memory_manager._estimate_size(BadObject())
    assert size > 0
    
    # Test score calculation error
    bad_item = CacheItem("bad_key", "value", 100)
    bad_item.created_at = "invalid"  # Will cause error in calculation
    
    score = memory_manager._calculate_item_score(bad_item, False)
    assert score == float('inf')  # Should return infinity for problematic items

@pytest.mark.asyncio
async def test_memory_estimation(memory_manager):
    """Test different memory estimation scenarios."""
    # Test string
    string_size = memory_manager._estimate_size("test string")
    assert string_size == len("test string".encode('utf-8'))
    
    # Test bytes
    bytes_obj = b"test bytes"
    bytes_size = memory_manager._estimate_size(bytes_obj)
    assert bytes_size == sys.getsizeof(bytes_obj)
    
    # Test custom object
    class SizedObject:
        def __sizeof__(self):
            return 1234
            
    obj_size = memory_manager._estimate_size(SizedObject())
    assert obj_size == 1234
    
@pytest.mark.asyncio
async def test_access_history_cleanup(memory_manager):
    """Test cleanup of access history for expired items."""
    # Add item and access it
    await memory_manager.set("test_key", "value")
    memory_manager.get("test_key")
    
    # Verify access history exists
    assert "test_key" in memory_manager.access_history
    assert len(memory_manager.access_history["test_key"]) == 1
    
    # Wait for TTL expiration
    await asyncio.sleep(memory_manager.config.cache_ttl + 1)
    
    # Access expired key
    result = memory_manager.get("test_key")
    assert result is None
    
    # Access history should be cleaned up
    assert "test_key" not in memory_manager.access_history