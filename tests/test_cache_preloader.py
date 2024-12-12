"""Tests for cache preloader."""

import time
import pytest
import asyncio
from unittest.mock import Mock, patch
import numpy as np

from ..core.cache_preloader import (
    CachePreloader,
    PreloaderConfig,
    AccessPattern
)

@pytest.fixture
def preloader():
    """Create a preloader for testing."""
    config = PreloaderConfig(
        window_size=3,
        min_confidence=0.5,
        max_predictions=5,
        update_interval=0.1
    )
    preloader = CachePreloader(config)
    yield preloader
    asyncio.run(preloader.shutdown())

def test_access_pattern():
    """Test access pattern detection."""
    pattern = AccessPattern(window_size=2)
    
    # Add sequence
    pattern.add_sequence(["A", "B", "C", "B", "C"])
    
    # Test predictions
    predictions = pattern.predict_next(["B", "C"], 0.5)
    assert predictions
    assert predictions[0][0] == "B"  # Most likely next key
    
def test_record_access(preloader):
    """Test access recording."""
    preloader.record_access("key1")
    preloader.record_access("key2")
    preloader.record_access("key1")
    
    # Check history
    assert len(preloader.access_history["key1"]) == 2
    assert len(preloader.access_history["key2"]) == 1
    
    # Check sequence
    assert preloader.access_sequences == ["key1", "key2", "key1"]
    
@pytest.mark.asyncio
async def test_preload_callback(preloader):
    """Test preload callback execution."""
    callback = Mock()
    preloader.add_preload_callback(callback)
    
    # Create pattern
    for _ in range(5):
        preloader.record_access("A")
        preloader.record_access("B")
        
    # Wait for update loop
    await asyncio.sleep(0.2)
    
    # Callback should be called with predicted key
    assert callback.called
    
@pytest.mark.asyncio
async def test_cleanup(preloader):
    """Test pattern cleanup."""
    # Add many accesses
    for i in range(preloader.config.cleanup_threshold + 1):
        preloader.record_access(f"key{i}")
        
    # Wait for cleanup
    await asyncio.sleep(0.2)
    
    # History should be cleaned
    total_accesses = sum(
        len(times) for times in preloader.access_history.values()
    )
    assert total_accesses < preloader.config.cleanup_threshold
    
def test_sequence_limit(preloader):
    """Test sequence length limiting."""
    # Add many accesses
    for i in range(preloader.config.window_size * 3):
        preloader.record_access(f"key{i}")
        
    # Sequence should be limited
    assert len(preloader.access_sequences) <= preloader.config.window_size * 2
    
def test_pattern_prediction(preloader):
    """Test pattern prediction."""
    # Create clear pattern
    sequence = ["A", "B", "C"] * 5
    for key in sequence:
        preloader.record_access(key)
        
    # Get predictions
    predictions = preloader.get_predictions(["A", "B"])
    assert predictions
    assert predictions[0][0] == "C"
    assert predictions[0][1] >= preloader.config.min_confidence
    
@pytest.mark.asyncio
async def test_update_loop(preloader):
    """Test pattern update loop."""
    # Create pattern
    sequence = ["A", "B", "C"] * 5
    for key in sequence:
        preloader.record_access(key)
        
    # Wait for update
    await asyncio.sleep(0.2)
    
    # Pattern should be detected
    assert preloader.pattern.sequences
    
def test_stats(preloader):
    """Test statistics collection."""
    # Add some accesses
    for key in ["A", "B", "A", "B", "C"]:
        preloader.record_access(key)
        time.sleep(0.1)  # Space out accesses
        
    stats = preloader.get_stats()
    
    assert stats['total_patterns'] > 0
    assert stats['total_keys'] == 3
    assert stats['total_accesses'] == 5
    assert 'avg_frequency' in stats
    
@pytest.mark.asyncio
async def test_error_handling(preloader):
    """Test error handling in callbacks."""
    def failing_callback(key):
        raise Exception("Test error")
        
    preloader.add_preload_callback(failing_callback)
    
    # Add accesses
    for key in ["A", "B"] * 5:
        preloader.record_access(key)
        
    # Wait for update - should not raise
    await asyncio.sleep(0.2)
    
@pytest.mark.asyncio
async def test_concurrent_access(preloader):
    """Test concurrent access handling."""
    async def worker(keys):
        for key in keys:
            preloader.record_access(key)
            await asyncio.sleep(0.01)
            
    # Run concurrent workers
    workers = [
        worker(["A", "B"] * 5),
        worker(["C", "D"] * 5)
    ]
    await asyncio.gather(*workers)
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Should have recorded all accesses
    assert len(preloader.access_sequences) == 20
    
def test_prediction_confidence(preloader):
    """Test prediction confidence thresholds."""
    # Create pattern with noise
    sequence = ["A", "B", "C"] * 8 + ["A", "B", "D"] * 2
    for key in sequence:
        preloader.record_access(key)
        
    # Get predictions
    predictions = preloader.get_predictions(["A", "B"])
    
    # C should be predicted with higher confidence than D
    c_confidence = next(conf for key, conf in predictions if key == "C")
    d_confidence = next((conf for key, conf in predictions if key == "D"), 0)
    assert c_confidence > d_confidence
    
@pytest.mark.asyncio
async def test_shutdown(preloader):
    """Test clean shutdown."""
    # Add some data
    preloader.record_access("A")
    
    # Shutdown
    await preloader.shutdown()
    
    # Data should be cleared
    assert not preloader.access_history
    assert not preloader.access_sequences
    assert not preloader.pattern.sequences
    
    # Tasks should be cancelled
    assert preloader._cleanup_task.cancelled()
    assert preloader._update_task.cancelled() 