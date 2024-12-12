"""Tests for resource manager."""

import time
import pytest
from unittest.mock import patch, MagicMock
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

from ..core.resource_manager import (
    ResourceManager,
    ResourcePool,
    ResourceMetrics,
    ResourceLimits
)

@pytest.fixture
def resource_manager():
    """Create a resource manager for testing."""
    limits = ResourceLimits(
        max_memory_percent=80.0,
        max_cpu_percent=90.0,
        max_threads=100,
        max_handles=1000,
        max_pool_size=50
    )
    return ResourceManager(limits=limits)

def test_resource_pool():
    """Test resource pool functionality."""
    pool = ResourcePool(max_size=2)
    
    # Test adding resources
    pool.release("resource1")
    pool.release("resource2")
    pool.release("resource3")  # Should be discarded
    
    # Test acquiring resources
    assert pool.acquire() == "resource1"
    assert pool.acquire() == "resource2"
    assert pool.acquire() is None
    
def test_resource_metrics():
    """Test resource metrics collection."""
    with patch('psutil.Process') as mock_process:
        process = MagicMock()
        process.cpu_percent.return_value = 50.0
        process.memory_percent.return_value = 60.0
        process.io_counters.return_value = MagicMock(_asdict=lambda: {'read_bytes': 1000})
        process.num_threads.return_value = 10
        process.num_handles.return_value = 100
        mock_process.return_value = process
        
        manager = ResourceManager()
        metrics = manager.get_metrics()
        
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.io_counters['read_bytes'] == 1000
        assert metrics.thread_count == 10
        assert metrics.open_handles == 100
        
def test_high_memory_handling():
    """Test handling of high memory usage."""
    with patch('psutil.Process') as mock_process:
        process = MagicMock()
        process.memory_percent.return_value = 90.0
        mock_process.return_value = process
        
        manager = ResourceManager()
        pool = manager.create_pool("test", 10)
        pool.release("resource")
        
        # Trigger high memory handling
        manager._handle_high_memory()
        
        # Pool should be cleared
        assert pool.acquire() is None
        
def test_thread_pool_execution():
    """Test thread pool task execution."""
    manager = ResourceManager()
    
    def task():
        time.sleep(0.1)
        return 42
        
    future = manager.submit_task(task)
    result = future.result()
    
    assert result == 42
    
def test_resource_cleanup():
    """Test resource cleanup."""
    manager = ResourceManager()
    pool = manager.create_pool("test", 10)
    pool.release("resource")
    
    with patch.object(ThreadPoolExecutor, 'shutdown') as mock_shutdown:
        manager.cleanup()
        
        # Pool should be cleared
        assert pool.acquire() is None
        # Thread pool should be shutdown
        assert mock_shutdown.called
        
def test_pool_management():
    """Test pool management."""
    manager = ResourceManager()
    
    # Create pool
    pool1 = manager.create_pool("pool1", 10)
    assert manager.get_pool("pool1") == pool1
    
    # Try to create duplicate pool
    with pytest.raises(ValueError):
        manager.create_pool("pool1", 10)
        
    # Get non-existent pool
    assert manager.get_pool("nonexistent") is None
    
@pytest.mark.asyncio
async def test_high_cpu_handling():
    """Test handling of high CPU usage."""
    with patch('psutil.Process') as mock_process:
        process = MagicMock()
        process.cpu_percent.return_value = 95.0
        mock_process.return_value = process
        
        manager = ResourceManager()
        original_workers = manager.executor._max_workers
        
        # Trigger high CPU handling
        manager._handle_high_cpu()
        
        # Thread pool size should be reduced
        assert manager.executor._max_workers == original_workers // 2
        
def test_resource_limits():
    """Test resource limits configuration."""
    limits = ResourceLimits(
        max_memory_percent=70.0,
        max_cpu_percent=80.0,
        max_threads=50,
        max_handles=500,
        max_pool_size=25
    )
    
    manager = ResourceManager(limits=limits)
    
    assert manager.limits.max_memory_percent == 70.0
    assert manager.limits.max_cpu_percent == 80.0
    assert manager.limits.max_threads == 50
    assert manager.limits.max_handles == 500
    assert manager.limits.max_pool_size == 25
    
def test_monitoring_thread():
    """Test resource monitoring thread."""
    with patch('threading.Thread') as mock_thread:
        manager = ResourceManager()
        
        # Monitor thread should be started
        assert mock_thread.called
        call_args = mock_thread.call_args[1]
        assert call_args['target'] == manager._monitor_resources
        assert call_args['daemon'] is True 