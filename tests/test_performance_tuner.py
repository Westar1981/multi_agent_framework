"""Tests for performance tuner."""

import pytest
import asyncio
import numpy as np
from unittest.mock import patch, MagicMock

from ..core.performance_tuner import (
    PerformanceTuner,
    TuningConfig,
    PerformanceMetrics
)

@pytest.fixture
def config():
    """Create test configuration."""
    return TuningConfig(
        measurement_window=10,
        tuning_interval=0.1,
        max_tuning_attempts=2,
        min_improvement=0.05,
        param_bounds={
            'batch_size': (1, 100),
            'queue_size': (10, 100),
            'worker_threads': (1, 4),
            'cache_size': (100, 1000),
            'timeout': (1, 10)
        }
    )

@pytest.fixture
def tuner(config):
    """Create performance tuner."""
    tuner = PerformanceTuner(config)
    tuner.current_params = {
        'batch_size': 10,
        'queue_size': 50,
        'worker_threads': 2,
        'cache_size': 500,
        'timeout': 5
    }
    return tuner

def test_config_initialization():
    """Test configuration initialization."""
    config = TuningConfig()
    assert config.param_bounds is not None
    assert 'batch_size' in config.param_bounds
    assert 'worker_threads' in config.param_bounds
    
def test_metrics_recording(tuner):
    """Test metrics recording."""
    metrics = PerformanceMetrics(
        response_time=0.1,
        throughput=100,
        error_rate=0.01,
        resource_usage=0.5,
        memory_usage=0.4,
        cpu_usage=0.6,
        latency=0.05
    )
    
    tuner.record_metrics(metrics)
    
    for field in metrics.__dict__:
        assert len(tuner.metrics_history[field]) == 1
        
def test_score_calculation(tuner):
    """Test performance score calculation."""
    metrics = {
        'response_time': 0.1,
        'throughput': 100,
        'error_rate': 0.01,
        'resource_usage': 0.5,
        'memory_usage': 0.4,
        'cpu_usage': 0.6,
        'latency': 0.05
    }
    
    score = tuner._calculate_score(metrics)
    assert score > 0  # Good performance should have positive score
    
    # Test resource limit penalties
    metrics['memory_usage'] = 0.9  # Above limit
    penalized_score = tuner._calculate_score(metrics)
    assert penalized_score < score
    
def test_metrics_prediction(tuner):
    """Test metrics prediction."""
    current_metrics = {
        'response_time': 0.1,
        'throughput': 100,
        'error_rate': 0.01,
        'resource_usage': 0.5,
        'memory_usage': 0.4,
        'cpu_usage': 0.6,
        'latency': 0.05
    }
    
    new_params = tuner.current_params.copy()
    new_params['batch_size'] *= 2
    
    predicted = tuner._predict_metrics(new_params, current_metrics)
    
    # Batch size increase should improve throughput
    assert predicted['throughput'] > current_metrics['throughput']
    # But also increase latency
    assert predicted['latency'] > current_metrics['latency']
    
@pytest.mark.asyncio
async def test_parameter_optimization(tuner):
    """Test parameter optimization."""
    # Add some metrics
    metrics = PerformanceMetrics(
        response_time=0.1,
        throughput=100,
        error_rate=0.01,
        resource_usage=0.5,
        memory_usage=0.4,
        cpu_usage=0.6,
        latency=0.05
    )
    
    for _ in range(5):
        tuner.record_metrics(metrics)
        
    result = await tuner._optimize_parameters(tuner._get_current_metrics())
    assert result.success
    
    new_params = tuner._result_to_params(result.x)
    assert all(
        tuner.config.param_bounds[name][0] <= value <= tuner.config.param_bounds[name][1]
        for name, value in new_params.items()
    )
    
@pytest.mark.asyncio
async def test_tuning_loop(tuner):
    """Test tuning loop execution."""
    # Add metrics
    metrics = PerformanceMetrics(
        response_time=0.1,
        throughput=100,
        error_rate=0.01,
        resource_usage=0.5,
        memory_usage=0.4,
        cpu_usage=0.6,
        latency=0.05
    )
    
    for _ in range(5):
        tuner.record_metrics(metrics)
        
    # Run one tuning iteration
    await tuner._tune_performance()
    
    assert tuner.tuning_attempts == 1
    assert tuner.best_score != float('-inf')
    
@pytest.mark.asyncio
async def test_parameter_application(tuner):
    """Test parameter application."""
    new_params = tuner.current_params.copy()
    new_params['batch_size'] = 20
    
    await tuner._apply_parameters(new_params)
    assert tuner.current_params['batch_size'] == 20
    
def test_stats_collection(tuner):
    """Test statistics collection."""
    # Add metrics
    metrics = PerformanceMetrics(
        response_time=0.1,
        throughput=100,
        error_rate=0.01,
        resource_usage=0.5,
        memory_usage=0.4,
        cpu_usage=0.6,
        latency=0.05
    )
    
    for _ in range(5):
        tuner.record_metrics(metrics)
        
    stats = tuner.get_stats()
    
    assert 'current_params' in stats
    assert 'metrics' in stats
    assert all(
        'mean' in metric_stats and 'std' in metric_stats
        for metric_stats in stats['metrics'].values()
    )
    
@pytest.mark.asyncio
async def test_shutdown(tuner):
    """Test clean shutdown."""
    # Add some data
    metrics = PerformanceMetrics(
        response_time=0.1,
        throughput=100,
        error_rate=0.01,
        resource_usage=0.5,
        memory_usage=0.4,
        cpu_usage=0.6,
        latency=0.05
    )
    tuner.record_metrics(metrics)
    
    await tuner.shutdown()
    
    # Should be cleared
    assert all(len(queue) == 0 for queue in tuner.metrics_history.values())
    assert not tuner.current_params
    assert not tuner.best_params
    assert tuner._tuning_task.cancelled()
    
@pytest.mark.asyncio
async def test_concurrent_access(tuner):
    """Test concurrent access to tuner."""
    async def worker(i):
        metrics = PerformanceMetrics(
            response_time=0.1 * (i + 1),
            throughput=100 / (i + 1),
            error_rate=0.01,
            resource_usage=0.5,
            memory_usage=0.4,
            cpu_usage=0.6,
            latency=0.05
        )
        tuner.record_metrics(metrics)
        
    # Run concurrent workers
    workers = [worker(i) for i in range(5)]
    await asyncio.gather(*workers)
    
    # Should have recorded all metrics
    assert all(len(queue) == 5 for queue in tuner.metrics_history.values()) 