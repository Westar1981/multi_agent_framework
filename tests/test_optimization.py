"""
Test suite for optimization and monitoring capabilities.
"""

import pytest
import asyncio
import time
import psutil
import os
from typing import Dict, Any
import numpy as np
from unittest.mock import patch, MagicMock

from ..core.framework import Framework
from ..agents.neural_symbolic_agent import NeuralSymbolicAgent
from ..agents.meta_reasoner import MetaReasoner
from ..core.self_analysis import SelfAnalysis
from ..core.optimization import OptimizationConfig, OptimizationManager

class TestSystemOptimization:
    """Test system optimization capabilities."""
    
    @pytest.fixture
    async def framework(self):
        """Create framework instance with optimization enabled."""
        framework = Framework()
        await framework.initialize()
        
        # Add agents
        agents = [
            NeuralSymbolicAgent("neural_1", ["code_analysis"]),
            MetaReasoner("meta_1", ["optimization"])
        ]
        
        for agent in agents:
            framework.add_agent(agent)
            
        yield framework
        await framework.shutdown()
        
    @pytest.mark.asyncio
    async def test_memory_optimization(self, framework):
        """Test memory usage optimization."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Generate load
        for i in range(100):
            await framework.process_message({
                'type': 'test',
                'content': f"data_{i}" * 100  # Create significant data
            })
            
            if i % 10 == 0:
                # Allow optimization to occur
                await asyncio.sleep(0.1)
                
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory growth should be controlled
        assert memory_increase < 50  # Less than 50MB increase
        
    @pytest.mark.asyncio
    async def test_performance_optimization(self, framework):
        """Test performance optimization under load."""
        # Track processing times
        times = []
        
        for batch_size in [10, 20, 50, 100]:
            start_time = time.time()
            
            # Process batch
            tasks = []
            for i in range(batch_size):
                task = framework.process_message({
                    'type': 'test',
                    'content': f'task_{i}'
                })
                tasks.append(task)
                
            await asyncio.gather(*tasks)
            duration = time.time() - start_time
            times.append(duration / batch_size)  # Time per task
            
        # Performance should improve (lower time per task)
        assert min(times) == times[-1]
        
    @pytest.mark.asyncio
    async def test_cache_optimization(self, framework):
        """Test cache optimization."""
        agent = framework.get_agent("neural_1")
        
        # Repeated queries
        queries = ["query_" + str(i % 5) for i in range(20)]
        times = []
        
        for query in queries:
            start_time = time.time()
            await agent.process_message({
                'type': 'query',
                'content': query
            })
            times.append(time.time() - start_time)
            
        # Response time should improve for repeated queries
        avg_initial = sum(times[:5]) / 5
        avg_final = sum(times[-5:]) / 5
        assert avg_final < avg_initial
        
    @pytest.mark.asyncio
    async def test_adaptive_batch_size(self, framework):
        """Test adaptation of batch size under load."""
        agent = framework.get_agent("neural_1")
        initial_batch_size = agent.optimizer_config.batch_size
        
        # Generate increasing load
        for intensity in [0.2, 0.4, 0.6, 0.8, 1.0]:
            tasks = []
            for i in range(int(100 * intensity)):
                task = agent.process_message({
                    'type': 'test',
                    'content': f'high_load_task_{i}'
                })
                tasks.append(task)
                
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.1)  # Allow optimization
            
        # Batch size should adapt to load
        assert agent.optimizer_config.batch_size > initial_batch_size
        
    @pytest.mark.asyncio
    async def test_error_recovery(self, framework):
        """Test recovery from optimization errors."""
        agent = framework.get_agent("neural_1")
        
        # Force optimization error
        agent.optimizer_config.cache_size = -1  # Invalid configuration
        
        # Should still process messages
        result = await agent.process_message({
            'type': 'test',
            'content': 'test_data'
        })
        
        assert result is not None
        
class TestSystemMonitoring:
    """Test system monitoring capabilities."""
    
    @pytest.fixture
    def analyzer(self):
        """Create self-analysis instance."""
        return SelfAnalysis()
        
    def test_metric_collection(self, analyzer):
        """Test collection of system metrics."""
        # Add test metrics
        for i in range(5):
            analyzer.add_metric('response_time', 0.1 * i)
            analyzer.add_metric('error_rate', 0.01 * i)
            analyzer.add_metric('memory_usage', 0.5 + 0.1 * i)
            
        # Get summary
        summary = analyzer.get_metric_summary()
        
        # Verify metrics
        assert len(analyzer.get_metric_history('response_time')) == 5
        assert 'response_time' in summary
        assert 'error_rate' in summary
        assert 'memory_usage' in summary
        
    def test_threshold_detection(self, analyzer):
        """Test threshold violation detection."""
        violations = []
        
        def mock_handler(metric: str, value: float):
            violations.append((metric, value))
            
        analyzer.set_threshold_handler(mock_handler)
        
        # Add metrics with violation
        analyzer.add_metric('response_time', 2.0)  # High response time
        analyzer.add_metric('error_rate', 0.01)
        analyzer.add_metric('memory_usage', 0.9)  # High memory usage
        
        # Should detect violations
        assert len(violations) == 2
        assert any(v[0] == 'response_time' for v in violations)
        assert any(v[0] == 'memory_usage' for v in violations)
        
    @pytest.mark.asyncio
    async def test_continuous_monitoring(self, analyzer):
        """Test continuous monitoring."""
        events = []
        
        def mock_handler(event: Dict[str, Any]):
            events.append(event)
            
        analyzer.set_event_handler(mock_handler)
        
        # Generate metrics over time
        for i in range(5):
            analyzer.add_metric('throughput', 100 - i * 10)  # Declining throughput
            analyzer.add_metric('error_rate', 0.01)
            await asyncio.sleep(0.1)
            
        # Should detect throughput decline
        assert any(
            event['type'] == 'throughput_decline'
            for event in events
        )
        
    def test_metric_aggregation(self, analyzer):
        """Test metric aggregation over time windows."""
        # Add metrics over time
        for i in range(10):
            analyzer.add_metric('response_time', 0.1 * (i % 3))  # Cyclic pattern
            analyzer.add_metric('error_rate', 0.01)
            
        # Get aggregated metrics
        hourly = analyzer.get_aggregated_metrics(window='1h')
        daily = analyzer.get_aggregated_metrics(window='1d')
        
        assert 'response_time' in hourly
        assert 'error_rate' in daily
        assert len(hourly['response_time']) <= len(daily['response_time'])
        
    def test_pattern_detection(self, analyzer):
        """Test detection of metric patterns."""
        # Generate pattern
        for i in range(20):
            # Sine wave pattern
            value = 0.5 + 0.4 * np.sin(i * np.pi / 5)
            analyzer.add_metric('cpu_usage', value)
            
        patterns = analyzer.detect_patterns()
        
        # Should detect cyclic pattern
        assert any(
            p['type'] == 'cyclic' and p['metric'] == 'cpu_usage'
            for p in patterns
        )
        
    @pytest.mark.asyncio
    async def test_alert_correlation(self, analyzer):
        """Test correlation of multiple alerts."""
        alerts = []
        
        def mock_alert_handler(alert: Dict[str, Any]):
            alerts.append(alert)
            
        analyzer.set_alert_handler(mock_alert_handler)
        
        # Generate correlated alerts
        await asyncio.gather(
            analyzer.add_metric('cpu_usage', 0.95),
            analyzer.add_metric('memory_usage', 0.9),
            analyzer.add_metric('response_time', 1.5)
        )
        
        # Should detect system stress
        assert any(
            alert['type'] == 'system_stress'
            for alert in alerts
        )

"""Tests for optimization manager."""

@pytest.fixture
def config():
    """Create test configuration."""
    return OptimizationConfig(
        memory_threshold=0.7,
        cpu_threshold=0.6,
        error_rate_threshold=0.05,
        optimization_interval=60,
        max_concurrent_optimizations=2,
        enable_auto_optimization=True
    )

@pytest.fixture
def manager(config):
    """Create test optimization manager."""
    return OptimizationManager(config)

@pytest.mark.asyncio
async def test_init(manager, config):
    """Test initialization."""
    assert manager.config == config
    assert manager.running_optimizations == []
    assert not manager.is_running
    assert manager._optimization_lock is not None

@pytest.mark.asyncio
async def test_optimize(manager):
    """Test optimization run."""
    with patch.object(manager, '_run_optimization') as mock_run:
        mock_run.return_value = None
        await manager.optimize()
        assert mock_run.called

@pytest.mark.asyncio
async def test_max_concurrent_optimizations(manager):
    """Test maximum concurrent optimizations."""
    # Create mock tasks
    mock_tasks = [
        asyncio.create_task(asyncio.sleep(0.1))
        for _ in range(manager.config.max_concurrent_optimizations)
    ]
    manager.running_optimizations.extend(mock_tasks)
    
    # Try to run another optimization
    await manager.optimize()
    
    # Should not have added another task
    assert len(manager.running_optimizations) == manager.config.max_concurrent_optimizations
    
    # Cleanup
    for task in mock_tasks:
        task.cancel()
    await asyncio.gather(*mock_tasks, return_exceptions=True)

@pytest.mark.asyncio
async def test_optimization_steps(manager):
    """Test optimization steps are called."""
    with patch.object(manager, '_optimize_memory_usage') as mock_memory, \
         patch.object(manager, '_optimize_cpu_usage') as mock_cpu, \
         patch.object(manager, '_optimize_error_handling') as mock_error:
             
        await manager._run_optimization()
        
        assert mock_memory.called
        assert mock_cpu.called
        assert mock_error.called

@pytest.mark.asyncio
async def test_shutdown(manager):
    """Test manager shutdown."""
    # Create mock tasks
    mock_tasks = [
        asyncio.create_task(asyncio.sleep(0.1))
        for _ in range(2)
    ]
    manager.running_optimizations.extend(mock_tasks)
    
    # Shutdown
    await manager.shutdown()
    
    assert not manager.is_running
    assert not manager.running_optimizations
    
    # Verify tasks were cancelled
    for task in mock_tasks:
        assert task.cancelled()

@pytest.mark.asyncio
async def test_optimization_error_handling(manager):
    """Test error handling during optimization."""
    with patch.object(manager, '_run_optimization', side_effect=Exception("Test error")):
        await manager.optimize()
        assert not manager.running_optimizations  # Should have cleaned up task

@pytest.mark.asyncio
async def test_concurrent_optimizations(manager):
    """Test concurrent optimization runs."""
    async def mock_optimization():
        await asyncio.sleep(0.1)
        
    with patch.object(manager, '_run_optimization', side_effect=mock_optimization):
        # Start multiple optimizations
        tasks = [
            asyncio.create_task(manager.optimize())
            for _ in range(3)
        ]
        
        # Wait for all to complete
        await asyncio.gather(*tasks)
        
        # Should have respected max concurrent limit
        assert len(manager.running_optimizations) == 0