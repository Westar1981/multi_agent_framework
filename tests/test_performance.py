"""
Performance test suite for the multi-agent framework.
"""

import pytest
import asyncio
import psutil
import os
import time
import numpy as np
from typing import Dict, List, Any
import torch

from ..core.framework import Framework
from ..core.coordinator import Coordinator
from ..agents.neural_symbolic_agent import NeuralSymbolicAgent
from ..agents.prolog_reasoner import PrologReasoner
from ..agents.meta_reasoner import MetaReasoner
from ..core.optimization import OptimizationConfig

class TestMemoryManagement:
    """Test memory management and resource utilization."""
    
    @pytest.fixture
    async def framework(self):
        """Create framework instance."""
        framework = Framework()
        await framework.initialize()
        yield framework
        await framework.shutdown()
        
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, framework):
        """Test for memory leaks during extended operation."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        memory_samples = []
        
        # Run intensive operations
        for _ in range(10):
            # Create and process messages
            for i in range(100):
                message = {
                    'type': 'test',
                    'data': f"Sample data {i}" * 100  # Create significant data
                }
                await framework.process_message(message)
                
            # Record memory usage
            current_memory = process.memory_info().rss
            memory_samples.append(current_memory)
            
            # Allow GC to run
            await asyncio.sleep(0.1)
            
        # Calculate memory growth
        memory_growth = np.diff(memory_samples)
        avg_growth = np.mean(memory_growth)
        
        # Memory growth should stabilize
        assert avg_growth < 1024 * 1024  # Less than 1MB average growth
        
    @pytest.mark.asyncio
    async def test_cache_efficiency(self, framework):
        """Test cache hit rates and memory usage."""
        agent = NeuralSymbolicAgent(
            name="test_agent",
            capabilities=["test"],
            optimizer_config=OptimizationConfig(
                cache_size=1000,
                batch_size=32
            )
        )
        
        # Generate repeated queries
        queries = ["test_query_" + str(i % 10) for i in range(100)]
        hit_count = 0
        
        for query in queries:
            result = await agent.process_message({
                'type': 'query',
                'content': query
            })
            
            if agent.reasoner.cache_hit(query):
                hit_count += 1
                
        # Cache hit rate should be good for repeated queries
        hit_rate = hit_count / len(queries)
        assert hit_rate > 0.7  # At least 70% cache hit rate
        
        # Cache size should be bounded
        assert len(agent.reasoner.cache) <= agent.optimizer_config.cache_size
        
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, framework):
        """Test resource cleanup after operations."""
        # Create temporary resources
        temp_agents = []
        for i in range(10):
            agent = NeuralSymbolicAgent(
                name=f"temp_agent_{i}",
                capabilities=["test"]
            )
            temp_agents.append(agent)
            framework.add_agent(agent)
            
        # Use resources
        for agent in temp_agents:
            await agent.process_message({
                'type': 'test',
                'content': 'test_data'
            })
            
        # Cleanup
        for agent in temp_agents:
            framework.remove_agent(agent.name)
            
        # Check resource cleanup
        process = psutil.Process(os.getpid())
        open_files = process.open_files()
        open_connections = process.connections()
        
        # Resources should be properly cleaned up
        assert len(open_files) < 100  # Reasonable number of open files
        assert len(open_connections) < 50  # Reasonable number of connections
        
class TestPerformanceOptimization:
    """Test performance optimization capabilities."""
    
    @pytest.fixture
    def coordinator(self):
        """Create coordinator instance."""
        return Coordinator()
        
    @pytest.mark.asyncio
    async def test_adaptive_batch_size(self, coordinator):
        """Test batch size adaptation under load."""
        agent = NeuralSymbolicAgent(
            name="test_agent",
            capabilities=["test"],
            optimizer_config=OptimizationConfig(
                initial_batch_size=16,
                max_batch_size=128
            )
        )
        
        coordinator.add_agent(agent)
        
        # Simulate increasing load
        batch_sizes = []
        for load_factor in np.linspace(0.1, 1.0, 10):
            # Generate batch of requests
            requests = [
                {'type': 'test', 'content': f'data_{i}'}
                for i in range(int(100 * load_factor))
            ]
            
            # Process requests
            start_time = time.time()
            for request in requests:
                await agent.process_message(request)
            processing_time = time.time() - start_time
            
            # Record batch size
            batch_sizes.append(agent.optimizer_config.batch_size)
            
            # Allow optimization to occur
            await asyncio.sleep(0.1)
            
        # Batch size should adapt to load
        assert batch_sizes[-1] > batch_sizes[0]  # Should increase under load
        assert all(size <= agent.optimizer_config.max_batch_size for size in batch_sizes)
        
    @pytest.mark.asyncio
    async def test_memory_optimization_trigger(self, coordinator):
        """Test memory optimization triggers."""
        agent = NeuralSymbolicAgent(
            name="test_agent",
            capabilities=["test"]
        )
        
        coordinator.add_agent(agent)
        
        # Fill memory until optimization triggers
        data_size = 0
        optimization_triggered = False
        
        while data_size < 1000000 and not optimization_triggered:  # 1MB limit
            # Add data to agent
            data = "x" * 1000  # 1KB of data
            await agent.process_message({
                'type': 'store',
                'content': data
            })
            
            data_size += 1000
            
            # Check if optimization was triggered
            if agent.get_optimization_status()['metrics']['memory_optimization_count'] > 0:
                optimization_triggered = True
                
        assert optimization_triggered  # Optimization should trigger before 1MB
        
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, coordinator):
        """Test concurrent processing capabilities."""
        agent = NeuralSymbolicAgent(
            name="test_agent",
            capabilities=["test"]
        )
        
        coordinator.add_agent(agent)
        
        # Create concurrent tasks
        async def process_batch(batch_id: int):
            results = []
            for i in range(10):
                result = await agent.process_message({
                    'type': 'test',
                    'content': f'batch_{batch_id}_item_{i}'
                })
                results.append(result)
            return results
            
        # Run concurrent batches
        tasks = [
            process_batch(i)
            for i in range(5)
        ]
        
        # Gather results
        results = await asyncio.gather(*tasks)
        
        # Verify all batches completed
        assert len(results) == 5
        assert all(len(batch) == 10 for batch in results)
        
        # Check resource usage after concurrent processing
        process = psutil.Process(os.getpid())
        cpu_percent = process.cpu_percent()
        memory_percent = process.memory_percent()
        
        # Resource usage should be reasonable
        assert cpu_percent < 90  # CPU usage under 90%
        assert memory_percent < 50  # Memory usage under 50% 