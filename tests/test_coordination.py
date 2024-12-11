"""
Test suite for agent coordination and error handling.
"""

import pytest
import asyncio
from typing import Dict, List, Any
import random
import time
from unittest.mock import Mock, patch

from ..core.coordinator import Coordinator
from ..core.framework import Framework
from ..agents.neural_symbolic_agent import NeuralSymbolicAgent
from ..agents.prolog_reasoner import PrologReasoner
from ..agents.meta_reasoner import MetaReasoner
from ..core.optimization import OptimizationConfig
from ..core.agent_coordination import TaskPriority

class TestAgentCoordination:
    """Test agent coordination and collaboration."""
    
    @pytest.fixture
    async def coordinator(self):
        """Create coordinator instance with agents."""
        coordinator = Coordinator()
        
        # Add different types of agents
        agents = [
            NeuralSymbolicAgent("neural_1", ["code_analysis"]),
            NeuralSymbolicAgent("neural_2", ["code_generation"]),
            PrologReasoner("prolog_1", ["logic_reasoning"]),
            MetaReasoner("meta_1", ["optimization"])
        ]
        
        for agent in agents:
            coordinator.add_agent(agent)
            
        yield coordinator
        await coordinator.shutdown()
        
    @pytest.mark.asyncio
    async def test_task_distribution(self, coordinator):
        """Test fair task distribution among agents."""
        # Create mixed workload
        tasks = [
            {'type': 'code_analysis', 'content': f'analyze_code_{i}'}
            for i in range(20)
        ] + [
            {'type': 'code_generation', 'content': f'generate_code_{i}'}
            for i in range(20)
        ]
        
        random.shuffle(tasks)
        
        # Process tasks
        results = []
        for task in tasks:
            result = await coordinator.process_task(task)
            results.append(result)
            
        # Check workload distribution
        agent_loads = coordinator.get_agent_loads()
        loads = list(agent_loads.values())
        
        # Load should be reasonably balanced
        assert max(loads) - min(loads) < 10  # Max difference of 10 tasks
        
    @pytest.mark.asyncio
    async def test_priority_handling(self, coordinator):
        """Test handling of priority tasks."""
        # Create mixed priority tasks
        high_priority = [
            {
                'type': 'code_analysis',
                'content': f'critical_analysis_{i}',
                'priority': TaskPriority.HIGH
            }
            for i in range(5)
        ]
        
        low_priority = [
            {
                'type': 'code_analysis',
                'content': f'routine_analysis_{i}',
                'priority': TaskPriority.LOW
            }
            for i in range(15)
        ]
        
        # Mix tasks but submit high priority first
        all_tasks = high_priority + low_priority
        random.shuffle(all_tasks)
        
        start_time = time.time()
        completion_times = []
        
        for task in all_tasks:
            result = await coordinator.process_task(task)
            completion_times.append({
                'priority': task['priority'],
                'time': time.time() - start_time
            })
            
        # High priority tasks should complete first
        high_priority_times = [t['time'] for t in completion_times 
                             if t['priority'] == TaskPriority.HIGH]
        low_priority_times = [t['time'] for t in completion_times 
                            if t['priority'] == TaskPriority.LOW]
                            
        assert max(high_priority_times) < min(low_priority_times)
        
    @pytest.mark.asyncio
    async def test_error_propagation(self, coordinator):
        """Test error handling and propagation."""
        # Create failing agent
        failing_agent = NeuralSymbolicAgent(
            "failing_agent",
            ["test"],
            error_rate=0.5  # 50% failure rate
        )
        coordinator.add_agent(failing_agent)
        
        # Process batch of tasks
        tasks = [
            {'type': 'test', 'content': f'test_{i}'}
            for i in range(20)
        ]
        
        results = []
        errors = []
        
        for task in tasks:
            try:
                result = await coordinator.process_task(task)
                results.append(result)
            except Exception as e:
                errors.append(e)
                
        # Errors should be handled gracefully
        assert len(errors) > 0  # Some errors should occur
        assert len(results) + len(errors) == len(tasks)  # All tasks accounted for
        
    @pytest.mark.asyncio
    async def test_recovery_mechanism(self, coordinator):
        """Test system recovery after failures."""
        # Simulate component failure
        async def simulate_failure():
            agent = coordinator.get_agent("neural_1")
            agent.healthy = False
            await asyncio.sleep(0.1)
            agent.healthy = True
            
        # Run tasks during failure
        tasks = []
        for i in range(10):
            if i == 5:  # Trigger failure midway
                await simulate_failure()
            
            task = {'type': 'code_analysis', 'content': f'task_{i}'}
            result = await coordinator.process_task(task)
            tasks.append(result)
            
        # System should recover and complete all tasks
        assert len(tasks) == 10
        assert all(task is not None for task in tasks)
        
    @pytest.mark.asyncio
    async def test_deadlock_prevention(self, coordinator):
        """Test prevention of deadlocks in agent communication."""
        # Create circular dependency scenario
        async def agent_interaction(agent1, agent2):
            # Simulate complex interaction
            await agent1.process_message({
                'type': 'request',
                'target': agent2.name,
                'content': 'data'
            })
            
            await agent2.process_message({
                'type': 'request',
                'target': agent1.name,
                'content': 'data'
            })
            
        # Run multiple concurrent interactions
        tasks = [
            agent_interaction(
                coordinator.get_agent("neural_1"),
                coordinator.get_agent("neural_2")
            )
            for _ in range(5)
        ]
        
        # Should complete without deadlock
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
        except asyncio.TimeoutError:
            pytest.fail("Deadlock detected in agent communication")
            
    @pytest.mark.asyncio
    async def test_load_balancing(self, coordinator):
        """Test load balancing under stress."""
        # Create high load scenario
        async def generate_load(intensity: float):
            tasks = []
            for i in range(int(100 * intensity)):
                task = {
                    'type': 'code_analysis',
                    'content': f'high_load_task_{i}'
                }
                tasks.append(coordinator.process_task(task))
                
            return await asyncio.gather(*tasks)
            
        # Generate increasing load
        for intensity in [0.2, 0.4, 0.6, 0.8, 1.0]:
            start_time = time.time()
            results = await generate_load(intensity)
            duration = time.time() - start_time
            
            # Check system stability
            assert all(r is not None for r in results)
            
            # Response time should scale sub-linearly
            assert duration < (intensity * 10)  # Reasonable scaling factor
            
    @pytest.mark.asyncio
    async def test_message_ordering(self, coordinator):
        """Test preservation of message ordering when needed."""
        # Create sequence-dependent tasks
        sequence = []
        
        async def ordered_task(i: int):
            await asyncio.sleep(random.random() * 0.1)  # Random delay
            sequence.append(i)
            
        # Submit ordered tasks
        tasks = []
        for i in range(10):
            task = {
                'type': 'ordered',
                'content': f'task_{i}',
                'sequence_id': i,
                'ordered': True
            }
            tasks.append(coordinator.process_task(task))
            
        await asyncio.gather(*tasks)
        
        # Check sequence preservation
        assert sequence == list(range(10))  # Order maintained
        
    @pytest.mark.asyncio
    async def test_system_stability(self, coordinator):
        """Test system stability under various conditions."""
        # Monitor system metrics during operations
        metrics_history = []
        
        async def monitor_metrics():
            while True:
                metrics = coordinator.get_system_metrics()
                metrics_history.append(metrics)
                if len(metrics_history) >= 10:
                    break
                await asyncio.sleep(0.5)
                
        # Run monitoring in background
        monitor_task = asyncio.create_task(monitor_metrics())
        
        # Generate mixed workload
        workload_tasks = []
        for i in range(50):
            task = {
                'type': random.choice(['code_analysis', 'code_generation']),
                'content': f'stability_test_{i}',
                'priority': random.choice(list(TaskPriority))
            }
            workload_tasks.append(coordinator.process_task(task))
            
        # Run workload
        await asyncio.gather(*workload_tasks)
        await monitor_task
        
        # Analyze stability
        response_times = [m['avg_response_time'] for m in metrics_history]
        error_rates = [m['error_rate'] for m in metrics_history]
        
        # Metrics should be stable
        assert max(response_times) - min(response_times) < 1.0  # Stable response time
        assert all(rate < 0.1 for rate in error_rates)  # Low error rate 