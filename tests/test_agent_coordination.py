"""
Tests for agent coordination and self-analysis capabilities.
"""

import pytest
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from ..core.agent_coordination import (
    AgentCoordinator,
    TaskPriority,
    CoordinationStrategy,
    AgentMessage
)
from ..core.self_analysis import (
    SelfAnalysis,
    MetricCollector,
    PatternDetector,
    OptimizationEngine
)

@dataclass
class AgentMetrics:
    """Test metrics for agent performance tracking."""
    response_time: float
    throughput: float
    error_rate: float
    resource_usage: float
    task_success_rate: float

class MockAgent:
    """Mock agent for testing coordination."""
    
    def __init__(self, name: str, specialization: str, 
                 processing_time: float = 0.1,
                 error_rate: float = 0.05):
        self.name = name
        self.specialization = specialization
        self.processing_time = processing_time
        self.error_rate = error_rate
        self.messages: List[AgentMessage] = []
        self.metrics = AgentMetrics(
            response_time=processing_time,
            throughput=1.0/processing_time,
            error_rate=error_rate,
            resource_usage=0.3,
            task_success_rate=1.0 - error_rate
        )
        
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message with simulated delay and errors."""
        await asyncio.sleep(self.processing_time)
        self.messages.append(message)
        
        if np.random.random() < self.error_rate:
            raise RuntimeError(f"Simulated error in {self.name}")
            
        return AgentMessage(
            sender=self.name,
            recipient=message.sender,
            content=f"Processed by {self.name}: {message.content}",
            priority=message.priority,
            timestamp=datetime.now()
        )

class TestAgentCoordination:
    """Test suite for agent coordination capabilities."""
    
    @pytest.fixture
    def coordinator(self):
        """Create agent coordinator with mock agents."""
        coordinator = AgentCoordinator()
        
        # Register mock agents
        agents = [
            MockAgent("code_analyzer", "analysis", processing_time=0.1),
            MockAgent("code_generator", "generation", processing_time=0.2),
            MockAgent("meta_reasoner", "reasoning", processing_time=0.15),
            MockAgent("code_transformer", "transformation", processing_time=0.25)
        ]
        
        for agent in agents:
            coordinator.register_agent(agent)
            
        return coordinator
        
    @pytest.mark.asyncio
    async def test_priority_handling(self, coordinator):
        """Test that high priority tasks are processed before lower priority ones."""
        messages = [
            AgentMessage(
                sender="test",
                recipient="code_analyzer",
                content=f"Task {i}",
                priority=TaskPriority.LOW if i % 2 == 0 else TaskPriority.HIGH,
                timestamp=datetime.now()
            )
            for i in range(10)
        ]
        
        # Submit messages in reverse priority order
        for msg in reversed(messages):
            await coordinator.submit_message(msg)
            
        # Process all messages
        results = await coordinator.process_queue()
        
        # Verify high priority messages were processed first
        high_priority_times = [
            r.timestamp for r in results 
            if r.priority == TaskPriority.HIGH
        ]
        low_priority_times = [
            r.timestamp for r in results
            if r.priority == TaskPriority.LOW
        ]
        
        assert all(ht < lt for ht in high_priority_times for lt in low_priority_times)
        
    @pytest.mark.asyncio
    async def test_load_balancing(self, coordinator):
        """Test that tasks are distributed evenly among agents."""
        # Submit many similar tasks
        num_tasks = 20
        messages = [
            AgentMessage(
                sender="test",
                recipient="code_analyzer",
                content=f"Analysis task {i}",
                priority=TaskPriority.NORMAL,
                timestamp=datetime.now()
            )
            for i in range(num_tasks)
        ]
        
        for msg in messages:
            await coordinator.submit_message(msg)
            
        results = await coordinator.process_queue()
        
        # Check distribution across agents
        agent_loads = {}
        for result in results:
            agent = result.sender
            agent_loads[agent] = agent_loads.get(agent, 0) + 1
            
        # Verify relatively even distribution
        load_values = list(agent_loads.values())
        assert max(load_values) - min(load_values) <= 2
        
    @pytest.mark.asyncio
    async def test_error_handling(self, coordinator):
        """Test graceful handling of agent errors."""
        # Create message that will trigger error
        error_msg = AgentMessage(
            sender="test",
            recipient="meta_reasoner",
            content="Task that will fail",
            priority=TaskPriority.HIGH,
            timestamp=datetime.now()
        )
        
        await coordinator.submit_message(error_msg)
        
        # Should handle error without crashing
        results = await coordinator.process_queue()
        assert len(results) == 0  # Failed task
        assert coordinator.error_count > 0
        
    @pytest.mark.asyncio
    async def test_message_routing(self, coordinator):
        """Test correct routing of messages between agents."""
        # Create chain of messages
        initial_msg = AgentMessage(
            sender="test",
            recipient="code_analyzer",
            content="Start analysis",
            priority=TaskPriority.NORMAL,
            timestamp=datetime.now()
        )
        
        await coordinator.submit_message(initial_msg)
        results = await coordinator.process_queue()
        
        # Verify message chain
        assert len(results) == 1
        assert results[0].sender == "code_analyzer"
        assert "Processed by code_analyzer" in results[0].content

class TestSelfAnalysis:
    """Test suite for self-analysis capabilities."""
    
    @pytest.fixture
    def analyzer(self):
        """Create self-analysis system with mock data."""
        return SelfAnalysis()
        
    def test_metric_collection(self, analyzer):
        """Test collection and aggregation of performance metrics."""
        # Add sample metrics
        metrics = [
            AgentMetrics(
                response_time=0.1 + i*0.01,
                throughput=100 - i*5,
                error_rate=0.05 + i*0.01,
                resource_usage=0.3 + i*0.05,
                task_success_rate=0.95 - i*0.02
            )
            for i in range(10)
        ]
        
        for metric in metrics:
            analyzer.collect_metrics(metric)
            
        # Verify metric aggregation
        summary = analyzer.get_metric_summary()
        assert 0.1 <= summary.avg_response_time <= 0.2
        assert 50 <= summary.avg_throughput <= 100
        assert summary.error_rate < 0.15
        
    def test_pattern_detection(self, analyzer):
        """Test detection of performance patterns."""
        # Simulate degrading performance pattern
        timestamps = [
            datetime.now() + timedelta(minutes=i)
            for i in range(20)
        ]
        
        response_times = [0.1 + i*0.02 for i in range(20)]
        
        for ts, rt in zip(timestamps, response_times):
            analyzer.collect_metrics(AgentMetrics(
                response_time=rt,
                throughput=100/(rt + 0.1),
                error_rate=0.05,
                resource_usage=0.3,
                task_success_rate=0.95
            ), timestamp=ts)
            
        patterns = analyzer.detect_patterns()
        
        # Should detect degrading performance
        assert any(
            p.pattern_type == "degrading_performance" 
            and p.confidence > 0.8 
            for p in patterns
        )
        
    def test_optimization_suggestions(self, analyzer):
        """Test generation of optimization suggestions."""
        # Simulate resource-intensive workload
        for _ in range(10):
            analyzer.collect_metrics(AgentMetrics(
                response_time=0.2,
                throughput=50,
                error_rate=0.05,
                resource_usage=0.8,  # High resource usage
                task_success_rate=0.95
            ))
            
        suggestions = analyzer.get_optimization_suggestions()
        
        # Should suggest resource optimization
        assert any(
            s.category == "resource_optimization"
            and s.priority == TaskPriority.HIGH
            for s in suggestions
        )
        
    def test_adaptive_thresholds(self, analyzer):
        """Test adaptation of performance thresholds."""
        # Simulate varying load conditions
        for i in range(30):
            # Simulate daily pattern
            hour = i % 24
            load_factor = 1.0 + 0.5 * np.sin(hour * np.pi / 12)
            
            analyzer.collect_metrics(AgentMetrics(
                response_time=0.1 * load_factor,
                throughput=100 / load_factor,
                error_rate=0.05,
                resource_usage=0.3 * load_factor,
                task_success_rate=0.95
            ))
            
        # Verify thresholds adapt to patterns
        thresholds = analyzer.get_current_thresholds()
        assert thresholds.response_time_threshold > 0.1  # Should adapt up
        assert thresholds.throughput_threshold < 100  # Should adapt down 

"""
Tests for agent coordination and collaboration capabilities.
"""

import pytest
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from ..core.agent_coordination import (
    AgentCoordinator,
    TaskPriority,
    CoordinationStrategy,
    AgentMessage,
    AgentCapability,
    CollaborationGroup
)

@pytest.fixture
def complex_coordinator():
    """Create coordinator with diverse agent capabilities."""
    coordinator = AgentCoordinator(strategy=CoordinationStrategy.ADAPTIVE)
    
    # Register agents with varied capabilities
    agents = [
        MockAgent("code_analyzer", "analysis", processing_time=0.1),
        MockAgent("code_generator", "generation", processing_time=0.2),
        MockAgent("meta_reasoner", "reasoning", processing_time=0.15),
        MockAgent("code_transformer", "transformation", processing_time=0.25),
        MockAgent("test_generator", "testing", processing_time=0.12),
        MockAgent("optimizer", "optimization", processing_time=0.18)
    ]
    
    # Define capabilities for each agent
    capabilities = {
        "code_analyzer": AgentCapability(
            specialization="analysis",
            expertise_level=0.9,
            supported_tasks={"code_review", "pattern_detection", "complexity_analysis"},
            performance_score=0.85
        ),
        "code_generator": AgentCapability(
            specialization="generation",
            expertise_level=0.85,
            supported_tasks={"code_generation", "refactoring", "documentation"},
            performance_score=0.8
        ),
        "meta_reasoner": AgentCapability(
            specialization="reasoning",
            expertise_level=0.95,
            supported_tasks={"optimization", "architecture_design", "pattern_detection"},
            performance_score=0.9
        ),
        "code_transformer": AgentCapability(
            specialization="transformation",
            expertise_level=0.8,
            supported_tasks={"refactoring", "optimization", "code_generation"},
            performance_score=0.75
        ),
        "test_generator": AgentCapability(
            specialization="testing",
            expertise_level=0.88,
            supported_tasks={"test_generation", "test_analysis", "coverage_analysis"},
            performance_score=0.82
        ),
        "optimizer": AgentCapability(
            specialization="optimization",
            expertise_level=0.92,
            supported_tasks={"optimization", "performance_analysis", "bottleneck_detection"},
            performance_score=0.88
        )
    }
    
    for agent in agents:
        coordinator.register_agent(agent, capabilities[agent.name])
        
    return coordinator

class TestAdvancedCoordination:
    """Test suite for advanced coordination strategies."""
    
    @pytest.mark.asyncio
    async def test_adaptive_strategy(self, complex_coordinator):
        """Test adaptive agent selection based on performance."""
        # Submit tasks and build performance history
        messages = []
        for i in range(20):
            msg = AgentMessage(
                sender="test",
                recipient="meta_reasoner",
                content=f"Task {i}",
                priority=TaskPriority.NORMAL,
                timestamp=datetime.now(),
                task_type="optimization",
                max_processing_time=0.5
            )
            messages.append(msg)
            
        # Process messages and build history
        for msg in messages[:10]:
            await complex_coordinator.submit_message(msg)
        results = await complex_coordinator.process_queue()
        
        # Verify adaptive selection
        initial_selection = set(r.sender for r in results)
        
        # Process more messages after history is built
        for msg in messages[10:]:
            await complex_coordinator.submit_message(msg)
        new_results = await complex_coordinator.process_queue()
        
        # Verify adaptation based on performance
        final_selection = set(r.sender for r in new_results)
        assert len(final_selection) <= len(initial_selection)  # Should focus on better performing agents
        
    @pytest.mark.asyncio
    async def test_auction_based_selection(self, complex_coordinator):
        """Test auction-based task allocation."""
        complex_coordinator.strategy = CoordinationStrategy.AUCTION_BASED
        
        # Create task requiring specific expertise
        message = AgentMessage(
            sender="test",
            recipient=None,  # Let auction decide
            content="Complex optimization task",
            priority=TaskPriority.HIGH,
            timestamp=datetime.now(),
            task_type="optimization"
        )
        
        await complex_coordinator.submit_message(message)
        results = await complex_coordinator.process_queue()
        
        # Verify task went to most suitable agent
        assert results[0].sender in {"optimizer", "meta_reasoner"}  # Highest expertise in optimization
        
    @pytest.mark.asyncio
    async def test_collaboration_setup(self, complex_coordinator):
        """Test formation of collaboration groups."""
        # Create task requiring collaboration
        message = AgentMessage(
            sender="test",
            recipient=None,
            content="Complex refactoring task",
            priority=TaskPriority.HIGH,
            timestamp=datetime.now(),
            task_type="refactoring",
            requires_collaboration=True
        )
        
        await complex_coordinator.submit_message(message)
        
        # Verify collaboration group formation
        assert len(complex_coordinator.collaboration_groups) > 0
        group = complex_coordinator.collaboration_groups[0]
        
        # Verify complementary capabilities
        assert group.primary_agent in {"code_transformer", "code_generator"}
        assert len(group.supporting_agents) <= 3
        assert len(set(group.supporting_agents)) == len(group.supporting_agents)  # No duplicates
        
    @pytest.mark.asyncio
    async def test_capability_based_routing(self, complex_coordinator):
        """Test routing based on agent capabilities."""
        # Create tasks requiring different capabilities
        tasks = [
            ("code_review", {"code_analyzer"}),
            ("test_generation", {"test_generator"}),
            ("optimization", {"optimizer", "meta_reasoner"}),
            ("refactoring", {"code_transformer", "code_generator"})
        ]
        
        for task_type, expected_agents in tasks:
            message = AgentMessage(
                sender="test",
                recipient=None,
                content=f"{task_type} task",
                priority=TaskPriority.NORMAL,
                timestamp=datetime.now(),
                task_type=task_type
            )
            
            await complex_coordinator.submit_message(message)
            results = await complex_coordinator.process_queue()
            
            assert results[0].sender in expected_agents
            
    @pytest.mark.asyncio
    async def test_adaptive_thresholds(self, complex_coordinator):
        """Test adaptation of performance thresholds."""
        initial_thresholds = complex_coordinator.adaptive_thresholds.copy()
        
        # Generate varying performance patterns
        for i in range(30):
            message = AgentMessage(
                sender="test",
                recipient="optimizer",
                content=f"Task {i}",
                priority=TaskPriority.NORMAL,
                timestamp=datetime.now(),
                task_type="optimization",
                max_processing_time=0.3
            )
            
            await complex_coordinator.submit_message(message)
            await complex_coordinator.process_queue()
            
        # Update thresholds
        complex_coordinator.update_adaptive_thresholds()
        
        # Verify thresholds adapted
        assert complex_coordinator.adaptive_thresholds != initial_thresholds
        assert all(0.5 <= v <= 0.95 for v in complex_coordinator.adaptive_thresholds.values())
        
    @pytest.mark.asyncio
    async def test_performance_history_impact(self, complex_coordinator):
        """Test impact of performance history on agent selection."""
        # Build performance history
        task_type = "optimization"
        messages = []
        
        # Phase 1: Equal distribution
        for i in range(10):
            msg = AgentMessage(
                sender="test",
                recipient=None,
                content=f"Task {i}",
                priority=TaskPriority.NORMAL,
                timestamp=datetime.now(),
                task_type=task_type,
                max_processing_time=0.5
            )
            messages.append(msg)
            
        # Process initial batch
        for msg in messages:
            await complex_coordinator.submit_message(msg)
        initial_results = await complex_coordinator.process_queue()
        
        # Artificially improve performance history for one agent
        target_agent = "optimizer"
        complex_coordinator.performance_history[target_agent].extend([1.0] * 5)
        
        # Phase 2: Should prefer high-performing agent
        new_messages = []
        for i in range(10):
            msg = AgentMessage(
                sender="test",
                recipient=None,
                content=f"New task {i}",
                priority=TaskPriority.NORMAL,
                timestamp=datetime.now(),
                task_type=task_type,
                max_processing_time=0.5
            )
            new_messages.append(msg)
            
        # Process second batch
        for msg in new_messages:
            await complex_coordinator.submit_message(msg)
        new_results = await complex_coordinator.process_queue()
        
        # Count assignments to high-performing agent
        initial_count = sum(1 for r in initial_results if r.sender == target_agent)
        new_count = sum(1 for r in new_results if r.sender == target_agent)
        
        # Should prefer the high-performing agent
        assert new_count > initial_count
        
    @pytest.mark.asyncio
    async def test_load_balancing_with_expertise(self, complex_coordinator):
        """Test load balancing that considers expertise levels."""
        # Submit many tasks of same type
        task_type = "optimization"
        num_tasks = 20
        
        messages = [
            AgentMessage(
                sender="test",
                recipient=None,
                content=f"Task {i}",
                priority=TaskPriority.NORMAL,
                timestamp=datetime.now(),
                task_type=task_type,
                max_processing_time=0.5
            )
            for i in range(num_tasks)
        ]
        
        # Submit all tasks
        for msg in messages:
            await complex_coordinator.submit_message(msg)
            
        results = await complex_coordinator.process_queue()
        
        # Count assignments
        assignments = {}
        for result in results:
            assignments[result.sender] = assignments.get(result.sender, 0) + 1
            
        # Verify distribution considers expertise
        optimizer_count = assignments.get("optimizer", 0)
        meta_reasoner_count = assignments.get("meta_reasoner", 0)
        
        # Higher expertise agents should get more tasks
        assert optimizer_count + meta_reasoner_count > len(results) * 0.6
        
    def test_metrics_tracking(self, complex_coordinator):
        """Test comprehensive metrics tracking."""
        # Get initial metrics
        initial_metrics = complex_coordinator.get_metrics()
        
        # Verify all expected metrics are present
        expected_metrics = {
            'messages_processed',
            'errors',
            'avg_response_time',
            'queue_size',
            'agent_loads',
            'error_rate',
            'active_collaborations',
            'agent_performance',
            'collaboration_success_rate'
        }
        
        assert all(metric in initial_metrics for metric in expected_metrics)
        
        # Verify metric types and ranges
        assert isinstance(initial_metrics['messages_processed'], int)
        assert isinstance(initial_metrics['error_rate'], float)
        assert 0 <= initial_metrics['error_rate'] <= 1
        assert isinstance(initial_metrics['agent_loads'], dict)
        assert isinstance(initial_metrics['agent_performance'], dict)