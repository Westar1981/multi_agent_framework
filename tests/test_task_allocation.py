"""
Tests for the hierarchical task allocation system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Set
from unittest.mock import Mock

from ..core.task_allocation import (
    TaskAllocationManager,
    Task,
    TaskPriority
)
from ..agents.base_agent import BaseAgent, AgentCapability

class MockAgent(BaseAgent):
    """Mock agent for testing."""
    def __init__(self, agent_id: str, capabilities: Set[AgentCapability]):
        self.id = agent_id
        self.capabilities = capabilities
        
    async def process_message(self, message):
        pass

@pytest.fixture
def task_manager():
    """Create task manager instance for testing."""
    return TaskAllocationManager()

@pytest.fixture
def mock_agents():
    """Create mock agents with different capabilities."""
    agents = [
        MockAgent("agent1", {AgentCapability.CODE_ANALYSIS, AgentCapability.CODE_GENERATION}),
        MockAgent("agent2", {AgentCapability.CODE_ANALYSIS}),
        MockAgent("agent3", {AgentCapability.CODE_GENERATION, AgentCapability.CODE_REVIEW})
    ]
    return agents

@pytest.mark.asyncio
async def test_agent_registration(task_manager, mock_agents):
    """Test agent registration process."""
    for agent in mock_agents:
        await task_manager.register_agent(agent)
        
    assert len(task_manager.agent_capabilities) == 3
    assert task_manager.agent_workloads["agent1"] == 0.0
    assert len(task_manager.performance_history["agent1"]) == 0

@pytest.mark.asyncio
async def test_task_submission(task_manager):
    """Test task submission and queuing."""
    task = Task(
        id="task1",
        name="Test Task",
        priority=TaskPriority.HIGH,
        required_capabilities={AgentCapability.CODE_ANALYSIS},
        estimated_complexity=0.5,
        dependencies=set(),
        created_at=datetime.now()
    )
    
    success = await task_manager.submit_task(task)
    assert success is True
    
    # Test duplicate submission
    success = await task_manager.submit_task(task)
    assert success is False

@pytest.mark.asyncio
async def test_task_allocation(task_manager, mock_agents):
    """Test task allocation to agents."""
    # Register agents
    for agent in mock_agents:
        await task_manager.register_agent(agent)
        
    # Create tasks
    tasks = [
        Task(
            id="task1",
            name="Analysis Task",
            priority=TaskPriority.HIGH,
            required_capabilities={AgentCapability.CODE_ANALYSIS},
            estimated_complexity=0.5,
            dependencies=set(),
            created_at=datetime.now()
        ),
        Task(
            id="task2",
            name="Generation Task",
            priority=TaskPriority.MEDIUM,
            required_capabilities={AgentCapability.CODE_GENERATION},
            estimated_complexity=0.3,
            dependencies=set(),
            created_at=datetime.now()
        )
    ]
    
    # Submit tasks
    for task in tasks:
        await task_manager.submit_task(task)
        
    # Allocate tasks
    allocations = await task_manager.allocate_tasks()
    assert len(allocations) == 2
    
    # Verify allocations
    task1_agent = task_manager.tasks["task1"].assigned_agent
    assert task1_agent in ["agent1", "agent2"]  # Both have CODE_ANALYSIS
    
    task2_agent = task_manager.tasks["task2"].assigned_agent
    assert task2_agent in ["agent1", "agent3"]  # Both have CODE_GENERATION

@pytest.mark.asyncio
async def test_task_dependencies(task_manager, mock_agents):
    """Test handling of task dependencies."""
    # Register agent
    await task_manager.register_agent(mock_agents[0])
    
    # Create dependent tasks
    task1 = Task(
        id="task1",
        name="First Task",
        priority=TaskPriority.HIGH,
        required_capabilities={AgentCapability.CODE_ANALYSIS},
        estimated_complexity=0.5,
        dependencies=set(),
        created_at=datetime.now()
    )
    
    task2 = Task(
        id="task2",
        name="Dependent Task",
        priority=TaskPriority.HIGH,
        required_capabilities={AgentCapability.CODE_ANALYSIS},
        estimated_complexity=0.3,
        dependencies={"task1"},
        created_at=datetime.now()
    )
    
    # Submit tasks
    await task_manager.submit_task(task1)
    await task_manager.submit_task(task2)
    
    # First allocation - only task1 should be allocated
    allocations = await task_manager.allocate_tasks()
    assert len(allocations) == 1
    assert allocations[0][0] == "task1"
    
    # Complete task1
    await task_manager.update_task_progress("task1", 1.0)
    
    # Second allocation - task2 should now be allocated
    allocations = await task_manager.allocate_tasks()
    assert len(allocations) == 1
    assert allocations[0][0] == "task2"

@pytest.mark.asyncio
async def test_workload_optimization(task_manager, mock_agents):
    """Test workload optimization and task reallocation."""
    # Register agents
    for agent in mock_agents:
        await task_manager.register_agent(agent)
        
    # Create tasks to overload agent1
    tasks = []
    for i in range(5):
        task = Task(
            id=f"task{i}",
            name=f"Task {i}",
            priority=TaskPriority.MEDIUM,
            required_capabilities={AgentCapability.CODE_ANALYSIS},
            estimated_complexity=0.2,  # Each task is 20% of capacity
            dependencies=set(),
            created_at=datetime.now()
        )
        tasks.append(task)
        await task_manager.submit_task(task)
        
    # Initial allocation
    await task_manager.allocate_tasks()
    
    # Update progress to create uneven workload
    for i in range(3):
        await task_manager.update_task_progress(f"task{i}", 0.5)
        
    # Optimize allocations
    await task_manager.optimize_allocations()
    
    # Verify workload balancing
    workloads = task_manager.agent_workloads
    assert all(workload <= 0.8 for workload in workloads.values())

@pytest.mark.asyncio
async def test_performance_tracking(task_manager, mock_agents):
    """Test agent performance tracking and scoring."""
    # Register agent
    await task_manager.register_agent(mock_agents[0])
    
    # Create and submit task
    task = Task(
        id="task1",
        name="Test Task",
        priority=TaskPriority.MEDIUM,
        required_capabilities={AgentCapability.CODE_ANALYSIS},
        estimated_complexity=0.5,
        dependencies=set(),
        created_at=datetime.now(),
        deadline=datetime.now() + timedelta(hours=1)
    )
    await task_manager.submit_task(task)
    
    # Allocate and complete task
    await task_manager.allocate_tasks()
    await task_manager.update_task_progress("task1", 1.0)
    
    # Verify performance tracking
    agent_id = mock_agents[0].id
    assert len(task_manager.performance_history[agent_id]) == 1
    assert task_manager.performance_history[agent_id][0] == 1.0  # Completed on time

@pytest.mark.asyncio
async def test_deadline_handling(task_manager, mock_agents):
    """Test handling of task deadlines."""
    # Register agent
    await task_manager.register_agent(mock_agents[0])
    
    # Create task with tight deadline
    task = Task(
        id="task1",
        name="Urgent Task",
        priority=TaskPriority.HIGH,
        required_capabilities={AgentCapability.CODE_ANALYSIS},
        estimated_complexity=0.5,
        dependencies=set(),
        created_at=datetime.now(),
        deadline=datetime.now() - timedelta(minutes=1)  # Already passed
    )
    
    await task_manager.submit_task(task)
    await task_manager.allocate_tasks()
    await task_manager.update_task_progress("task1", 1.0)
    
    # Verify performance penalty for missing deadline
    agent_id = mock_agents[0].id
    assert task_manager.performance_history[agent_id][0] == 0.5  # Penalty score 