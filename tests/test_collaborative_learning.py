"""
Tests for the collaborative learning system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Set
from unittest.mock import Mock

from ..learning.collaborative_learning import (
    CollaborativeLearningSystem,
    Experience,
    LearningPreference
)
from ..agents.base_agent import BaseAgent, AgentCapability
from ..core.task_allocation import Task, TaskPriority

class MockAgent(BaseAgent):
    """Mock agent for testing."""
    def __init__(self, agent_id: str, capabilities: Set[AgentCapability]):
        self.id = agent_id
        self.capabilities = capabilities
        
    async def process_message(self, message):
        pass

@pytest.fixture
def learning_system():
    """Create learning system instance for testing."""
    return CollaborativeLearningSystem()

@pytest.fixture
def mock_agents():
    """Create mock agents with different capabilities."""
    agents = [
        MockAgent("agent1", {AgentCapability.CODE_ANALYSIS, AgentCapability.CODE_GENERATION}),
        MockAgent("agent2", {AgentCapability.CODE_ANALYSIS}),
        MockAgent("agent3", {AgentCapability.CODE_GENERATION, AgentCapability.CODE_REVIEW})
    ]
    return agents

@pytest.fixture
def learning_preferences():
    """Create learning preferences for agents."""
    preferences = [
        LearningPreference(
            preferred_tasks={"code_analysis", "code_generation"},
            expertise_areas={
                AgentCapability.CODE_ANALYSIS: 0.8,
                AgentCapability.CODE_GENERATION: 0.6
            },
            learning_rate=0.1,
            exploration_rate=0.2
        ),
        LearningPreference(
            preferred_tasks={"code_analysis"},
            expertise_areas={
                AgentCapability.CODE_ANALYSIS: 0.9
            },
            learning_rate=0.1,
            exploration_rate=0.1
        ),
        LearningPreference(
            preferred_tasks={"code_generation", "code_review"},
            expertise_areas={
                AgentCapability.CODE_GENERATION: 0.7,
                AgentCapability.CODE_REVIEW: 0.8
            },
            learning_rate=0.1,
            exploration_rate=0.15
        )
    ]
    return preferences

@pytest.mark.asyncio
async def test_agent_registration(learning_system, mock_agents, learning_preferences):
    """Test agent registration process."""
    for agent, prefs in zip(mock_agents, learning_preferences):
        await learning_system.register_agent(agent, prefs)
        
    assert len(learning_system.agent_preferences) == 3
    assert len(learning_system.similarity_matrix) == 6  # 3 agents, bidirectional pairs

@pytest.mark.asyncio
async def test_experience_recording(learning_system, mock_agents, learning_preferences):
    """Test recording and managing experiences."""
    # Register agent
    await learning_system.register_agent(mock_agents[0], learning_preferences[0])
    
    # Create and record experience
    experience = Experience(
        agent_id="agent1",
        task_id="task1",
        task_type="code_analysis",
        capabilities_used={AgentCapability.CODE_ANALYSIS},
        context={"file_type": "python"},
        actions_taken=["analyze_imports", "check_style"],
        outcome=0.9,
        timestamp=datetime.now(),
        execution_time=1.5,
        resource_usage={"memory": 0.2, "cpu": 0.3}
    )
    
    await learning_system.record_experience(experience)
    
    assert len(learning_system.experiences["agent1"]) == 1
    assert len(learning_system.performance_history["agent1"]) == 1
    assert learning_system.performance_history["agent1"][0] == 0.9

@pytest.mark.asyncio
async def test_recommendation_system(learning_system, mock_agents, learning_preferences):
    """Test action recommendation system."""
    # Register agents and record experiences
    for agent, prefs in zip(mock_agents[:2], learning_preferences[:2]):
        await learning_system.register_agent(agent, prefs)
        
    # Create similar experiences
    for i in range(5):
        experience = Experience(
            agent_id="agent1",
            task_id=f"task{i}",
            task_type="code_analysis",
            capabilities_used={AgentCapability.CODE_ANALYSIS},
            context={"file_type": "python"},
            actions_taken=["analyze_imports", "check_style"],
            outcome=0.8 + i * 0.02,  # Increasing success
            timestamp=datetime.now() - timedelta(hours=i),
            execution_time=1.5,
            resource_usage={"complexity": 0.5}
        )
        await learning_system.record_experience(experience)
        
    # Create test task
    task = Task(
        id="new_task",
        name="Test Analysis",
        priority=TaskPriority.MEDIUM,
        required_capabilities={AgentCapability.CODE_ANALYSIS},
        estimated_complexity=0.5,
        dependencies=set(),
        created_at=datetime.now()
    )
    
    # Get recommendations
    recommendations = await learning_system.get_recommendations("agent1", task)
    
    assert len(recommendations) > 0
    assert recommendations[0][0] in ["analyze_imports", "check_style"]
    assert 0.0 <= recommendations[0][1] <= 1.0

@pytest.mark.asyncio
async def test_preference_updates(learning_system, mock_agents, learning_preferences):
    """Test updating agent preferences based on performance."""
    # Register agent
    agent = mock_agents[0]
    prefs = learning_preferences[0]
    await learning_system.register_agent(agent, prefs)
    
    # Record successful experiences
    for i in range(5):
        await learning_system.update_preferences(
            agent.id,
            task_outcome=0.9,
            task_type="code_analysis"
        )
        
    # Verify preference updates
    updated_prefs = learning_system.agent_preferences[agent.id]
    assert "code_analysis" in updated_prefs.preferred_tasks
    assert updated_prefs.learning_rate < prefs.learning_rate  # Should decrease
    assert updated_prefs.exploration_rate < prefs.exploration_rate  # Should decrease

@pytest.mark.asyncio
async def test_learning_insights(learning_system, mock_agents, learning_preferences):
    """Test generation of learning insights."""
    # Register agent and record varied experiences
    agent = mock_agents[0]
    await learning_system.register_agent(agent, learning_preferences[0])
    
    # Record mixed performance experiences
    experiences = [
        ("code_analysis", 0.9),
        ("code_analysis", 0.8),
        ("code_analysis", 0.9),
        ("code_generation", 0.5),
        ("code_generation", 0.4),
        ("code_review", 0.3)
    ]
    
    for task_type, outcome in experiences:
        experience = Experience(
            agent_id=agent.id,
            task_id=f"task_{task_type}_{outcome}",
            task_type=task_type,
            capabilities_used={AgentCapability.CODE_ANALYSIS},
            context={},
            actions_taken=["action1"],
            outcome=outcome,
            timestamp=datetime.now(),
            execution_time=1.0,
            resource_usage={}
        )
        await learning_system.record_experience(experience)
        
    # Get insights
    insights = await learning_system.get_learning_insights(agent.id)
    
    assert insights["performance_trend"] > 0
    assert "code_analysis" in insights["strengths"]
    assert "code_generation" in insights["weaknesses"]
    assert len(insights["improvement_areas"]) > 0

@pytest.mark.asyncio
async def test_task_clustering(learning_system, mock_agents, learning_preferences):
    """Test task clustering based on similarities."""
    # Register agent
    agent = mock_agents[0]
    await learning_system.register_agent(agent, learning_preferences[0])
    
    # Record similar experiences
    similar_tasks = [
        ("task_type_1", {AgentCapability.CODE_ANALYSIS}, 0.9),
        ("task_type_2", {AgentCapability.CODE_ANALYSIS}, 0.85),
        ("task_type_3", {AgentCapability.CODE_GENERATION}, 0.4)
    ]
    
    for task_type, caps, outcome in similar_tasks:
        experience = Experience(
            agent_id=agent.id,
            task_id=f"task_{task_type}",
            task_type=task_type,
            capabilities_used=caps,
            context={},
            actions_taken=["action1"],
            outcome=outcome,
            timestamp=datetime.now(),
            execution_time=1.0,
            resource_usage={}
        )
        await learning_system.record_experience(experience)
        
    # Verify clustering
    assert "task_type_1" in learning_system.task_clusters["task_type_2"]
    assert "task_type_3" not in learning_system.task_clusters["task_type_1"]

@pytest.mark.asyncio
async def test_experience_cleanup(learning_system, mock_agents, learning_preferences):
    """Test cleanup of old experiences."""
    # Register agent
    agent = mock_agents[0]
    await learning_system.register_agent(agent, learning_preferences[0])
    
    # Create many experiences
    for i in range(1200):  # More than max limit
        experience = Experience(
            agent_id=agent.id,
            task_id=f"task_{i}",
            task_type="test_type",
            capabilities_used={AgentCapability.CODE_ANALYSIS},
            context={},
            actions_taken=["action1"],
            outcome=0.8,
            timestamp=datetime.now() - timedelta(hours=i),
            execution_time=1.0,
            resource_usage={}
        )
        await learning_system.record_experience(experience)
        
    # Verify cleanup
    assert len(learning_system.experiences[agent.id]) <= 1000  # Max limit
    # Verify most recent experiences are kept
    latest_exp = learning_system.experiences[agent.id][-1]
    assert (datetime.now() - latest_exp.timestamp).total_seconds() < 3600  # Within last hour 