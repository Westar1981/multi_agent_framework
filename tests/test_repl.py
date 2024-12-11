"""
Tests for the interactive REPL system.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import json
from datetime import datetime

from ..repl.interactive_shell import AgentREPL
from ..agents.base_agent import AgentCapability
from ..core.task_allocation import TaskPriority

@pytest.fixture
def repl():
    """Create REPL instance for testing."""
    return AgentREPL()

@pytest.mark.asyncio
async def test_create_agent(repl):
    """Test agent creation command."""
    result = await repl._handle_command(
        'create_agent("test_agent", ["CODE_ANALYSIS", "CODE_GENERATION"])'
    )
    
    assert "Created agent 'test_agent'" in result
    assert "test_agent" in repl.agents
    assert len(repl.agents["test_agent"].capabilities) == 2
    assert AgentCapability.CODE_ANALYSIS in repl.agents["test_agent"].capabilities
    assert AgentCapability.CODE_GENERATION in repl.agents["test_agent"].capabilities

@pytest.mark.asyncio
async def test_list_agents(repl):
    """Test listing agents command."""
    # Empty list
    result = await repl._handle_command('list_agents()')
    assert result == "No agents created"
    
    # Create some agents
    await repl._handle_command(
        'create_agent("agent1", ["CODE_ANALYSIS"])'
    )
    await repl._handle_command(
        'create_agent("agent2", ["CODE_GENERATION"])'
    )
    
    result = await repl._handle_command('list_agents()')
    assert "agent1: ['CODE_ANALYSIS']" in result
    assert "agent2: ['CODE_GENERATION']" in result

@pytest.mark.asyncio
async def test_create_task(repl):
    """Test task creation command."""
    # Create agent first
    await repl._handle_command(
        'create_agent("test_agent", ["CODE_ANALYSIS"])'
    )
    
    result = await repl._handle_command(
        'create_task("analyze_code", "HIGH", ["CODE_ANALYSIS"])'
    )
    
    assert "Created and submitted task 'analyze_code'" in result
    
    # Verify task in manager
    tasks = repl.task_manager.task_manager.tasks
    assert len(tasks) == 1
    task = list(tasks.values())[0]
    assert task.name == "analyze_code"
    assert task.priority == TaskPriority.HIGH
    assert AgentCapability.CODE_ANALYSIS in task.required_capabilities

@pytest.mark.asyncio
async def test_get_metrics(repl):
    """Test getting agent metrics."""
    # Test non-existent agent
    result = await repl._handle_command('get_metrics("unknown")')
    assert "not found" in result
    
    # Create agent and get metrics
    await repl._handle_command(
        'create_agent("test_agent", ["CODE_ANALYSIS"])'
    )
    result = await repl._handle_command('get_metrics("test_agent")')
    
    metrics = json.loads(result)
    assert "workload" in metrics
    assert "capabilities" in metrics
    assert metrics["workload"] == 0.0
    assert "CODE_ANALYSIS" in metrics["capabilities"]

@pytest.mark.asyncio
async def test_get_insights(repl):
    """Test getting learning insights."""
    # Create agent
    await repl._handle_command(
        'create_agent("test_agent", ["CODE_ANALYSIS"])'
    )
    
    # Create and complete task
    await repl._handle_command(
        'create_task("analyze_code", "HIGH", ["CODE_ANALYSIS"])'
    )
    
    result = await repl._handle_command('get_insights("test_agent")')
    insights = json.loads(result)
    
    assert "learning_rate" in insights
    assert "exploration_rate" in insights

@pytest.mark.asyncio
async def test_dump_state(repl):
    """Test state dumping command."""
    # Create test data
    await repl._handle_command(
        'create_agent("test_agent", ["CODE_ANALYSIS"])'
    )
    await repl._handle_command(
        'create_task("analyze_code", "HIGH", ["CODE_ANALYSIS"])'
    )
    
    # Test agent dump
    result = await repl._handle_command('dump("agents")')
    agents_data = json.loads(result)
    assert "test_agent" in agents_data
    assert "CODE_ANALYSIS" in agents_data["test_agent"]["capabilities"]
    
    # Test task dump
    result = await repl._handle_command('dump("tasks")')
    tasks_data = json.loads(result)
    assert len(tasks_data) == 1
    task_id = list(tasks_data.keys())[0]
    assert tasks_data[task_id]["name"] == "analyze_code"
    assert tasks_data[task_id]["priority"] == "HIGH"
    
    # Test learning dump
    result = await repl._handle_command('dump("learning")')
    learning_data = json.loads(result)
    assert "test_agent" in learning_data

@pytest.mark.asyncio
async def test_python_evaluation(repl):
    """Test Python code evaluation."""
    # Test simple expression
    result = await repl._handle_command('2 + 2')
    assert result == 4
    
    # Test accessing framework objects
    await repl._handle_command(
        'create_agent("test_agent", ["CODE_ANALYSIS"])'
    )
    result = await repl._handle_command('len(agents)')
    assert result == 1
    
    # Test async operation
    result = await repl._handle_command(
        'await task_manager.get_agent_metrics("test_agent")'
    )
    assert isinstance(result, dict)

@pytest.mark.asyncio
async def test_error_handling(repl):
    """Test error handling in commands."""
    # Test invalid agent creation
    result = await repl._handle_command(
        'create_agent("test_agent", ["INVALID_CAPABILITY"])'
    )
    assert "Error" in result
    
    # Test invalid task creation
    result = await repl._handle_command(
        'create_task("test_task", "INVALID_PRIORITY", [])'
    )
    assert "Error" in result
    
    # Test invalid Python code
    result = await repl._handle_command('invalid_code')
    assert result is None  # Should not raise exception

@pytest.mark.asyncio
async def test_command_history(repl):
    """Test command history tracking."""
    commands = [
        'create_agent("agent1", ["CODE_ANALYSIS"])',
        'list_agents()',
        'create_task("task1", "HIGH", ["CODE_ANALYSIS"])'
    ]
    
    for cmd in commands:
        await repl._handle_command(cmd)
        
    assert len(repl.history) == len(commands)
    assert all(cmd in repl.history for cmd in commands)

@pytest.mark.asyncio
async def test_last_result(repl):
    """Test last result tracking."""
    await repl._handle_command('2 + 2')
    assert repl.last_result == 4
    
    result = await repl._handle_command('last_result * 2')
    assert result == 8 