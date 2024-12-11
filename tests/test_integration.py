"""
Tests for agent integration module.
"""

import pytest
import asyncio
from typing import Dict, Any
import time
import json
from pathlib import Path

from ..core.integration import AgentIntegration, IntegrationConfig
from ..agents.base_agent import BaseAgent, Message
from ..agents.neural_symbolic_agent import NeuralSymbolicAgent
from ..agents.prolog_reasoner import PrologReasoner
from ..agents.meta_reasoner import MetaReasoner

class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, agent_id: str):
        super().__init__()
        self.agent_id = agent_id
        self.received_messages = []
        self.state = {}
        
    async def process_message(self, message: Message) -> Message:
        """Process incoming message."""
        self.received_messages.append(message)
        return Message(
            id=f"response_{message.id}",
            sender=self.agent_id,
            receiver=message.sender,
            content={'success': True}
        )
        
    def get_state(self) -> Dict[str, Any]:
        """Get agent state."""
        return self.state

@pytest.fixture
def integration():
    """Create an integration instance."""
    config = IntegrationConfig(
        communication_timeout=1.0,
        batch_size=5
    )
    return AgentIntegration(config)

@pytest.fixture
def agents():
    """Create test agents."""
    return {
        'neural': MockAgent('neural'),
        'prolog': MockAgent('prolog'),
        'meta': MockAgent('meta')
    }

@pytest.mark.asyncio
async def test_sync_agents(integration, agents):
    """Test agent synchronization."""
    source = agents['neural']
    target = agents['prolog']
    
    # Set source state
    source.state = {
        'learned_rules': ['rule1', 'rule2'],
        'embeddings': [0.1, 0.2, 0.3]
    }
    
    # Sync agents
    success = await integration.sync_agents(source, target, 'knowledge')
    
    assert success
    assert len(target.received_messages) == 1
    assert target.received_messages[0].content['type'] == 'state_update'
    
@pytest.mark.asyncio
async def test_route_message_direct(integration, agents):
    """Test direct message routing."""
    # Register message queues
    for agent in agents.values():
        integration.message_queues[agent.agent_id] = asyncio.Queue()
        
    # Create test message
    message = Message(
        id="test",
        sender="neural",
        receiver="prolog",
        content={'data': 'test'}
    )
    
    # Route message
    success = await integration.route_message(message, 'direct')
    
    assert success
    routed_message = await integration.message_queues['prolog'].get()
    assert routed_message.id == message.id
    assert routed_message.content == message.content
    
@pytest.mark.asyncio
async def test_route_message_broadcast(integration, agents):
    """Test broadcast message routing."""
    # Register message queues
    for agent in agents.values():
        integration.message_queues[agent.agent_id] = asyncio.Queue()
        
    # Create test message
    message = Message(
        id="test",
        sender="neural",
        receiver="broadcast",
        content={'data': 'test'}
    )
    
    # Route message
    success = await integration.route_message(message, 'broadcast')
    
    assert success
    # Verify all agents received message
    for agent_id in agents:
        routed_message = await integration.message_queues[agent_id].get()
        assert routed_message.id == message.id
        
@pytest.mark.asyncio
async def test_route_message_filtered(integration, agents):
    """Test filtered message routing."""
    # Register message queues and states
    for agent_id, agent in agents.items():
        integration.message_queues[agent_id] = asyncio.Queue()
        integration.agent_states[agent_id] = {'type': 'neural' if agent_id == 'neural' else 'other'}
        
    # Create test message with filter
    message = Message(
        id="test",
        sender="meta",
        receiver="filtered",
        content={
            'data': 'test',
            'filters': {'type': 'neural'}
        }
    )
    
    # Route message
    success = await integration.route_message(message, 'filtered')
    
    assert success
    # Verify only neural agent received message
    assert integration.message_queues['neural'].qsize() == 1
    assert integration.message_queues['prolog'].qsize() == 0
    assert integration.message_queues['meta'].qsize() == 0
    
@pytest.mark.asyncio
async def test_batch_processing(integration, agents):
    """Test batch message processing."""
    source = agents['neural']
    target = agents['prolog']
    
    # Create test messages
    messages = [
        Message(
            id=f"test_{i}",
            sender="neural",
            receiver="prolog",
            content={'data': f'test_{i}'}
        )
        for i in range(10)
    ]
    
    # Process batch
    results = await integration.batch_process(source, target, messages)
    
    assert len(results) == 10
    assert all(results)
    assert len(target.received_messages) == 2  # 2 batches of 5 messages
    
def test_transform_state(integration):
    """Test state transformation."""
    # Neural to Prolog
    neural_state = {
        'learned_rules': ['X -> Y', 'Y -> Z'],
        'embeddings': [0.1, 0.2, 0.3]
    }
    
    prolog_state = integration._transform_state(
        neural_state,
        NeuralSymbolicAgent,
        PrologReasoner
    )
    
    assert 'rules' in prolog_state
    assert len(prolog_state['rules']) == 2
    
    # Prolog to Neural
    prolog_state = {
        'rules': ['parent(X, Y)', 'sibling(X, Y)'],
        'facts': ['parent(john, mary)']
    }
    
    neural_state = integration._transform_state(
        prolog_state,
        PrologReasoner,
        NeuralSymbolicAgent
    )
    
    assert 'learned_rules' in neural_state
    assert len(neural_state['learned_rules']) == 2
    
@pytest.mark.asyncio
async def test_concurrent_sync(integration, agents):
    """Test concurrent synchronization handling."""
    source = agents['neural']
    target = agents['prolog']
    
    # Start first sync
    task1 = asyncio.create_task(
        integration.sync_agents(source, target, 'knowledge')
    )
    
    # Try concurrent sync
    task2 = asyncio.create_task(
        integration.sync_agents(source, target, 'knowledge')
    )
    
    # Wait for both tasks
    result1 = await task1
    result2 = await task2
    
    assert result1 != result2  # One should succeed, one should fail
    
def test_sync_history(integration, agents):
    """Test synchronization history tracking."""
    source = agents['neural']
    target = agents['prolog']
    
    # Record sync
    integration._record_sync(source, target, 'knowledge', True)
    
    assert len(integration.sync_history) == 1
    sync_record = integration.sync_history[0]
    assert sync_record['source_id'] == source.agent_id
    assert sync_record['target_id'] == target.agent_id
    assert sync_record['sync_type'] == 'knowledge'
    assert sync_record['success']
    
@pytest.mark.parametrize("agent_type,state_type,expected_keys", [
    (NeuralSymbolicAgent, 'knowledge', ['learned_rules', 'embeddings']),
    (PrologReasoner, 'knowledge', ['rules', 'facts']),
    (MetaReasoner, 'patterns', ['patterns', 'dependencies'])
])
@pytest.mark.asyncio
async def test_get_agent_state(integration, agent_type, state_type, expected_keys):
    """Test getting state from different agent types."""
    agent = agent_type()  # type: ignore
    state = await integration._get_agent_state(agent, state_type)
    
    for key in expected_keys:
        assert key in state 