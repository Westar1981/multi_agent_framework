"""
Tests for MetaReasoner agent.
"""

import pytest
import asyncio
from typing import Dict, Any
import time
import json
from pathlib import Path

from ..agents.meta_reasoner import MetaReasoner, ReasoningPattern
from ..agents.base_agent import Message
from ..core.optimization import OptimizationConfig

@pytest.fixture
def meta_reasoner():
    """Create a MetaReasoner instance."""
    config = OptimizationConfig(
        memory_threshold=0.8,
        processing_threshold=0.9,
        accuracy_threshold=0.95
    )
    return MetaReasoner(config)

@pytest.fixture
def test_patterns():
    """Create test reasoning patterns."""
    return [
        ReasoningPattern(
            name="memory_optimization",
            description="Optimize memory usage",
            conditions=["state['memory_usage'] > 0.8"],
            actions=["optimize_memory"],
            success_rate=0.9,
            usage_count=10
        ),
        ReasoningPattern(
            name="processing_optimization",
            description="Optimize processing efficiency",
            conditions=["state['processing_time'] > 0.9"],
            actions=["optimize_processing"],
            success_rate=0.85,
            usage_count=8
        )
    ]

@pytest.mark.asyncio
async def test_analyze_system(meta_reasoner, test_patterns):
    """Test system analysis."""
    # Add test patterns
    for pattern in test_patterns:
        meta_reasoner.patterns[pattern.name] = pattern
        
    # Test system state
    system_state = {
        'memory_usage': 0.9,
        'processing_time': 1.0
    }
    
    # Create analysis message
    message = Message(
        id="test",
        sender="test_sender",
        receiver="test_receiver",
        content={'analyze_system': system_state}
    )
    
    # Process message
    response = await meta_reasoner.process_message(message)
    
    assert response is not None
    assert response.content is not None
    assert 'optimization_plan' in response.content
    assert len(response.content['optimization_plan']) == 2
    
@pytest.mark.asyncio
async def test_optimize_component(meta_reasoner):
    """Test component optimization."""
    # Test component data
    component_data = {
        'id': 'test_component',
        'metrics': {
            'memory_usage': 0.9,
            'processing_time': 1.0,
            'error_rate': 0.15
        }
    }
    
    # Create optimization message
    message = Message(
        id="test",
        sender="test_sender",
        receiver="test_receiver",
        content={'optimize_component': component_data}
    )
    
    # Process message
    response = await meta_reasoner.process_message(message)
    
    assert response is not None
    assert response.content is not None
    assert response.content['status'] == 'success'
    assert len(response.content['strategy']) > 0
    
def test_pattern_conditions(meta_reasoner, test_patterns):
    """Test pattern condition checking."""
    pattern = test_patterns[0]  # memory_optimization pattern
    
    # Test matching state
    state1 = {'memory_usage': 0.9}
    assert meta_reasoner._check_pattern_conditions(pattern, state1)
    
    # Test non-matching state
    state2 = {'memory_usage': 0.7}
    assert not meta_reasoner._check_pattern_conditions(pattern, state2)
    
def test_optimization_strategy_selection(meta_reasoner):
    """Test optimization strategy selection."""
    # Test component data
    component_data = {
        'metrics': {
            'memory_usage': 0.9,
            'processing_time': 1.0,
            'error_rate': 0.15
        }
    }
    
    strategy = meta_reasoner._select_optimization_strategy(component_data)
    
    assert 'optimize_memory' in strategy
    assert 'optimize_processing' in strategy
    assert 'optimize_accuracy' in strategy
    
@pytest.mark.asyncio
async def test_pattern_persistence(meta_reasoner, test_patterns, tmp_path):
    """Test pattern saving and loading."""
    # Set up test patterns
    for pattern in test_patterns:
        meta_reasoner.patterns[pattern.name] = pattern
        
    # Save patterns
    patterns_path = tmp_path / "test_patterns.json"
    with open(patterns_path, 'w') as f:
        patterns_data = [
            {
                'name': pattern.name,
                'description': pattern.description,
                'conditions': pattern.conditions,
                'actions': pattern.actions,
                'success_rate': pattern.success_rate,
                'usage_count': pattern.usage_count,
                'avg_processing_time': pattern.avg_processing_time
            }
            for pattern in test_patterns
        ]
        json.dump(patterns_data, f)
        
    # Create new reasoner and load patterns
    new_reasoner = MetaReasoner()
    new_reasoner.patterns.clear()
    
    # Load patterns
    with open(patterns_path) as f:
        patterns_data = json.load(f)
        for pattern_data in patterns_data:
            pattern = ReasoningPattern(**pattern_data)
            new_reasoner.patterns[pattern.name] = pattern
            
    # Verify patterns loaded correctly
    assert len(new_reasoner.patterns) == len(test_patterns)
    assert all(p.name in new_reasoner.patterns for p in test_patterns)
    
def test_pattern_dependencies(meta_reasoner, test_patterns):
    """Test pattern dependency handling."""
    # Add dependencies
    meta_reasoner.pattern_dependencies["processing_optimization"].add("memory_optimization")
    
    # Add patterns
    for pattern in test_patterns:
        meta_reasoner.patterns[pattern.name] = pattern
        
    # Sort patterns
    sorted_patterns = meta_reasoner._sort_patterns_by_dependencies(test_patterns)
    
    # Verify order (memory_optimization should come before processing_optimization)
    assert sorted_patterns[0].name == "memory_optimization"
    assert sorted_patterns[1].name == "processing_optimization"
    
@pytest.mark.asyncio
async def test_concurrent_optimization(meta_reasoner):
    """Test concurrent optimization handling."""
    component_id = "test_component"
    
    # Start first optimization
    result1 = await meta_reasoner._optimize_component({
        'id': component_id,
        'metrics': {'memory_usage': 0.9}
    })
    
    # Try concurrent optimization
    result2 = await meta_reasoner._optimize_component({
        'id': component_id,
        'metrics': {'memory_usage': 0.9}
    })
    
    assert result1['status'] == 'success'
    assert result2['status'] == 'optimization_in_progress'
    
def test_performance_metrics(meta_reasoner):
    """Test performance metrics tracking."""
    # Add test metrics
    meta_reasoner._update_performance_metrics(0.1, True)
    meta_reasoner._update_performance_metrics(0.2, True)
    meta_reasoner._update_performance_metrics(0.3, False)
    
    # Get optimization status
    status = meta_reasoner.get_optimization_status()
    
    assert 'performance_metrics' in status
    assert 'processing_time' in status['performance_metrics']
    assert 'success_rate' in status['performance_metrics']
    
@pytest.mark.parametrize("system_state,expected_patterns", [
    ({'memory_usage': 0.9, 'processing_time': 0.7}, ['memory_optimization']),
    ({'memory_usage': 0.7, 'processing_time': 1.0}, ['processing_optimization']),
    ({'memory_usage': 0.9, 'processing_time': 1.0}, ['memory_optimization', 'processing_optimization'])
])
def test_pattern_matching(meta_reasoner, test_patterns, system_state, expected_patterns):
    """Test pattern matching with different system states."""
    # Add patterns
    for pattern in test_patterns:
        meta_reasoner.patterns[pattern.name] = pattern
        
    # Find applicable patterns
    applicable = meta_reasoner._find_applicable_patterns(system_state)
    
    assert len(applicable) == len(expected_patterns)
    assert all(p.name in expected_patterns for p in applicable) 