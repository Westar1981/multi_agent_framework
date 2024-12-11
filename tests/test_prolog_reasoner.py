"""
Tests for PrologReasoner agent.
"""

import pytest
import asyncio
from typing import Dict, Any
import time

from ..agents.prolog_reasoner import PrologReasoner, RuleStats
from ..agents.base_agent import Message
from ..core.optimization import OptimizationConfig

@pytest.fixture
def reasoner():
    """Create a PrologReasoner instance."""
    config = OptimizationConfig(
        cache_size=100,
        memory_threshold=0.8,
        processing_threshold=0.9
    )
    return PrologReasoner(config)

@pytest.mark.asyncio
async def test_process_query(reasoner):
    """Test query processing."""
    # Add test rule
    if reasoner.prolog:
        reasoner.prolog.assertz("test_fact(x)")
        
        # Create test message
        message = Message(
            id="test",
            sender="test_sender",
            receiver="test_receiver",
            content={'query': "test_fact(X)"}
        )
        
        # Process message
        response = await reasoner.process_message(message)
        
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        
@pytest.mark.asyncio
async def test_query_caching(reasoner):
    """Test query result caching."""
    if reasoner.prolog:
        # Add test rule
        reasoner.prolog.assertz("cached_fact(y)")
        
        # First query
        message1 = Message(
            id="test1",
            sender="test_sender",
            receiver="test_receiver",
            content={'query': "cached_fact(X)"}
        )
        
        # Process first query
        await reasoner.process_message(message1)
        
        # Second query (should use cache)
        message2 = Message(
            id="test2",
            sender="test_sender",
            receiver="test_receiver",
            content={'query': "cached_fact(X)"}
        )
        
        # Process second query
        start_time = time.time()
        response = await reasoner.process_message(message2)
        processing_time = time.time() - start_time
        
        # Verify cache hit
        assert processing_time < 0.1  # Should be very fast
        assert response is not None
        assert response.content is not None
        
def test_cleanup_knowledge_base(reasoner):
    """Test knowledge base cleanup."""
    if reasoner.prolog:
        # Add test rules
        reasoner.prolog.assertz("old_fact(z)")
        reasoner.rule_stats["old_fact(z)"].last_used = time.time() - 7200  # 2 hours ago
        
        # Cleanup
        reasoner.cleanup_knowledge_base()
        
        # Verify cleanup
        assert "old_fact(z)" not in reasoner.rule_stats
        
def test_performance_metrics(reasoner):
    """Test performance metrics tracking."""
    # Add test metrics
    reasoner._update_performance_metrics(0.1, True)
    reasoner._update_performance_metrics(0.2, True)
    reasoner._update_performance_metrics(0.3, False)
    
    metrics = reasoner.get_performance_metrics()
    
    assert 'processing_time' in metrics
    assert 'success_rate' in metrics
    assert metrics['success_rate'] == pytest.approx(0.67, 0.01)
    
def test_optimization_status(reasoner):
    """Test optimization status reporting."""
    if reasoner.prolog:
        # Add test data
        reasoner.prolog.assertz("test_rule(x)")
        reasoner._update_stats("test_rule(x)", 0.1)
        
        status = reasoner.get_optimization_status()
        
        assert 'metrics' in status
        assert 'rule_stats' in status
        assert 'cache_stats' in status
        assert status['cache_stats']['max_size'] == reasoner.config.cache_size
        
@pytest.mark.asyncio
async def test_rule_refinement(reasoner):
    """Test rule refinement process."""
    if reasoner.prolog:
        # Add test rule with poor performance
        reasoner.prolog.assertz("slow_rule(x)")
        stats = reasoner.rule_stats["slow_rule(x)"]
        stats.avg_processing_time = 1.0  # Slow
        stats.success_rate = 0.7  # Low success
        
        # Trigger refinement
        reasoner.refine_inference_rules()
        
        # Verify refinement attempt
        assert "slow_rule(x)" in reasoner.rule_stats
        
@pytest.mark.parametrize("query,expected_cache", [
    ("frequent_query(x)", True),   # High usage, slow
    ("rare_query(x)", False),      # Low usage, fast
    ("medium_query(x)", True)      # Medium usage, very slow
])
def test_caching_strategy(reasoner, query, expected_cache):
    """Test caching strategy decisions."""
    # Setup query stats
    stats = reasoner.rule_stats[query]
    if query == "frequent_query(x)":
        stats.usage_count = 10
        stats.avg_processing_time = 0.2
    elif query == "rare_query(x)":
        stats.usage_count = 2
        stats.avg_processing_time = 0.05
    else:  # medium_query
        stats.usage_count = 5
        stats.avg_processing_time = 0.5
        
    should_cache = reasoner._should_cache(query, ["result"])
    assert should_cache == expected_cache
    
def test_performance_history_bounds(reasoner):
    """Test performance history size limits."""
    # Add many metrics
    for i in range(2000):
        reasoner._update_performance_metrics(0.1, True)
        
    # Verify history is bounded
    assert len(reasoner.performance_history['processing_time']) <= 1000
    assert len(reasoner.performance_history['success_rate']) <= 1000
    
@pytest.mark.asyncio
async def test_error_handling(reasoner):
    """Test error handling in query processing."""
    # Create invalid query message
    message = Message(
        id="test",
        sender="test_sender",
        receiver="test_receiver",
        content={'query': "invalid_syntax("}
    )
    
    # Process message
    response = await reasoner.process_message(message)
    
    # Verify error handling
    assert response is None 