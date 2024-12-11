"""
Tests for self-prompting system.
"""

import pytest
from datetime import datetime
from typing import Dict, Any, List

from ..core.self_prompting import (
    PromptContext,
    PromptTemplate,
    GeneratedPrompt,
    ContextualPromptStrategy,
    AdaptivePromptStrategy,
    SelfPromptManager
)

@pytest.fixture
def sample_context() -> PromptContext:
    """Create sample context for testing."""
    return PromptContext(
        task_type="analysis",
        current_state={
            "performance_metrics": {"latency": 0.5, "throughput": 1000},
            "error_rates": {"service_a": 0.01, "service_b": 0.02},
            "resource_metrics": {"cpu": 0.7, "memory": 0.8}
        },
        historical_context=[],
        metrics={"accuracy": 0.95, "f1_score": 0.92},
        constraints={"max_latency": 1.0, "min_throughput": 500},
        priority=1.0
    )

@pytest.fixture
def sample_templates() -> List[PromptTemplate]:
    """Create sample templates for testing."""
    return [
        PromptTemplate(
            name="performance",
            template="Analyze performance metrics: {performance_metrics}",
            required_context=["performance_metrics"],
            optional_context=["resource_metrics"],
            examples=[],
            metadata={}
        ),
        PromptTemplate(
            name="error_analysis",
            template="Analyze error rates: {error_rates}",
            required_context=["error_rates"],
            optional_context=[],
            examples=[],
            metadata={}
        )
    ]

class TestContextualStrategy:
    """Test contextual prompt generation strategy."""
    
    def test_template_selection(self, sample_context, sample_templates):
        """Test template selection based on context."""
        strategy = ContextualPromptStrategy()
        prompt = strategy.generate(sample_context, sample_templates)
        
        assert prompt.template_name in ["performance", "error_analysis"]
        assert isinstance(prompt.confidence, float)
        assert 0 <= prompt.confidence <= 1
        
    def test_missing_context(self, sample_templates):
        """Test handling of missing context."""
        strategy = ContextualPromptStrategy()
        context = PromptContext(
            task_type="analysis",
            current_state={},  # Empty state
            historical_context=[],
            metrics={},
            constraints={},
            priority=1.0
        )
        
        prompt = strategy.generate(context, sample_templates)
        assert "[performance_metrics]" in prompt.prompt
        assert prompt.confidence < 0.5
        
    def test_confidence_calculation(self, sample_context, sample_templates):
        """Test confidence score calculation."""
        strategy = ContextualPromptStrategy()
        prompt = strategy.generate(sample_context, sample_templates)
        
        # Should have high confidence with all required context
        assert prompt.confidence > 0.8

class TestAdaptiveStrategy:
    """Test adaptive prompt generation strategy."""
    
    def test_effectiveness_adaptation(self, sample_context, sample_templates):
        """Test adaptation based on effectiveness history."""
        strategy = AdaptivePromptStrategy()
        
        # Generate initial prompt
        prompt1 = strategy.generate(sample_context, sample_templates)
        
        # Update effectiveness
        strategy.update_effectiveness(prompt1.template_name, 0.3)  # Low effectiveness
        
        # Generate new prompt
        prompt2 = strategy.generate(sample_context, sample_templates)
        
        # Should adapt template selection
        assert prompt2.metadata.get('strategy') == 'adaptive'
        
    def test_historical_effectiveness(self, sample_context, sample_templates):
        """Test historical effectiveness tracking."""
        strategy = AdaptivePromptStrategy()
        
        # Generate prompts and update effectiveness
        for _ in range(5):
            prompt = strategy.generate(sample_context, sample_templates)
            strategy.update_effectiveness(prompt.template_name, 0.8)
            
        # Check history
        for template_name in strategy.effectiveness_history:
            history = strategy.effectiveness_history[template_name]
            assert len(history) <= 100  # Should maintain bounded history
            if history:
                assert all(0 <= score <= 1 for score in history)

class TestSelfPromptManager:
    """Test self-prompt manager functionality."""
    
    def test_strategy_selection(self, sample_context):
        """Test selection between contextual and adaptive strategies."""
        manager = SelfPromptManager()
        
        # Without history - should use contextual
        prompt1 = manager.generate_prompt(sample_context)
        assert prompt1.metadata['strategy'] == 'contextual'
        
        # With history - should use adaptive
        context_with_history = PromptContext(
            task_type=sample_context.task_type,
            current_state=sample_context.current_state,
            historical_context=[{'previous': 'state'}],
            metrics=sample_context.metrics,
            constraints=sample_context.constraints
        )
        prompt2 = manager.generate_prompt(context_with_history)
        assert prompt2.metadata['strategy'] == 'adaptive'
        
    def test_effectiveness_analysis(self, sample_context):
        """Test effectiveness analysis functionality."""
        manager = SelfPromptManager()
        
        # Generate prompts and update effectiveness
        for _ in range(3):
            prompt = manager.generate_prompt(sample_context)
            manager.update_effectiveness(prompt, 0.9)
            
        analysis = manager.analyze_prompt_effectiveness()
        assert isinstance(analysis, dict)
        
        for template_stats in analysis.values():
            assert 'mean' in template_stats
            assert 'trend' in template_stats
            assert template_stats['mean'] > 0
            
    def test_prompt_statistics(self, sample_context):
        """Test prompt usage statistics."""
        manager = SelfPromptManager()
        
        # Generate some prompts
        for _ in range(3):
            manager.generate_prompt(sample_context)
            
        stats = manager.get_prompt_statistics()
        assert stats['total_prompts'] == 3
        assert isinstance(stats['template_usage'], dict)
        assert isinstance(stats['average_confidence'], float)
        
    def test_template_extraction(self):
        """Test template context extraction."""
        manager = SelfPromptManager()
        
        template = "Analyze {required} with optional {optional:default}"
        required = manager._extract_required_context(template)
        optional = manager._extract_optional_context(template)
        
        assert "required" in required
        assert "optional" in optional 