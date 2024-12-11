"""
Tests for state transformations module.
"""

import pytest
import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple

from ..core.transformations import StateTransformer, TransformationRule, TransformationError, TransformationMetrics

@pytest.fixture
def transformer():
    """Create a StateTransformer instance."""
    return StateTransformer()

@pytest.fixture
def test_rules():
    """Create test transformation rules."""
    return [
        TransformationRule(
            source_format="neural",
            target_format="symbolic",
            pattern="X\\s*->\\s*Y",
            template="implies(X, Y)",
            bidirectional=True,
            priority=10
        ),
        TransformationRule(
            source_format="symbolic",
            target_format="neural",
            pattern="fact_(\\w+)",
            template="knowledge_{}",
            bidirectional=True,
            priority=7
        )
    ]

@pytest.fixture
def sample_states() -> Dict[str, Dict[str, Any]]:
    """Sample states for different formats."""
    return {
        'neural': {
            'learned_rule_1': 'X -> Y',
            'embedding_1': [0.1, 0.2, 0.3],
            'confidence_high': 0.9
        },
        'symbolic': {
            'fact_test': 'example',
            'rule_1': 'test_rule',
            'predicate_relation': 'test_pred'
        },
        'meta': {
            'pattern_1': 'meta_pattern',
            'strategy_test': 'test_strategy'
        }
    }

class TestBasicTransformations:
    """Test basic transformation functionality."""
    
    def test_simple_transform(self, transformer):
        """Test basic state transformation."""
        state = {'learned_rule_1': 'X -> Y'}
        result = transformer.transform(state, 'neural', 'symbolic')
        assert 'rules' in result or 'facts' in result
        
    def test_bidirectional_consistency(self, transformer):
        """Test bidirectional transformation maintains consistency."""
        state = {'learned_rule_1': 'X -> Y'}
        symbolic = transformer.transform(state, 'neural', 'symbolic')
        neural = transformer.transform(symbolic, 'symbolic', 'neural')
        assert any(key.startswith('learned_rule_') for key in neural.keys())
        
    @pytest.mark.parametrize("source_format,target_format,expected_keys", [
        ('neural', 'symbolic', ['implies', 'vector', 'certainty']),
        ('symbolic', 'neural', ['learned_rule', 'embedding']),
        ('meta', 'neural', ['pattern', 'strategy'])
    ])
    def test_format_specific_transforms(self, transformer, sample_states, 
                                     source_format, target_format, expected_keys):
        """Test format-specific transformations."""
        state = sample_states[source_format]
        result = transformer.transform(state, source_format, target_format)
        result_str = str(result)
        assert any(key in result_str for key in expected_keys)

class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_invalid_transformation(self, transformer):
        """Test error handling with invalid transformations."""
        state = {'invalid_key': object()}  # Untransformable object
        
        with pytest.raises(TransformationError) as exc_info:
            transformer.transform(state, 'unknown_format', 'symbolic')
            
        assert exc_info.value.source_format == 'unknown_format'
        assert exc_info.value.target_format == 'symbolic'
        assert 'state' in exc_info.value.context
        
    def test_invalid_rule_pattern(self, transformer):
        """Test handling of invalid regex patterns."""
        invalid_rule = TransformationRule(
            source_format="test",
            target_format="other",
            pattern="[invalid",  # Invalid regex
            template="test",
            priority=1
        )
        
        transformer.register_rule(invalid_rule)
        transformed = transformer.transform({'test': 'value'}, 'test', 'other')
        assert isinstance(transformed, dict)
        
    def test_cycle_detection(self, transformer):
        """Test detection of transformation rule cycles."""
        transformer.register_rule(TransformationRule(
            source_format='A',
            target_format='B',
            pattern='.*',
            template='{}',
            bidirectional=True
        ))
        
        transformer.register_rule(TransformationRule(
            source_format='B',
            target_format='A',
            pattern='.*',
            template='{}',
            bidirectional=True
        ))
        
        transformer._validate_rules()
        # Test should pass as cycles are warned but not blocked
        
    @pytest.mark.parametrize("invalid_state", [
        {'key': object()},  # Non-serializable object
        {'key': complex(1, 2)},  # Complex number
        {'key': lambda x: x}  # Function
    ])
    def test_invalid_value_types(self, transformer, invalid_state):
        """Test handling of invalid value types."""
        result = transformer.transform(invalid_state, 'source', 'target')
        assert isinstance(result, dict)
        assert all(isinstance(v, str) for v in result.values())

class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    def test_rule_caching(self, transformer):
        """Test caching of rule lookups."""
        rules1 = transformer._get_applicable_rules('neural', 'symbolic')
        rules2 = transformer._get_applicable_rules('neural', 'symbolic')
        assert rules1 == rules2
        
    def test_pattern_compilation(self, transformer):
        """Test pattern compilation in transformation rules."""
        rule = TransformationRule(
            source_format='test',
            target_format='test',
            pattern=r'test_(\d+)',
            template='result_{}',
        )
        
        assert rule.matches('test_123')
        assert not rule.matches('invalid')
        
    @pytest.mark.parametrize("num_transforms", [1, 5, 10])
    def test_metrics_collection(self, transformer, num_transforms):
        """Test metrics collection with different numbers of transformations."""
        state = {'learned_rule_1': 'X -> Y'}
        
        for _ in range(num_transforms):
            transformer.transform(state, 'neural', 'symbolic')
        
        metrics = transformer.get_metrics()
        assert 'neural->symbolic' in metrics
        
        metric = metrics['neural->symbolic']
        assert isinstance(metric, TransformationMetrics)
        assert metric.total_time > 0
        assert metric.rule_matches == num_transforms

class TestComplexTransformations:
    """Test complex transformation scenarios."""
    
    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_nested_state_transform(self, transformer, depth):
        """Test transformation of nested states with varying depths."""
        def create_nested_state(current_depth: int) -> Dict:
            if current_depth <= 0:
                return {'learned_rule': 'X -> Y'}
            return {
                'nested': create_nested_state(current_depth - 1),
                f'learned_rule_{current_depth}': f'A_{current_depth} -> B_{current_depth}'
            }
            
        state = create_nested_state(depth)
        result = transformer.transform(state, 'neural', 'symbolic')
        
        def verify_depth(d: Dict, current_depth: int) -> bool:
            if current_depth <= 0:
                return True
            return ('nested' in d and 
                   isinstance(d['nested'], dict) and 
                   verify_depth(d['nested'], current_depth - 1))
                   
        assert verify_depth(result, depth)
        
    @pytest.mark.parametrize("value_type,test_value", [
        (int, 42),
        (float, 3.14),
        (list, [1, 2, 3]),
        (dict, {'key': 'value'}),
        (np.ndarray, np.array([1.0, 2.0, 3.0])),
        (bool, True),
        (str, "test_string")
    ])
    def test_value_types(self, transformer, value_type, test_value):
        """Test transformation of different value types."""
        state = {'test_key': test_value}
        transformed = transformer.transform(state, 'source', 'target')
        
        if value_type == np.ndarray:
            assert isinstance(transformed['test_key'], list)
        else:
            assert isinstance(transformed['test_key'], (value_type, str))
            
    def test_fallback_behavior(self, transformer):
        """Test fallback transformation when no rules match."""
        state = {'custom_key': 'value'}
        result = transformer.transform(state, 'unknown', 'symbolic')
        assert isinstance(result, dict)
        assert 'custom_key' in result

class TestValidation:
    """Test validation and verification features."""
    
    @pytest.mark.parametrize("validation_criteria", [
        lambda r, s, t: isinstance(r, dict),
        lambda r, s, t: len(r) > 0,
        lambda r, s, t: all(isinstance(k, str) for k in r.keys())
    ])
    def test_validation_hooks(self, transformer, validation_criteria):
        """Test different validation hook criteria."""
        state = {'learned_rule_1': 'X -> Y'}
        
        result = transformer.transform(
            state,
            'neural',
            'symbolic',
            validation_hook=validation_criteria
        )
        
    def test_metrics_reset(self, transformer):
        """Test metrics reset functionality."""
        state = {'learned_rule_1': 'X -> Y'}
        
        transformer.transform(state, 'neural', 'symbolic')
        assert len(transformer.get_metrics()) > 0
        
        transformer.clear_metrics()
        assert len(transformer.get_metrics()) == 0