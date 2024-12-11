"""
Transformations module for converting states and knowledge between different agent types.
"""

from typing import Dict, Any, List, Optional, Union, Set, Tuple
from dataclasses import dataclass
import logging
import re
import json
from pathlib import Path
import numpy as np
from functools import lru_cache
from time import time
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class TransformationError(Exception):
    """Custom exception for transformation errors with context."""
    message: str
    source_format: str
    target_format: str
    context: Dict[str, Any]

@dataclass
class TransformationMetrics:
    """Metrics for transformation operations."""
    total_time: float
    rule_matches: int
    fallbacks_used: int
    errors_caught: int

@dataclass
class TransformationRule:
    """Rule for transforming data between formats."""
    source_format: str
    target_format: str
    pattern: str
    template: str
    bidirectional: bool = False
    priority: int = 0
    
    def __post_init__(self):
        """Compile pattern for better performance."""
        self._compiled_pattern = re.compile(self.pattern)
        
    def matches(self, key: str) -> bool:
        """Check if key matches rule pattern."""
        return bool(self._compiled_pattern.match(str(key)))

class StateTransformer:
    """Handles state transformations between different agent types."""
    
    def __init__(self):
        self.rules: List[TransformationRule] = []
        self.metrics: Dict[str, TransformationMetrics] = defaultdict(
            lambda: TransformationMetrics(0.0, 0, 0, 0)
        )
        self._load_rules()
        self._validate_rules()
        
    def _validate_rules(self):
        """Validate rules for cycles and conflicts."""
        # Check for transformation cycles
        visited: Set[Tuple[str, str]] = set()
        
        def check_cycle(source: str, target: str, path: List[str]) -> bool:
            if (source, target) in visited:
                logger.warning(f"Transformation cycle detected: {' -> '.join(path + [source])}")
                return True
            visited.add((source, target))
            return False
            
        for rule in self.rules:
            path = [rule.source_format]
            if check_cycle(rule.source_format, rule.target_format, path):
                # Don't raise, just warn and continue
                continue
                
    @lru_cache(maxsize=1000)
    def _get_applicable_rules(self, source_format: str, target_format: str) -> List[TransformationRule]:
        """Get cached list of applicable rules for a transformation."""
        return sorted(
            [
                rule for rule in self.rules
                if (rule.source_format == source_format and rule.target_format == target_format) or
                   (rule.bidirectional and rule.source_format == target_format and rule.target_format == source_format)
            ],
            key=lambda r: r.priority,
            reverse=True
        )
        
    def transform(self,
                 state: Dict[str, Any],
                 source_format: str,
                 target_format: str,
                 validation_hook: Optional[callable] = None) -> Dict[str, Any]:
        """Transform state from source format to target format with metrics and validation."""
        start_time = time()
        metrics = TransformationMetrics(0.0, 0, 0, 0)
        
        try:
            if source_format == target_format:
                return state.copy()
                
            # Get cached applicable rules
            applicable_rules = self._get_applicable_rules(source_format, target_format)
            
            if not applicable_rules:
                logger.warning(f"No rules found for {source_format} -> {target_format}")
                metrics.fallbacks_used += 1
                result = self._default_transform(state, source_format, target_format)
            else:
                # Apply transformations with metrics
                transformed = {}
                for rule in applicable_rules:
                    try:
                        if rule.source_format == target_format and rule.bidirectional:
                            # Reverse transformation
                            new_state = self._apply_reverse_rule(rule, state)
                        else:
                            new_state = self._apply_rule(rule, state)
                            
                        transformed.update(new_state)
                        metrics.rule_matches += 1
                        
                    except Exception as e:
                        metrics.errors_caught += 1
                        logger.error(f"Error applying rule {rule.pattern}: {e}")
                        # Continue with other rules
                        continue
                        
                result = transformed
                
            # Apply validation hook if provided
            if validation_hook:
                try:
                    validation_hook(result, source_format, target_format)
                except Exception as e:
                    logger.error(f"Validation failed: {e}")
                    metrics.errors_caught += 1
                    
            metrics.total_time = time() - start_time
            self._update_metrics(source_format, target_format, metrics)
            
            return result
            
        except Exception as e:
            metrics.errors_caught += 1
            metrics.total_time = time() - start_time
            self._update_metrics(source_format, target_format, metrics)
            
            raise TransformationError(
                message=str(e),
                source_format=source_format,
                target_format=target_format,
                context={'state': state}
            )
            
    def _update_metrics(self,
                       source_format: str,
                       target_format: str,
                       new_metrics: TransformationMetrics):
        """Update running metrics for transformation pair."""
        key = f"{source_format}->{target_format}"
        current = self.metrics[key]
        
        # Update running averages and counts
        self.metrics[key] = TransformationMetrics(
            total_time=(current.total_time + new_metrics.total_time) / 2,
            rule_matches=current.rule_matches + new_metrics.rule_matches,
            fallbacks_used=current.fallbacks_used + new_metrics.fallbacks_used,
            errors_caught=current.errors_caught + new_metrics.errors_caught
        )
        
    def get_metrics(self) -> Dict[str, TransformationMetrics]:
        """Get current transformation metrics."""
        return dict(self.metrics)
        
    def clear_metrics(self):
        """Reset all transformation metrics."""
        self.metrics.clear()
        
    def _load_rules(self):
        """Load transformation rules from configuration."""
        rules_path = Path(__file__).parent / "transformation_rules.json"
        if rules_path.exists():
            try:
                with open(rules_path) as f:
                    rules_data = json.load(f)
                    for rule_data in rules_data:
                        self.rules.append(TransformationRule(**rule_data))
            except Exception as e:
                logger.error(f"Error loading transformation rules: {e}")
        
    def _apply_rule(self, rule: TransformationRule, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformation rule to state."""
        result = {}
        pattern = re.compile(rule.pattern)
        
        for key, value in state.items():
            if pattern.match(str(key)):
                try:
                    transformed_key = self._transform_key(key, rule.template)
                    transformed_value = self._transform_value(value, rule)
                    result[transformed_key] = transformed_value
                except Exception as e:
                    logger.error(f"Error transforming {key}: {e}")
                    
        return result
        
    def _apply_reverse_rule(self, rule: TransformationRule, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply reverse transformation rule."""
        result = {}
        reverse_pattern = self._reverse_pattern(rule.template)
        reverse_template = self._reverse_pattern(rule.pattern)
        
        pattern = re.compile(reverse_pattern)
        
        for key, value in state.items():
            if pattern.match(str(key)):
                try:
                    transformed_key = self._transform_key(key, reverse_template)
                    transformed_value = self._transform_value(value, rule, reverse=True)
                    result[transformed_key] = transformed_value
                except Exception as e:
                    logger.error(f"Error reverse transforming {key}: {e}")
                    
        return result
        
    def _transform_key(self, key: str, template: str) -> str:
        """Transform key using template."""
        return template.replace('{}', key)
        
    def _transform_value(self,
                        value: Any,
                        rule: TransformationRule,
                        reverse: bool = False) -> Any:
        """Transform value based on rule."""
        if isinstance(value, dict):
            return self.transform(
                value,
                rule.target_format if reverse else rule.source_format,
                rule.source_format if reverse else rule.target_format
            )
        elif isinstance(value, list):
            return [
                self._transform_value(item, rule, reverse)
                for item in value
            ]
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return str(value)
            
    def _default_transform(self,
                         state: Dict[str, Any],
                         source_format: str,
                         target_format: str) -> Dict[str, Any]:
        """Default transformation when no rules match."""
        # Basic transformations for common formats
        if source_format == 'neural' and target_format == 'symbolic':
            return self._neural_to_symbolic(state)
        elif source_format == 'symbolic' and target_format == 'neural':
            return self._symbolic_to_neural(state)
        else:
            return state.copy()
            
    def _neural_to_symbolic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert neural state to symbolic format."""
        transformed = {}
        
        # Convert embeddings to rules
        if 'embeddings' in state:
            transformed['rules'] = [
                f"embedding({i}, {list(e)})"
                for i, e in enumerate(state['embeddings'])
            ]
            
        # Convert learned rules
        if 'learned_rules' in state:
            transformed['facts'] = [
                self._neural_rule_to_symbolic(rule)
                for rule in state['learned_rules']
            ]
            
        return transformed
        
    def _symbolic_to_neural(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert symbolic state to neural format."""
        transformed = {}
        
        # Convert rules to embeddings
        if 'rules' in state:
            embeddings = []
            for rule in state['rules']:
                if rule.startswith('embedding'):
                    try:
                        _, values = rule.split('(')[1].rstrip(')').split(',', 1)
                        embeddings.append(json.loads(values))
                    except Exception:
                        continue
            if embeddings:
                transformed['embeddings'] = embeddings
                
        # Convert facts to learned rules
        if 'facts' in state:
            transformed['learned_rules'] = [
                self._symbolic_fact_to_neural(fact)
                for fact in state['facts']
            ]
            
        return transformed
        
    def _neural_rule_to_symbolic(self, rule: str) -> str:
        """Convert neural rule to symbolic format."""
        # Example: "X -> Y" becomes "implies(X, Y)"
        if '->' in rule:
            antecedent, consequent = rule.split('->')
            return f"implies({antecedent.strip()}, {consequent.strip()})"
        return rule
        
    def _symbolic_fact_to_neural(self, fact: str) -> str:
        """Convert symbolic fact to neural format."""
        # Example: "implies(X, Y)" becomes "X -> Y"
        if fact.startswith('implies('):
            try:
                args = fact[8:-1]  # Remove implies( and )
                antecedent, consequent = args.split(',', 1)
                return f"{antecedent.strip()} -> {consequent.strip()}"
            except Exception:
                pass
        return fact
        
    def _reverse_pattern(self, pattern: str) -> str:
        """Reverse a pattern for bidirectional transformation."""
        # Simple pattern reversal - could be more sophisticated
        return pattern.replace('{', '{{').replace('}', '}}').format('(.*)')
        
    def register_rule(self, rule: TransformationRule):
        """Register a new transformation rule."""
        self.rules.append(rule)
        # Sort rules by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        
    def save_rules(self):
        """Save transformation rules to file."""
        rules_path = Path(__file__).parent / "transformation_rules.json"
        try:
            rules_data = [
                {
                    'source_format': rule.source_format,
                    'target_format': rule.target_format,
                    'pattern': rule.pattern,
                    'template': rule.template,
                    'bidirectional': rule.bidirectional,
                    'priority': rule.priority
                }
                for rule in self.rules
            ]
            
            with open(rules_path, 'w') as f:
                json.dump(rules_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving transformation rules: {e}")
            
    def get_available_transformations(self) -> List[Dict[str, str]]:
        """Get list of available transformations."""
        transformations = []
        
        for rule in self.rules:
            transformations.append({
                'source': rule.source_format,
                'target': rule.target_format,
                'pattern': rule.pattern,
                'bidirectional': rule.bidirectional
            })
            if rule.bidirectional:
                transformations.append({
                    'source': rule.target_format,
                    'target': rule.source_format,
                    'pattern': rule.template,
                    'bidirectional': True
                })
                
        return transformations 