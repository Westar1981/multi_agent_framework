"""
Self-prompting system for dynamic prompt generation and adaptation.
Enables autonomous prompt refinement and context-aware instruction generation.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PromptContext:
    """Context information for prompt generation."""
    task_type: str
    current_state: Dict[str, Any]
    historical_context: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    constraints: Dict[str, Any]
    priority: float = 1.0

@dataclass
class PromptTemplate:
    """Template for generating prompts."""
    name: str
    template: str
    required_context: List[str]
    optional_context: List[str]
    examples: List[Dict[str, str]]
    metadata: Dict[str, Any]

@dataclass
class GeneratedPrompt:
    """A generated prompt with metadata."""
    prompt: str
    context_used: Dict[str, Any]
    template_name: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]

class PromptStrategy(ABC):
    """Base class for prompt generation strategies."""
    
    @abstractmethod
    def generate(self, context: PromptContext, templates: List[PromptTemplate]) -> GeneratedPrompt:
        """Generate a prompt using the strategy."""
        pass

class ContextualPromptStrategy(PromptStrategy):
    """Generates prompts based on context matching."""
    
    def generate(self, context: PromptContext, templates: List[PromptTemplate]) -> GeneratedPrompt:
        # Find best matching template
        best_template = self._find_best_template(context, templates)
        
        # Generate prompt with context
        filled_prompt = self._fill_template(best_template, context)
        
        # Calculate confidence
        confidence = self._calculate_confidence(context, best_template)
        
        return GeneratedPrompt(
            prompt=filled_prompt,
            context_used=context.current_state,
            template_name=best_template.name,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata={'strategy': 'contextual'}
        )
        
    def _find_best_template(self, context: PromptContext, templates: List[PromptTemplate]) -> PromptTemplate:
        """Find best matching template for context."""
        scores = []
        for template in templates:
            # Calculate context match score
            required_match = sum(1 for key in template.required_context if key in context.current_state)
            optional_match = sum(1 for key in template.optional_context if key in context.current_state)
            
            # Weight required matches more heavily
            score = (required_match * 2 + optional_match) * context.priority
            scores.append((template, score))
            
        return max(scores, key=lambda x: x[1])[0]
        
    def _fill_template(self, template: PromptTemplate, context: PromptContext) -> str:
        """Fill template with context values."""
        try:
            return template.template.format(**context.current_state)
        except KeyError as e:
            logger.warning(f"Missing context key {e} for template {template.name}")
            # Use default values or placeholders for missing context
            return self._fill_with_defaults(template, context)
            
    def _fill_with_defaults(self, template: PromptTemplate, context: PromptContext) -> str:
        """Fill template with default values for missing context."""
        values = context.current_state.copy()
        for key in template.required_context:
            if key not in values:
                values[key] = f"[{key}]"  # Placeholder
        return template.template.format(**values)
        
    def _calculate_confidence(self, context: PromptContext, template: PromptTemplate) -> float:
        """Calculate confidence score for generated prompt."""
        required_match = sum(1 for key in template.required_context if key in context.current_state)
        optional_match = sum(1 for key in template.optional_context if key in context.current_state)
        
        required_score = required_match / len(template.required_context) if template.required_context else 1.0
        optional_score = optional_match / len(template.optional_context) if template.optional_context else 1.0
        
        return (required_score * 0.7 + optional_score * 0.3) * context.priority

class AdaptivePromptStrategy(PromptStrategy):
    """Generates prompts that adapt based on historical effectiveness."""
    
    def __init__(self):
        self.effectiveness_history: Dict[str, List[float]] = defaultdict(list)
        
    def generate(self, context: PromptContext, templates: List[PromptTemplate]) -> GeneratedPrompt:
        # Select template based on historical effectiveness
        best_template = self._select_template(context, templates)
        
        # Adapt template based on historical patterns
        adapted_template = self._adapt_template(best_template, context)
        
        # Generate prompt
        filled_prompt = self._fill_template(adapted_template, context)
        
        return GeneratedPrompt(
            prompt=filled_prompt,
            context_used=context.current_state,
            template_name=best_template.name,
            confidence=self._calculate_confidence(context, best_template),
            timestamp=datetime.now(),
            metadata={'strategy': 'adaptive'}
        )
        
    def _select_template(self, context: PromptContext, templates: List[PromptTemplate]) -> PromptTemplate:
        """Select template based on historical effectiveness."""
        scores = []
        for template in templates:
            effectiveness = np.mean(self.effectiveness_history[template.name]) if self.effectiveness_history[template.name] else 0.5
            context_match = self._calculate_context_match(template, context)
            score = effectiveness * 0.7 + context_match * 0.3
            scores.append((template, score))
            
        return max(scores, key=lambda x: x[1])[0]
        
    def _adapt_template(self, template: PromptTemplate, context: PromptContext) -> PromptTemplate:
        """Adapt template based on historical patterns."""
        # Create adapted copy
        adapted = PromptTemplate(
            name=template.name,
            template=template.template,
            required_context=template.required_context.copy(),
            optional_context=template.optional_context.copy(),
            examples=template.examples.copy(),
            metadata=template.metadata.copy()
        )
        
        # Adapt based on historical effectiveness
        if self.effectiveness_history[template.name]:
            effectiveness = np.mean(self.effectiveness_history[template.name])
            if effectiveness < 0.5:
                # Add more context requirements for low-performing templates
                adapted.required_context.extend(adapted.optional_context)
                adapted.optional_context = []
                
        return adapted
        
    def _calculate_context_match(self, template: PromptTemplate, context: PromptContext) -> float:
        """Calculate how well template matches current context."""
        required_match = sum(1 for key in template.required_context if key in context.current_state)
        optional_match = sum(1 for key in template.optional_context if key in context.current_state)
        
        required_score = required_match / len(template.required_context) if template.required_context else 1.0
        optional_score = optional_match / len(template.optional_context) if template.optional_context else 1.0
        
        return required_score * 0.7 + optional_score * 0.3
        
    def _fill_template(self, template: PromptTemplate, context: PromptContext) -> str:
        """Fill template with context values."""
        try:
            return template.template.format(**context.current_state)
        except KeyError:
            return self._fill_with_defaults(template, context)
            
    def _fill_with_defaults(self, template: PromptTemplate, context: PromptContext) -> str:
        """Fill template with default values for missing context."""
        values = context.current_state.copy()
        for key in template.required_context:
            if key not in values:
                values[key] = f"[{key}]"
        return template.template.format(**values)
        
    def update_effectiveness(self, template_name: str, effectiveness: float):
        """Update historical effectiveness for a template."""
        self.effectiveness_history[template_name].append(effectiveness)
        # Keep history bounded
        if len(self.effectiveness_history[template_name]) > 100:
            self.effectiveness_history[template_name] = self.effectiveness_history[template_name][-100:]

class SelfPromptManager:
    """Manages prompt generation and adaptation."""
    
    def __init__(self):
        self.templates = self._load_templates()
        self.contextual_strategy = ContextualPromptStrategy()
        self.adaptive_strategy = AdaptivePromptStrategy()
        self.prompt_history: List[GeneratedPrompt] = []
        
    def _load_templates(self) -> List[PromptTemplate]:
        """Load prompt templates from configuration."""
        templates_path = Path(__file__).parent / "analysis_prompts.json"
        if not templates_path.exists():
            return []
            
        with open(templates_path) as f:
            data = json.load(f)
            
        return [
            PromptTemplate(
                name=name,
                template=template,
                required_context=self._extract_required_context(template),
                optional_context=self._extract_optional_context(template),
                examples=[],  # Could be loaded from separate examples file
                metadata={}
            )
            for name, template in data.items()
        ]
        
    def _extract_required_context(self, template: str) -> List[str]:
        """Extract required context keys from template."""
        import re
        # Find all {variable} patterns that don't have default values
        return re.findall(r'\{([^{}:]+)\}', template)
        
    def _extract_optional_context(self, template: str) -> List[str]:
        """Extract optional context keys from template."""
        import re
        # Find all {variable:default} patterns
        return re.findall(r'\{([^{}]+):[^{}]+\}', template)
        
    def generate_prompt(self, context: PromptContext) -> GeneratedPrompt:
        """Generate prompt using appropriate strategy."""
        if context.historical_context:
            # Use adaptive strategy when historical context is available
            prompt = self.adaptive_strategy.generate(context, self.templates)
        else:
            # Use contextual strategy for new scenarios
            prompt = self.contextual_strategy.generate(context, self.templates)
            
        self.prompt_history.append(prompt)
        return prompt
        
    def update_effectiveness(self, prompt: GeneratedPrompt, effectiveness: float):
        """Update effectiveness metrics for prompt."""
        self.adaptive_strategy.update_effectiveness(prompt.template_name, effectiveness)
        
    def analyze_prompt_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of different prompt templates."""
        effectiveness = defaultdict(list)
        
        for template_name, history in self.adaptive_strategy.effectiveness_history.items():
            if history:
                effectiveness[template_name] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'trend': self._calculate_trend(history),
                    'sample_size': len(history)
                }
                
        return dict(effectiveness)
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in effectiveness values."""
        if len(values) < 2:
            return 'insufficient_data'
            
        slope = np.polyfit(range(len(values)), values, 1)[0]
        if slope > 0.1:
            return 'improving'
        elif slope < -0.1:
            return 'degrading'
        return 'stable'
        
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get statistics about prompt generation."""
        return {
            'total_prompts': len(self.prompt_history),
            'template_usage': self._count_template_usage(),
            'average_confidence': np.mean([p.confidence for p in self.prompt_history]) if self.prompt_history else 0,
            'effectiveness_analysis': self.analyze_prompt_effectiveness()
        }
        
    def _count_template_usage(self) -> Dict[str, int]:
        """Count usage of each template."""
        usage = defaultdict(int)
        for prompt in self.prompt_history:
            usage[prompt.template_name] += 1
        return dict(usage) 