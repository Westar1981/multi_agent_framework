"""
Meta-reasoning agent for high-level optimization and coordination.
"""

from typing import Dict, Any, List, Optional, Set
import logging
from dataclasses import dataclass
import time
from pathlib import Path
import json
from collections import defaultdict
import numpy as np

from .base_agent import BaseAgent, Message
from ..core.optimization import OptimizationConfig

logger = logging.getLogger(__name__)

@dataclass
class ReasoningPattern:
    """Represents a meta-reasoning pattern."""
    name: str
    description: str
    conditions: List[str]
    actions: List[str]
    success_rate: float = 1.0
    usage_count: int = 0
    avg_processing_time: float = 0.0

class MetaReasoner(BaseAgent):
    """Meta-reasoning agent for system-wide optimization."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        super().__init__()
        self.config = config or OptimizationConfig()
        self.patterns: Dict[str, ReasoningPattern] = {}
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.active_optimizations: Set[str] = set()
        self.pattern_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
    async def start(self):
        """Start the agent."""
        logger.info("Starting MetaReasoner agent")
        await self._load_patterns()
        await super().start()
        
    async def stop(self):
        """Stop the agent."""
        logger.info("Stopping MetaReasoner agent")
        await self._save_patterns()
        await super().stop()
        
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming message."""
        try:
            start_time = time.time()
            
            # Process meta-reasoning request
            content = message.content
            if isinstance(content, dict):
                if 'analyze_system' in content:
                    result = await self._analyze_system(content['analyze_system'])
                elif 'optimize_component' in content:
                    result = await self._optimize_component(content['optimize_component'])
                else:
                    result = None
            else:
                result = None
                
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, bool(result))
            
            return Message(
                id=message.id,
                sender=self.agent_id,
                receiver=message.sender,
                content=result
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
            
    async def _load_patterns(self):
        """Load reasoning patterns."""
        patterns_path = Path(__file__).parent / "meta_patterns.json"
        if patterns_path.exists():
            try:
                with open(patterns_path) as f:
                    patterns_data = json.load(f)
                    
                for pattern_data in patterns_data:
                    pattern = ReasoningPattern(**pattern_data)
                    self.patterns[pattern.name] = pattern
                    
            except Exception as e:
                logger.error(f"Error loading patterns: {e}")
                
    async def _save_patterns(self):
        """Save reasoning patterns."""
        patterns_path = Path(__file__).parent / "meta_patterns.json"
        try:
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
                for pattern in self.patterns.values()
            ]
            
            with open(patterns_path, 'w') as f:
                json.dump(patterns_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
            
    async def _analyze_system(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system state and suggest optimizations."""
        try:
            # Find applicable patterns
            applicable_patterns = self._find_applicable_patterns(system_state)
            
            # Generate optimization plan
            optimization_plan = self._generate_optimization_plan(applicable_patterns)
            
            # Update pattern statistics
            self._update_pattern_stats(applicable_patterns)
            
            return {
                'optimization_plan': optimization_plan,
                'applicable_patterns': [p.name for p in applicable_patterns],
                'confidence': self._calculate_confidence(applicable_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing system: {e}")
            return {}
            
    async def _optimize_component(self, 
                                component_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a specific component."""
        try:
            component_id = component_data.get('id')
            if not component_id:
                return {}
                
            # Check if optimization already active
            if component_id in self.active_optimizations:
                return {'status': 'optimization_in_progress'}
                
            self.active_optimizations.add(component_id)
            
            try:
                # Select optimization strategy
                strategy = self._select_optimization_strategy(component_data)
                
                # Apply optimization
                result = await self._apply_optimization(component_id, strategy)
                
                return {
                    'status': 'success',
                    'strategy': strategy,
                    'result': result
                }
                
            finally:
                self.active_optimizations.remove(component_id)
                
        except Exception as e:
            logger.error(f"Error optimizing component: {e}")
            return {'status': 'error', 'message': str(e)}
            
    def _find_applicable_patterns(self, 
                                system_state: Dict[str, Any]) -> List[ReasoningPattern]:
        """Find patterns applicable to current system state."""
        applicable = []
        
        for pattern in self.patterns.values():
            if self._check_pattern_conditions(pattern, system_state):
                applicable.append(pattern)
                
        return sorted(
            applicable,
            key=lambda p: (p.success_rate, -p.avg_processing_time),
            reverse=True
        )
        
    def _check_pattern_conditions(self,
                                pattern: ReasoningPattern,
                                system_state: Dict[str, Any]) -> bool:
        """Check if pattern conditions are met."""
        try:
            for condition in pattern.conditions:
                if not eval(condition, {'state': system_state}):
                    return False
            return True
        except Exception as e:
            logger.error(f"Error checking pattern conditions: {e}")
            return False
            
    def _generate_optimization_plan(self,
                                  patterns: List[ReasoningPattern]) -> List[Dict[str, Any]]:
        """Generate optimization plan from applicable patterns."""
        plan = []
        
        # Sort patterns by dependencies
        sorted_patterns = self._sort_patterns_by_dependencies(patterns)
        
        for pattern in sorted_patterns:
            plan.extend([
                {'action': action, 'pattern': pattern.name}
                for action in pattern.actions
            ])
            
        return plan
        
    def _sort_patterns_by_dependencies(self,
                                     patterns: List[ReasoningPattern]) -> List[ReasoningPattern]:
        """Sort patterns considering dependencies."""
        # Build dependency graph
        graph = {p.name: self.pattern_dependencies[p.name] for p in patterns}
        
        # Topological sort
        sorted_names = []
        visited = set()
        
        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for dep in graph.get(name, []):
                visit(dep)
            sorted_names.append(name)
            
        for pattern in patterns:
            visit(pattern.name)
            
        # Map back to pattern objects
        name_to_pattern = {p.name: p for p in patterns}
        return [name_to_pattern[name] for name in sorted_names]
        
    def _select_optimization_strategy(self,
                                   component_data: Dict[str, Any]) -> List[str]:
        """Select optimization strategy for component."""
        strategies = []
        
        # Check performance metrics
        metrics = component_data.get('metrics', {})
        
        if metrics.get('memory_usage', 0) > self.config.memory_threshold:
            strategies.append('optimize_memory')
            
        if metrics.get('processing_time', 0) > self.config.processing_threshold:
            strategies.append('optimize_processing')
            
        if metrics.get('error_rate', 0) > 0.1:
            strategies.append('optimize_accuracy')
            
        return strategies
        
    async def _apply_optimization(self,
                                component_id: str,
                                strategy: List[str]) -> Dict[str, Any]:
        """Apply optimization strategy to component."""
        results = {}
        
        for step in strategy:
            try:
                # Execute optimization step
                step_result = await self._execute_optimization_step(
                    component_id,
                    step
                )
                results[step] = step_result
                
            except Exception as e:
                logger.error(f"Error in optimization step {step}: {e}")
                results[step] = {'status': 'error', 'message': str(e)}
                
        return results
        
    async def _execute_optimization_step(self,
                                      component_id: str,
                                      step: str) -> Dict[str, Any]:
        """Execute a single optimization step."""
        # Placeholder for actual optimization logic
        return {'status': 'success', 'step': step}
        
    def _update_pattern_stats(self, patterns: List[ReasoningPattern]):
        """Update pattern statistics."""
        for pattern in patterns:
            pattern.usage_count += 1
            
            # Update success rate based on recent performance
            if self.performance_history.get(pattern.name, []):
                recent_success = np.mean(self.performance_history[pattern.name][-100:])
                pattern.success_rate = (
                    0.9 * pattern.success_rate + 0.1 * recent_success
                )
                
    def _calculate_confidence(self, patterns: List[ReasoningPattern]) -> float:
        """Calculate confidence in optimization plan."""
        if not patterns:
            return 0.0
            
        # Weight patterns by usage and success
        weights = [
            pattern.usage_count * pattern.success_rate
            for pattern in patterns
        ]
        
        return sum(weights) / (len(patterns) * max(weights))
        
    def _update_performance_metrics(self,
                                  processing_time: float,
                                  success: bool):
        """Update performance metrics."""
        self.performance_history['processing_time'].append(processing_time)
        self.performance_history['success_rate'].append(float(success))
        
        # Keep history bounded
        max_history = 1000
        for metric in self.performance_history.values():
            if len(metric) > max_history:
                metric[:] = metric[-max_history:]
                
    def optimize_reasoning_patterns(self):
        """Optimize reasoning patterns based on performance."""
        try:
            # Analyze pattern performance
            for pattern in self.patterns.values():
                if pattern.success_rate < 0.8:
                    # Simplify conditions
                    self._simplify_pattern_conditions(pattern)
                elif pattern.avg_processing_time > 0.5:
                    # Optimize actions
                    self._optimize_pattern_actions(pattern)
                    
        except Exception as e:
            logger.error(f"Error optimizing reasoning patterns: {e}")
            
    def _simplify_pattern_conditions(self, pattern: ReasoningPattern):
        """Simplify pattern conditions for better performance."""
        try:
            # Analyze and simplify conditions
            # This is a placeholder for actual condition optimization
            pass
        except Exception as e:
            logger.error(f"Error simplifying pattern conditions: {e}")
            
    def _optimize_pattern_actions(self, pattern: ReasoningPattern):
        """Optimize pattern actions for better performance."""
        try:
            # Analyze and optimize actions
            # This is a placeholder for actual action optimization
            pass
        except Exception as e:
            logger.error(f"Error optimizing pattern actions: {e}")
            
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            'active_optimizations': list(self.active_optimizations),
            'pattern_stats': {
                name: {
                    'success_rate': pattern.success_rate,
                    'usage_count': pattern.usage_count,
                    'avg_processing_time': pattern.avg_processing_time
                }
                for name, pattern in self.patterns.items()
            },
            'performance_metrics': {
                metric: np.mean(values[-100:])
                for metric, values in self.performance_history.items()
            }
        } 