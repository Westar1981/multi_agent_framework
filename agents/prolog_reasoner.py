"""
Prolog-based reasoning agent with optimization capabilities.
"""

from typing import Dict, Any, List, Optional, Set
import logging
from dataclasses import dataclass
import time
from pathlib import Path
import json
from collections import defaultdict

try:
    from pyswip import Prolog
except ImportError:
    logging.warning("pyswip not found. Prolog functionality will be limited.")
    Prolog = None

from .base_agent import BaseAgent, Message
from ..core.optimization import OptimizationConfig

logger = logging.getLogger(__name__)

@dataclass
class RuleStats:
    """Statistics for a Prolog rule."""
    usage_count: int = 0
    last_used: float = 0.0
    avg_processing_time: float = 0.0
    success_rate: float = 1.0

class PrologReasoner(BaseAgent):
    """Prolog-based reasoning agent with optimization capabilities."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        super().__init__()
        self.prolog = Prolog() if Prolog else None
        self.config = config or OptimizationConfig()
        self.rule_stats: Dict[str, RuleStats] = defaultdict(RuleStats)
        self.query_cache: Dict[str, Any] = {}
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        
    async def start(self):
        """Start the agent."""
        if not self.prolog:
            logger.error("Prolog engine not available")
            return
            
        logger.info("Starting PrologReasoner agent")
        await super().start()
        
    async def stop(self):
        """Stop the agent."""
        logger.info("Stopping PrologReasoner agent")
        self.cleanup_knowledge_base()
        await super().stop()
        
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming message."""
        if not self.prolog:
            return None
            
        try:
            start_time = time.time()
            
            # Process query
            content = message.content
            if isinstance(content, dict) and 'query' in content:
                result = await self._process_query(content['query'])
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
            
    async def _process_query(self, query: str) -> Optional[Any]:
        """Process a Prolog query with caching."""
        # Check cache
        if query in self.query_cache:
            self._update_stats(query, cached=True)
            return self.query_cache[query]
            
        try:
            start_time = time.time()
            
            # Execute query
            results = list(self.prolog.query(query))
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(query, processing_time=processing_time)
            
            # Cache results if useful
            if self._should_cache(query, results):
                self.query_cache[query] = results
                
            return results
            
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return None
            
    def cleanup_knowledge_base(self):
        """Clean up unused rules from knowledge base."""
        if not self.prolog:
            return
            
        try:
            # Get rules sorted by usage
            sorted_rules = sorted(
                self.rule_stats.items(),
                key=lambda x: (x[1].usage_count, -x[1].last_used)
            )
            
            # Remove least used rules
            for rule, stats in sorted_rules:
                if (stats.usage_count == 0 or 
                    time.time() - stats.last_used > 3600):  # 1 hour threshold
                    self.prolog.retract(rule)
                    del self.rule_stats[rule]
                    
        except Exception as e:
            logger.error(f"Error cleaning knowledge base: {e}")
            
    def _update_stats(self, 
                     query: str,
                     processing_time: Optional[float] = None,
                     cached: bool = False):
        """Update query statistics."""
        stats = self.rule_stats[query]
        stats.usage_count += 1
        stats.last_used = time.time()
        
        if processing_time is not None:
            # Update average processing time
            stats.avg_processing_time = (
                (stats.avg_processing_time * (stats.usage_count - 1) + processing_time)
                / stats.usage_count
            )
            
    def _should_cache(self, query: str, results: List[Any]) -> bool:
        """Determine if query results should be cached."""
        stats = self.rule_stats[query]
        
        # Cache if:
        # 1. Frequently used (high usage count)
        # 2. Expensive to compute (high processing time)
        # 3. Cache isn't too full
        return (
            stats.usage_count > 5 and
            stats.avg_processing_time > 0.1 and
            len(self.query_cache) < self.config.cache_size
        )
        
    def _update_performance_metrics(self, 
                                  processing_time: float,
                                  success: bool):
        """Update agent performance metrics."""
        self.performance_history['processing_time'].append(processing_time)
        self.performance_history['success_rate'].append(float(success))
        
        # Keep history bounded
        max_history = 1000
        if len(self.performance_history['processing_time']) > max_history:
            self.performance_history['processing_time'] = \
                self.performance_history['processing_time'][-max_history:]
        if len(self.performance_history['success_rate']) > max_history:
            self.performance_history['success_rate'] = \
                self.performance_history['success_rate'][-max_history:]
                
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return {
            'processing_time': sum(self.performance_history['processing_time']) / 
                             max(1, len(self.performance_history['processing_time'])),
            'success_rate': sum(self.performance_history['success_rate']) /
                          max(1, len(self.performance_history['success_rate'])),
            'cache_size': len(self.query_cache),
            'rule_count': len(self.rule_stats)
        }
        
    def refine_inference_rules(self):
        """Refine inference rules based on performance."""
        if not self.prolog:
            return
            
        try:
            # Analyze rule performance
            for rule, stats in self.rule_stats.items():
                if stats.success_rate < 0.8:  # Low success rate
                    # Add validation or constraints
                    self._add_rule_constraints(rule)
                elif stats.avg_processing_time > 0.5:  # Slow processing
                    # Optimize rule structure
                    self._optimize_rule_structure(rule)
                    
        except Exception as e:
            logger.error(f"Error refining inference rules: {e}")
            
    def _add_rule_constraints(self, rule: str):
        """Add constraints to improve rule accuracy."""
        try:
            # Parse rule and add type checks or range constraints
            # This is a placeholder for actual rule refinement logic
            pass
        except Exception as e:
            logger.error(f"Error adding rule constraints: {e}")
            
    def _optimize_rule_structure(self, rule: str):
        """Optimize rule structure for better performance."""
        try:
            # Analyze and optimize rule structure
            # This is a placeholder for actual optimization logic
            pass
        except Exception as e:
            logger.error(f"Error optimizing rule structure: {e}")
            
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            'metrics': self.get_performance_metrics(),
            'rule_stats': {
                rule: stats.__dict__
                for rule, stats in self.rule_stats.items()
            },
            'cache_stats': {
                'size': len(self.query_cache),
                'max_size': self.config.cache_size
            }
        }
``` 