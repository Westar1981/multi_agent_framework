"""
Enhanced optimization system with improved memory and performance management.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
from collections import deque
import psutil
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for optimization system."""
    # Memory management
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds
    memory_threshold: float = 0.8
    
    # Performance tuning
    initial_batch_size: int = 32
    max_batch_size: int = 256
    min_batch_size: int = 8
    
    # Resource management
    max_workers: int = 4
    max_queue_size: int = 1000
    
    # Optimization parameters
    learning_rate: float = 0.01
    adaptation_threshold: float = 0.1

class MemoryManager:
    """Enhanced memory management system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache: Dict[str, Any] = {}
        self.cache_times: Dict[str, float] = {}
        self.memory_usage_history = deque(maxlen=100)
        
    def cache_get(self, key: str) -> Optional[Any]:
        """Get item from cache with TTL check."""
        if key in self.cache:
            if time.time() - self.cache_times[key] > self.config.cache_ttl:
                del self.cache[key]
                del self.cache_times[key]
                return None
            return self.cache[key]
        return None
        
    def cache_set(self, key: str, value: Any):
        """Set cache item with memory check."""
        current_memory = psutil.Process().memory_percent()
        self.memory_usage_history.append(current_memory)
        
        if current_memory > self.config.memory_threshold:
            self._cleanup_cache()
            
        if len(self.cache) < self.config.cache_size:
            self.cache[key] = value
            self.cache_times[key] = time.time()
            
    def _cleanup_cache(self):
        """Smart cache cleanup based on usage patterns."""
        if not self.cache:
            return
            
        # Remove expired items
        current_time = time.time()
        expired = [
            k for k, t in self.cache_times.items()
            if current_time - t > self.config.cache_ttl
        ]
        for k in expired:
            del self.cache[k]
            del self.cache_times[k]
            
        # If still need cleanup
        if len(self.cache) > self.config.cache_size * 0.8:
            # Remove oldest items
            sorted_items = sorted(
                self.cache_times.items(),
                key=lambda x: x[1]
            )
            to_remove = sorted_items[:len(sorted_items)//4]
            for k, _ in to_remove:
                del self.cache[k]
                del self.cache_times[k]

class PerformanceOptimizer:
    """Enhanced performance optimization system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.batch_size = config.initial_batch_size
        self.processing_times = deque(maxlen=100)
        self.batch_history = deque(maxlen=100)
        
    async def optimize_batch_size(self, current_load: float):
        """Adapt batch size based on load and performance."""
        self.batch_history.append((self.batch_size, current_load))
        
        if len(self.batch_history) < 10:
            return
            
        recent_performance = self._analyze_performance()
        
        if recent_performance['trend'] > 0:  # Performance improving
            self.batch_size = min(
                self.batch_size * (1 + self.config.learning_rate),
                self.config.max_batch_size
            )
        else:  # Performance degrading
            self.batch_size = max(
                self.batch_size * (1 - self.config.learning_rate),
                self.config.min_batch_size
            )
            
    def _analyze_performance(self) -> Dict[str, float]:
        """Analyze recent performance trends."""
        recent_batches = list(self.batch_history)
        batch_sizes, loads = zip(*recent_batches)
        
        # Calculate trends
        batch_trend = np.polyfit(range(len(batch_sizes)), batch_sizes, 1)[0]
        load_trend = np.polyfit(range(len(loads)), loads, 1)[0]
        
        return {
            'trend': batch_trend,
            'load_trend': load_trend,
            'avg_load': np.mean(loads)
        }
        
    def record_processing_time(self, duration: float):
        """Record processing time for optimization."""
        self.processing_times.append(duration)

class ResourceManager:
    """Enhanced resource management system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.active_workers = 0
        self.queue_size = 0
        self.resource_usage = deque(maxlen=100)
        
    async def acquire_worker(self) -> bool:
        """Acquire worker with resource check."""
        if self.active_workers >= self.config.max_workers:
            return False
            
        current_usage = psutil.Process().cpu_percent()
        self.resource_usage.append(current_usage)
        
        if current_usage > 80:  # High CPU usage
            return False
            
        self.active_workers += 1
        return True
        
    def release_worker(self):
        """Release worker."""
        self.active_workers = max(0, self.active_workers - 1)
        
    def can_queue_task(self) -> bool:
        """Check if can queue new task."""
        return self.queue_size < self.config.max_queue_size
        
    def get_resource_stats(self) -> Dict[str, float]:
        """Get resource usage statistics."""
        return {
            'cpu_usage': np.mean(list(self.resource_usage)) if self.resource_usage else 0,
            'worker_usage': self.active_workers / self.config.max_workers,
            'queue_usage': self.queue_size / self.config.max_queue_size
        }

class OptimizationManager:
    """Main optimization manager."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.memory_manager = MemoryManager(self.config)
        self.performance_optimizer = PerformanceOptimizer(self.config)
        self.resource_manager = ResourceManager(self.config)
        
    async def optimize_operation(self, operation_type: str, load: float):
        """Optimize operation execution."""
        stats = {}
        
        # Memory optimization
        current_memory = psutil.Process().memory_percent()
        if current_memory > self.config.memory_threshold:
            self.memory_manager._cleanup_cache()
            stats['memory_cleaned'] = True
            
        # Performance optimization
        await self.performance_optimizer.optimize_batch_size(load)
        stats['batch_size'] = self.performance_optimizer.batch_size
        
        # Resource optimization
        if await self.resource_manager.acquire_worker():
            try:
                stats['worker_acquired'] = True
                # Operation would go here
            finally:
                self.resource_manager.release_worker()
                
        return stats
        
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get current optimization metrics."""
        return {
            'memory_usage': len(self.memory_manager.cache) / self.config.cache_size,
            'batch_size': self.performance_optimizer.batch_size,
            'resource_stats': self.resource_manager.get_resource_stats()
        }
        
    async def adapt_to_load(self, load_metrics: Dict[str, float]):
        """Adapt optimization parameters to current load."""
        if load_metrics.get('memory_pressure', 0) > 0.8:
            self.config.cache_size *= 0.8
            await self.memory_manager._cleanup_cache()
            
        if load_metrics.get('cpu_pressure', 0) > 0.8:
            self.config.max_workers = max(2, self.config.max_workers - 1)
            
        if load_metrics.get('queue_pressure', 0) > 0.8:
            self.config.max_queue_size *= 1.2 