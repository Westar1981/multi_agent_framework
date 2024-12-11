"""
Enhanced memory management system with smart caching and optimization.
"""

import time
import sys
from typing import Dict, Any, Optional, List, Tuple, Union, NoReturn
from collections import defaultdict, deque
import psutil
import numpy as np
from dataclasses import dataclass
import logging
import asyncio
from concurrent.futures import TimeoutError

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Memory management configuration."""
    max_cache_size: int = 10000
    cache_ttl: int = 3600  # seconds
    memory_threshold: float = 0.8
    cleanup_interval: int = 300  # seconds
    min_hit_rate: float = 0.2
    max_item_size: int = 1024 * 1024  # 1MB
    monitoring_timeout: int = 30  # seconds

class CacheItem:
    """Cache item with metadata."""
    def __init__(self, key: str, value: Any, size: int):
        self.key = key
        self.value = value
        self.size = size
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.hit_rate = 0.0

class MemoryManager:
    """Enhanced memory management system."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.cache: Dict[str, CacheItem] = {}
        self.access_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.size_history = deque(maxlen=100)
        self.last_cleanup = time.time()
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'memory_pressure': 0.0
        }
        
    async def start_monitoring(self) -> NoReturn:
        """Start background monitoring."""
        while True:
            try:
                await asyncio.wait_for(
                    self._monitor_memory(),
                    timeout=self.config.monitoring_timeout
                )
                await asyncio.sleep(10)  # Check every 10 seconds
            except TimeoutError:
                logger.error("Memory monitoring timed out")
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(30)  # Back off on error
                
    async def _monitor_memory(self) -> None:
        """Monitor memory usage and trigger cleanup if needed."""
        try:
            current_memory = psutil.Process().memory_percent()
            self.metrics['memory_pressure'] = current_memory / 100.0
            
            if current_memory / 100.0 > self.config.memory_threshold:
                await self.cleanup(aggressive=True)
        except psutil.NoSuchProcess:
            logger.error("Process not found during memory monitoring")
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with smart access tracking."""
        if key not in self.cache:
            self.metrics['cache_misses'] += 1
            # Clean up access history for non-existent keys
            if key in self.access_history:
                del self.access_history[key]
            return None
            
        item = self.cache[key]
        current_time = time.time()
        
        # Check TTL
        if current_time - item.created_at > self.config.cache_ttl:
            del self.cache[key]
            self.metrics['evictions'] += 1
            return None
            
        # Update access metrics
        item.last_accessed = current_time
        item.access_count += 1
        self.access_history[key].append(current_time)
        self.metrics['cache_hits'] += 1
        
        # Update hit rate
        total_accesses = self.metrics['cache_hits'] + self.metrics['cache_misses']
        item.hit_rate = item.access_count / total_accesses if total_accesses > 0 else 0
        
        return item.value
        
    async def set(self, key: str, value: Any):
        """Set cache item with size checking and optimization."""
        # Calculate item size
        size = self._estimate_size(value)
        
        if size > self.config.max_item_size:
            logger.warning(f"Item too large to cache: {size} bytes")
            return False
            
        # Check if cleanup needed
        current_time = time.time()
        if current_time - self.last_cleanup > self.config.cleanup_interval:
            await self.cleanup()
            
        # Add item
        self.cache[key] = CacheItem(key, value, size)
        self.size_history.append(size)
        
        return True
        
    async def cleanup(self, aggressive: bool = False):
        """Smart cache cleanup based on usage patterns."""
        self.last_cleanup = time.time()
        
        if not self.cache:
            return
            
        # Calculate scores for each item
        scores: List[Tuple[str, float]] = []
        for key, item in self.cache.items():
            score = self._calculate_item_score(item, aggressive)
            scores.append((key, score))
            
        # Sort by score (lower is better to keep)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Remove items until under threshold
        target_size = len(self.cache) * (0.6 if aggressive else 0.8)
        for key, _ in scores[int(target_size):]:
            del self.cache[key]
            self.metrics['evictions'] += 1
            
    def _calculate_item_score(self, item: CacheItem, aggressive: bool) -> float:
        """Calculate item score for eviction (higher = more likely to evict)."""
        try:
            current_time = time.time()
            
            # Base score components
            age_factor = (current_time - item.created_at) / self.config.cache_ttl
            size_factor = item.size / self.config.max_item_size
            hit_factor = 1 - item.hit_rate
            
            # Recent access pattern
            recent_accesses = sum(1 for t in self.access_history[item.key]
                                if current_time - t < 300)  # Last 5 minutes
            recency_factor = 1 - (recent_accesses / 10 if recent_accesses < 10 else 1)
            
            # Combine factors
            if aggressive:
                # Prioritize size and hit rate in aggressive mode
                score = 0.4 * size_factor + 0.3 * hit_factor + \
                       0.2 * age_factor + 0.1 * recency_factor
            else:
                # Balance all factors in normal mode
                score = 0.25 * size_factor + 0.25 * hit_factor + \
                       0.25 * age_factor + 0.25 * recency_factor
                       
            return score
        except Exception as e:
            logger.error(f"Error calculating item score: {e}")
            return float('inf')  # Ensure problematic items are evicted
            
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            if hasattr(value, '__sizeof__'):
                return value.__sizeof__()
            return len(str(value).encode('utf-8'))
        except Exception as e:
            logger.error(f"Error estimating size: {e}")
            return sys.getsizeof(value)
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current memory metrics."""
        total_size = sum(item.size for item in self.cache.values())
        
        return {
            **self.metrics,
            'cache_size': len(self.cache),
            'total_size_bytes': total_size,
            'hit_rate': self.metrics['cache_hits'] / 
                       (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                       if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0
                       else 0,
            'avg_item_size': np.mean(list(self.size_history)) if self.size_history else 0
        }
        
    async def optimize(self):
        """Optimize memory usage based on metrics."""
        metrics = self.get_metrics()
        
        # Adjust cache size based on hit rate
        if metrics['hit_rate'] < self.config.min_hit_rate:
            self.config.max_cache_size = int(self.config.max_cache_size * 0.8)
            await self.cleanup()
            
        # Adjust TTL based on memory pressure
        if metrics['memory_pressure'] > self.config.memory_threshold:
            self.config.cache_ttl = int(self.config.cache_ttl * 0.8)
            
        return metrics 