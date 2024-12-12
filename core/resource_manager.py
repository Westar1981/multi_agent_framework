"""Resource manager for optimizing system resource usage."""

import gc
import logging
import psutil
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    io_counters: Dict[str, int]
    thread_count: int
    open_handles: int

@dataclass
class ResourceLimits:
    """Resource usage limits."""
    max_memory_percent: float = 80.0
    max_cpu_percent: float = 90.0
    max_threads: int = 100
    max_handles: int = 1000
    max_pool_size: int = 50

class ResourcePool:
    """Pool for reusable resources."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.resources: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        
    def acquire(self) -> Optional[Any]:
        """Acquire a resource from the pool."""
        with self._lock:
            return self.resources.popleft() if self.resources else None
            
    def release(self, resource: Any) -> None:
        """Release a resource back to the pool."""
        with self._lock:
            if len(self.resources) < self.max_size:
                self.resources.append(resource)

class ResourceManager:
    """Manages system resources and optimization."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.pools: Dict[str, ResourcePool] = {}
        self.executor = ThreadPoolExecutor(max_workers=self.limits.max_pool_size)
        self._setup_monitoring()
        
    def _setup_monitoring(self) -> None:
        """Setup resource monitoring."""
        self.process = psutil.Process()
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self._monitor_thread.start()
        
    def _monitor_resources(self) -> None:
        """Monitor resource usage."""
        while True:
            try:
                metrics = self.get_metrics()
                
                # Check thresholds
                if metrics.memory_percent > self.limits.max_memory_percent:
                    self._handle_high_memory()
                    
                if metrics.cpu_percent > self.limits.max_cpu_percent:
                    self._handle_high_cpu()
                    
                if metrics.thread_count > self.limits.max_threads:
                    self._handle_high_threads()
                    
                if metrics.open_handles > self.limits.max_handles:
                    self._handle_high_handles()
                    
            except Exception as e:
                logger.error(f"Error monitoring resources: {str(e)}")
                
            threading.Event().wait(60)  # Check every minute
            
    def _handle_high_memory(self) -> None:
        """Handle high memory usage."""
        logger.warning("High memory usage detected")
        
        # Force garbage collection
        gc.collect()
        
        # Clear pools
        for pool in self.pools.values():
            pool.resources.clear()
            
    def _handle_high_cpu(self) -> None:
        """Handle high CPU usage."""
        logger.warning("High CPU usage detected")
        
        # Reduce thread pool size
        self.executor._max_workers = max(
            1,
            self.executor._max_workers // 2
        )
        
    def _handle_high_threads(self) -> None:
        """Handle high thread count."""
        logger.warning("High thread count detected")
        
        # Force thread pool shutdown
        self.executor.shutdown(wait=False)
        self.executor = ThreadPoolExecutor(
            max_workers=self.limits.max_pool_size
        )
        
    def _handle_high_handles(self) -> None:
        """Handle high handle count."""
        logger.warning("High handle count detected")
        
        # Clear pools to release handles
        for pool in self.pools.values():
            pool.resources.clear()
            
    def get_metrics(self) -> ResourceMetrics:
        """Get current resource metrics."""
        return ResourceMetrics(
            cpu_percent=self.process.cpu_percent(),
            memory_percent=self.process.memory_percent(),
            io_counters=self.process.io_counters()._asdict(),
            thread_count=self.process.num_threads(),
            open_handles=self.process.num_handles()
        )
        
    def create_pool(self, name: str, max_size: int) -> ResourcePool:
        """Create a new resource pool."""
        if name in self.pools:
            raise ValueError(f"Pool {name} already exists")
            
        pool = ResourcePool(max_size=max_size)
        self.pools[name] = pool
        return pool
        
    def get_pool(self, name: str) -> Optional[ResourcePool]:
        """Get a resource pool by name."""
        return self.pools.get(name)
        
    def submit_task(self, fn, *args, **kwargs) -> Any:
        """Submit a task to the thread pool."""
        return self.executor.submit(fn, *args, **kwargs)
        
    def cleanup(self) -> None:
        """Clean up resources."""
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Clear all pools
        for pool in self.pools.values():
            pool.resources.clear()
            
        # Force garbage collection
        gc.collect()
        
        logger.info("Resource cleanup completed")
        
 