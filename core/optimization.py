"""Optimization module for system performance tuning."""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Callable, Protocol, TypeVar, Union
from dataclasses import dataclass, field
import json
import threading
from queue import Queue
from collections import defaultdict
from datetime import datetime
from abc import ABC, abstractmethod

try:
    import numpy as np
    from scipy.optimize import minimize
    from scipy import stats
    Array = np.ndarray
except ImportError:
    np = None
    minimize = None
    stats = None
    Array = List[float]

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
MetricName = str
MetricValue = float
ParamName = str
ParamValue = float

class OptimizationError(Exception):
    """Base class for optimization errors."""
    pass

class ResourceLimitError(OptimizationError):
    """Raised when resource limits are exceeded."""
    pass

class ParameterValidationError(OptimizationError):
    """Raised when parameters fail validation."""
    pass

class OptimizationStrategy(Protocol):
    """Protocol for optimization strategies."""
    
    async def optimize(self, 
                      current_params: Dict[ParamName, ParamValue],
                      metrics: Dict[MetricName, List[MetricValue]]) -> Dict[ParamName, ParamValue]:
        """Optimize parameters based on current state."""
        ...

class MetricPredictor(Protocol):
    """Protocol for metric prediction."""
    
    def predict(self,
               params: Dict[ParamName, ParamValue],
               history: Dict[MetricName, List[MetricValue]]) -> Dict[MetricName, MetricValue]:
        """Predict metrics for parameter set."""
        ...

@dataclass
class OptimizationMetrics:
    """Performance metrics for optimization."""
    throughput: float = 0.0
    latency: float = 0.0
    error_rate: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    network_io: float = 0.0
    disk_io: float = 0.0
    queue_depth: float = 0.0
    cache_hit_rate: float = 0.0
    
    def validate(self) -> None:
        """Validate metric values."""
        for name, value in self.__dict__.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid {name} value: {value}")
            if value < 0:
                raise ValueError(f"Negative {name} value: {value}")
            if name.endswith('_rate') and value > 1:
                raise ValueError(f"Rate exceeds 1.0: {name}={value}")

@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    window_size: int = 100
    update_interval: int = 60
    min_improvement: float = 0.05
    max_regression: float = 0.1
    enable_monitoring: bool = True
    enable_ml_optimization: bool = True
    enable_anomaly_detection: bool = True
    resource_limits: Optional[Dict[str, float]] = None
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    metric_weights: Optional[Dict[str, float]] = None
    optimization_strategies: List[str] = field(default_factory=lambda: [
        'gradient',
        'bayesian',
        'evolutionary'
    ])
    anomaly_threshold: float = 3.0
    history_retention_days: int = 30
    backup_interval: int = 3600
    
    def __post_init__(self) -> None:
        """Initialize and validate configuration."""
        self._init_resource_limits()
        self._init_param_bounds()
        self._init_metric_weights()
        self.validate()
        
    def _init_resource_limits(self) -> None:
        """Initialize resource limits."""
        if self.resource_limits is None:
            self.resource_limits = {
                'memory': 0.8,  # 80% max memory
                'cpu': 0.9,     # 90% max CPU
                'gpu': 0.9,     # 90% max GPU
                'throughput': 1000,  # ops/sec
                'network': 1000,     # MB/s
                'disk': 500         # MB/s
            }
            
    def _init_param_bounds(self) -> None:
        """Initialize parameter bounds."""
        if self.param_bounds is None:
            self.param_bounds = {
                'batch_size': (1, 1000),
                'queue_size': (10, 1000),
                'worker_threads': (1, 32),
                
                'cache_size': (100, 10000),
                'timeout': (1, 60),
                'prefetch_count': (1, 100),
                'retry_limit': (1, 10),
                'buffer_size': (1024, 1024*1024)
            }
            
    def _init_metric_weights(self) -> None:
        """Initialize metric weights."""
        if self.metric_weights is None:
            self.metric_weights = {
                'throughput': 2.0,
                'latency': -1.5,
                'error_rate': -3.0,
                'memory_usage': -1.0,
                'cpu_usage': -1.0,
                'gpu_usage': -1.0,
                'network_io': -0.5,
                'disk_io': -0.5,
                'queue_depth': -0.5,
                'cache_hit_rate': 1.0
            }
            
    def validate(self) -> None:
        """Validate configuration values."""
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.update_interval < 1:
            raise ValueError("update_interval must be positive")
        if not 0 < self.min_improvement < 1:
            raise ValueError("min_improvement must be between 0 and 1")
        if not 0 < self.max_regression < 1:
            raise ValueError("max_regression must be between 0 and 1")
        if self.anomaly_threshold <= 0:
            raise ValueError("anomaly_threshold must be positive")
        if self.history_retention_days < 1:
            raise ValueError("history_retention_days must be positive")
        if self.backup_interval < 1:
            raise ValueError("backup_interval must be positive")

"""Optimization manager for coordinating system-wide optimizations."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for optimization manager."""
    memory_threshold: float = 0.8  # Memory usage threshold for optimization
    cpu_threshold: float = 0.7  # CPU usage threshold for optimization
    error_rate_threshold: float = 0.1  # Error rate threshold for optimization
    optimization_interval: int = 300  # Seconds between optimization runs
    max_concurrent_optimizations: int = 3  # Maximum concurrent optimization tasks
    enable_auto_optimization: bool = True  # Whether to enable automatic optimization

class OptimizationManager:
    """Manages system-wide optimizations."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None) -> None:
        """Initialize optimization manager.
        
        Args:
            config: Optional optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.running_optimizations: List[asyncio.Task[Any]] = []
        self.is_running: bool = False
        self._optimization_lock = asyncio.Lock()
        
    async def optimize(self) -> None:
        """Run system optimization."""
        async with self._optimization_lock:
            if len(self.running_optimizations) >= self.config.max_concurrent_optimizations:
                logger.warning("Maximum concurrent optimizations reached")
                return
                
            try:
                # Create optimization task
                task = asyncio.create_task(self._run_optimization())
                self.running_optimizations.append(task)
                
                # Wait for completion
                await task
                
            except Exception as e:
                logger.error(f"Error during optimization: {str(e)}")
                
            finally:
                if task in self.running_optimizations:
                    self.running_optimizations.remove(task)
                    
    async def _run_optimization(self) -> None:
        """Run optimization steps."""
        try:
            # Memory optimization
            await self._optimize_memory_usage()
            
            # CPU optimization
            await self._optimize_cpu_usage()
            
            # Error handling optimization
            await self._optimize_error_handling()
            
        except Exception as e:
            logger.error(f"Error in optimization run: {str(e)}")
            
    async def _optimize_memory_usage(self) -> None:
        """Optimize system memory usage."""
        logger.info("Running memory optimization")
        # Memory optimization logic here
        
    async def _optimize_cpu_usage(self) -> None:
        """Optimize CPU usage."""
        logger.info("Running CPU optimization")
        # CPU optimization logic here
        
    async def _optimize_error_handling(self) -> None:
        """Optimize error handling."""
        logger.info("Running error handling optimization")
        # Error handling optimization logic here
        
    async def shutdown(self) -> None:
        """Shutdown optimization manager."""
        self.is_running = False
        
        # Cancel running optimizations
        for task in self.running_optimizations:
            task.cancel()
            
        # Wait for tasks to complete
        if self.running_optimizations:
            await asyncio.wait(self.running_optimizations)
            
        self.running_optimizations.clear()
        logger.info("Optimization manager shutdown complete")