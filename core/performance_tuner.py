"""Performance tuner for automatic system optimization."""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    response_time: float
    throughput: float
    error_rate: float
    resource_usage: float
    memory_usage: float
    cpu_usage: float
    latency: float

@dataclass
class TuningConfig:
    """Configuration for performance tuning."""
    # Tuning intervals
    measurement_window: int = 100  # Number of operations to measure
    tuning_interval: int = 300  # Seconds between tuning attempts
    max_tuning_attempts: int = 5  # Maximum optimization attempts
    
    # Thresholds
    min_improvement: float = 0.05  # Minimum improvement to accept changes
    max_regression: float = 0.1  # Maximum allowed performance regression
    
    # Optimization bounds
    param_bounds: Dict[str, Tuple[float, float]] = None
    
    # Resource limits
    max_memory_usage: float = 0.8  # 80% of available memory
    max_cpu_usage: float = 0.9  # 90% of CPU
    
    def __post_init__(self):
        if self.param_bounds is None:
            self.param_bounds = {
                'batch_size': (1, 1000),
                'queue_size': (10, 1000),
                'worker_threads': (1, 32),
                'cache_size': (100, 10000),
                'timeout': (1, 60)
            }

class PerformanceTuner:
    """Automatic performance tuner."""
    
    def __init__(self, config: TuningConfig):
        self.config = config
        self.metrics_history: Dict[str, deque] = {
            'response_time': deque(maxlen=config.measurement_window),
            'throughput': deque(maxlen=config.measurement_window),
            'error_rate': deque(maxlen=config.measurement_window),
            'resource_usage': deque(maxlen=config.measurement_window),
            'memory_usage': deque(maxlen=config.measurement_window),
            'cpu_usage': deque(maxlen=config.measurement_window),
            'latency': deque(maxlen=config.measurement_window)
        }
        self.current_params: Dict[str, float] = {}
        self.best_params: Dict[str, float] = {}
        self.best_score: float = float('-inf')
        self.tuning_attempts: int = 0
        self._setup_tasks()
        
    def _setup_tasks(self) -> None:
        """Setup background tasks."""
        self._tuning_task = asyncio.create_task(self._tuning_loop())
        
    async def _tuning_loop(self) -> None:
        """Periodic performance tuning."""
        while True:
            try:
                if self.tuning_attempts < self.config.max_tuning_attempts:
                    await self._tune_performance()
                    self.tuning_attempts += 1
                    
            except Exception as e:
                logger.error(f"Error in tuning loop: {str(e)}")
                
            await asyncio.sleep(self.config.tuning_interval)
            
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        for field, value in metrics.__dict__.items():
            if field in self.metrics_history:
                self.metrics_history[field].append(value)
                
    def _calculate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score."""
        # Higher is better for throughput
        # Lower is better for others
        score = (
            metrics['throughput'] * 2.0 -
            metrics['response_time'] * 1.5 -
            metrics['error_rate'] * 3.0 -
            metrics['resource_usage'] * 1.0 -
            metrics['latency'] * 1.0
        )
        
        # Apply penalties for resource limits
        if metrics['memory_usage'] > self.config.max_memory_usage:
            score -= (metrics['memory_usage'] - self.config.max_memory_usage) * 10
            
        if metrics['cpu_usage'] > self.config.max_cpu_usage:
            score -= (metrics['cpu_usage'] - self.config.max_cpu_usage) * 10
            
        return score
        
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current average metrics."""
        return {
            name: np.mean(list(values)) if values else 0
            for name, values in self.metrics_history.items()
        }
        
    async def _tune_performance(self) -> None:
        """Perform performance tuning."""
        current_metrics = self._get_current_metrics()
        if not current_metrics['throughput']:  # No data yet
            return
            
        current_score = self._calculate_score(current_metrics)
        
        # Try to optimize parameters
        result = await self._optimize_parameters(current_metrics)
        
        if result.success:
            # Test new parameters
            new_params = self._result_to_params(result.x)
            await self._test_parameters(new_params)
            
            # Get new metrics
            new_metrics = self._get_current_metrics()
            new_score = self._calculate_score(new_metrics)
            
            # Check if improvement is significant
            if new_score > current_score * (1 + self.config.min_improvement):
                logger.info("Performance improvement found")
                self.best_params = new_params
                self.best_score = new_score
                await self._apply_parameters(new_params)
            else:
                logger.info("No significant improvement found")
                # Revert to previous parameters
                await self._apply_parameters(self.current_params)
                
    async def _optimize_parameters(self, 
                                 current_metrics: Dict[str, float]) -> Any:
        """Optimize system parameters."""
        # Define objective function
        def objective(x):
            params = self._result_to_params(x)
            predicted_metrics = self._predict_metrics(params, current_metrics)
            return -self._calculate_score(predicted_metrics)  # Minimize negative score
            
        # Initial guess
        x0 = self._params_to_array(self.current_params)
        
        # Parameter bounds
        bounds = [
            self.config.param_bounds[name]
            for name in self.current_params.keys()
        ]
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        return result
        
    def _predict_metrics(self, 
                        params: Dict[str, float], 
                        current_metrics: Dict[str, float]) -> Dict[str, float]:
        """Predict metrics for parameter set."""
        # Simple prediction model
        predicted = current_metrics.copy()
        
        # Batch size effects
        batch_ratio = params['batch_size'] / self.current_params.get('batch_size', 1)
        predicted['throughput'] *= np.sqrt(batch_ratio)  # Sub-linear scaling
        predicted['latency'] *= np.sqrt(batch_ratio)  # Increased latency
        
        # Worker threads effects
        worker_ratio = params['worker_threads'] / self.current_params.get('worker_threads', 1)
        predicted['throughput'] *= np.sqrt(worker_ratio)  # Sub-linear scaling
        predicted['cpu_usage'] *= worker_ratio  # Linear CPU usage
        
        # Queue size effects
        queue_ratio = params['queue_size'] / self.current_params.get('queue_size', 1)
        predicted['memory_usage'] *= queue_ratio  # Linear memory usage
        
        # Cache size effects
        cache_ratio = params['cache_size'] / self.current_params.get('cache_size', 1)
        predicted['memory_usage'] *= np.sqrt(cache_ratio)  # Sub-linear memory usage
        predicted['response_time'] /= np.sqrt(cache_ratio)  # Improved response time
        
        return predicted
        
    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameters dict to array."""
        return np.array([params[name] for name in sorted(params.keys())])
        
    def _result_to_params(self, x: np.ndarray) -> Dict[str, float]:
        """Convert optimization result to parameters dict."""
        return {
            name: float(value)
            for name, value in zip(sorted(self.current_params.keys()), x)
        }
        
    async def _test_parameters(self, params: Dict[str, float]) -> None:
        """Test new parameters temporarily."""
        # Apply parameters
        await self._apply_parameters(params)
        
        # Wait for measurements
        await asyncio.sleep(10)
        
    async def _apply_parameters(self, params: Dict[str, float]) -> None:
        """Apply new system parameters."""
        self.current_params = params
        # Actual parameter application would happen here
        # This would involve updating various system components
        
    def get_stats(self) -> Dict[str, Any]:
        """Get tuning statistics."""
        return {
            'current_params': self.current_params,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'tuning_attempts': self.tuning_attempts,
            'metrics': {
                name: {
                    'mean': np.mean(list(values)) if values else 0,
                    'std': np.std(list(values)) if values else 0
                }
                for name, values in self.metrics_history.items()
            }
        }
        
    async def shutdown(self) -> None:
        """Shutdown tuner."""
        if self._tuning_task:
            self._tuning_task.cancel()
            
        # Clear history
        for queue in self.metrics_history.values():
            queue.clear()
            
        self.current_params.clear()
        self.best_params.clear() 