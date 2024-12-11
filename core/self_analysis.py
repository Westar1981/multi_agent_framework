"""
Enhanced self-analysis system with improved monitoring and pattern detection.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import logging
import asyncio
from scipy import stats
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

@dataclass
class MetricSummary:
    """Summary of collected performance metrics."""
    avg_response_time: float
    avg_throughput: float
    error_rate: float
    avg_resource_usage: float
    avg_success_rate: float

@dataclass
class PerformancePattern:
    """Detected performance pattern."""
    pattern_type: str
    confidence: float
    description: str
    affected_metrics: List[str]
    timestamp: datetime

@dataclass
class PerformanceAlert:
    """Performance alert with context."""
    alert_type: str
    severity: str
    metrics: Dict[str, float]
    context: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None

class MetricCollector:
    """Enhanced metric collection and analysis."""
    
    def __init__(self, window_size: int = 1000):
        self.metrics: Dict[str, deque] = {}
        self.timestamps: Dict[str, deque] = {}
        self.window_size = window_size
        self.thresholds: Dict[str, float] = {
            'response_time': 1.0,  # seconds
            'error_rate': 0.1,     # 10%
            'memory_usage': 0.8,   # 80%
class OptimizationSuggestion:
    """Suggested optimization action."""
    category: str
    priority: TaskPriority
    description: str
    expected_impact: Dict[str, float]
    confidence: float

@dataclass
class PerformanceThresholds:
    """Adaptive performance thresholds."""
    response_time_threshold: float
    throughput_threshold: float
    error_rate_threshold: float
    resource_usage_threshold: float
    success_rate_threshold: float

class MetricCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
    def add_metrics(self, metrics: Any, timestamp: Optional[datetime] = None):
        """Add metrics to the collection."""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.metrics_history.append(metrics)
        self.timestamps.append(timestamp)
        
    def get_summary(self) -> MetricSummary:
        """Get summary statistics of collected metrics."""
        if not self.metrics_history:
            return MetricSummary(0.0, 0.0, 0.0, 0.0, 0.0)
            
        metrics_list = list(self.metrics_history)
        return MetricSummary(
            avg_response_time=np.mean([m.response_time for m in metrics_list]),
            avg_throughput=np.mean([m.throughput for m in metrics_list]),
            error_rate=np.mean([m.error_rate for m in metrics_list]),
            avg_resource_usage=np.mean([m.resource_usage for m in metrics_list]),
            avg_success_rate=np.mean([m.task_success_rate for m in metrics_list])
        )

class PatternDetector:
    """Detects patterns in performance metrics."""
    
    def __init__(self, sensitivity: float = 0.8):
        self.sensitivity = sensitivity
        
    def detect_patterns(self, metrics: List[Any], 
                       timestamps: List[datetime]) -> List[PerformancePattern]:
        """Detect patterns in the metrics data."""
        patterns = []
        
        # Detect trends
        if self._detect_trend(metrics, timestamps):
            patterns.append(self._create_trend_pattern(metrics, timestamps))
            
        # Detect anomalies
        anomalies = self._detect_anomalies(metrics)
        patterns.extend(self._create_anomaly_patterns(anomalies, timestamps))
        
        # Detect cycles
        if self._detect_cycles(metrics, timestamps):
            patterns.append(self._create_cycle_pattern(metrics, timestamps))
            
        return patterns
        
    def _detect_trend(self, metrics: List[Any], timestamps: List[datetime]) -> bool:
        """Detect if there's a significant trend in the metrics."""
        if not metrics:
            return False
            
        response_times = [m.response_time for m in metrics]
        slope = np.polyfit(range(len(response_times)), response_times, 1)[0]
        return abs(slope) > self.sensitivity * np.std(response_times)
        
    def _detect_anomalies(self, metrics: List[Any]) -> List[int]:
        """Detect anomalous points in the metrics."""
        if not metrics:
            return []
            
        response_times = np.array([m.response_time for m in metrics])
        mean = np.mean(response_times)
        std = np.std(response_times)
        threshold = 3 * std  # 3-sigma rule
        
        return [i for i, rt in enumerate(response_times) 
                if abs(rt - mean) > threshold]
        
    def _detect_cycles(self, metrics: List[Any], timestamps: List[datetime]) -> bool:
        """Detect if there are cyclic patterns in the metrics."""
        if len(metrics) < 24:  # Need enough data for cycle detection
            return False
            
        response_times = np.array([m.response_time for m in metrics])
        # Simple autocorrelation check
        autocorr = np.correlate(response_times, response_times, mode='full')
        peaks = self._find_peaks(autocorr[len(autocorr)//2:])
        
        return len(peaks) > 1
        
    def _find_peaks(self, data: np.ndarray) -> List[int]:
        """Find peaks in the data."""
        return [i for i in range(1, len(data)-1)
                if data[i-1] < data[i] > data[i+1]]
        
    def _create_trend_pattern(self, metrics: List[Any], 
                            timestamps: List[datetime]) -> PerformancePattern:
        """Create a trend pattern object."""
        response_times = [m.response_time for m in metrics]
        slope = np.polyfit(range(len(response_times)), response_times, 1)[0]
        
        return PerformancePattern(
            pattern_type="degrading_performance" if slope > 0 else "improving_performance",
            confidence=min(1.0, abs(slope) / np.std(response_times)),
            description=f"{'Degrading' if slope > 0 else 'Improving'} performance trend detected",
            affected_metrics=["response_time", "throughput"],
            timestamp=timestamps[-1]
        )
        
    def _create_anomaly_patterns(self, anomaly_indices: List[int],
                               timestamps: List[datetime]) -> List[PerformancePattern]:
        """Create patterns for detected anomalies."""
        return [
            PerformancePattern(
                pattern_type="anomaly",
                confidence=0.9,  # High confidence for 3-sigma anomalies
                description=f"Performance anomaly detected at index {idx}",
                affected_metrics=["response_time"],
                timestamp=timestamps[idx]
            )
            for idx in anomaly_indices
        ]

class OptimizationEngine:
    """Generates optimization suggestions based on patterns."""
    
    def __init__(self):
        self.optimization_rules = {
            "high_resource_usage": {
                "threshold": 0.8,
                "category": "resource_optimization",
                "priority": TaskPriority.HIGH,
                "impact": {"resource_usage": -0.3, "response_time": -0.2}
            },
            "high_error_rate": {
                "threshold": 0.1,
                "category": "reliability_optimization",
                "priority": TaskPriority.CRITICAL,
                "impact": {"error_rate": -0.5, "success_rate": 0.3}
            }
        }
        
    def generate_suggestions(self, metrics: MetricSummary) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions based on current metrics."""
        suggestions = []
        
        # Check resource usage
        if metrics.avg_resource_usage > self.optimization_rules["high_resource_usage"]["threshold"]:
            suggestions.append(OptimizationSuggestion(
                category="resource_optimization",
                priority=TaskPriority.HIGH,
                description="High resource usage detected. Consider scaling or optimization.",
                expected_impact=self.optimization_rules["high_resource_usage"]["impact"],
                confidence=0.9
            ))
            
        # Check error rate
        if metrics.error_rate > self.optimization_rules["high_error_rate"]["threshold"]:
            suggestions.append(OptimizationSuggestion(
                category="reliability_optimization",
                priority=TaskPriority.CRITICAL,
                description="High error rate detected. Review error handling and validation.",
                expected_impact=self.optimization_rules["high_error_rate"]["impact"],
                confidence=0.95
            ))
            
        return suggestions

class SelfAnalysis:
    """Main self-analysis system."""
    
    def __init__(self, window_size: int = 100):
        self.metric_collector = MetricCollector(window_size)
        self.pattern_detector = PatternDetector()
        self.optimization_engine = OptimizationEngine()
        self.thresholds = PerformanceThresholds(
            response_time_threshold=0.1,
            throughput_threshold=100,
            error_rate_threshold=0.05,
            resource_usage_threshold=0.7,
            success_rate_threshold=0.95
        )
        
    def collect_metrics(self, metrics: Any, timestamp: Optional[datetime] = None):
        """Collect new metrics."""
        self.metric_collector.add_metrics(metrics, timestamp)
        
    def detect_patterns(self) -> List[PerformancePattern]:
        """Detect patterns in collected metrics."""
        return self.pattern_detector.detect_patterns(
            list(self.metric_collector.metrics_history),
            list(self.metric_collector.timestamps)
        )
        
    def get_optimization_suggestions(self) -> List[OptimizationSuggestion]:
        """Get optimization suggestions."""
        summary = self.metric_collector.get_summary()
        return self.optimization_engine.generate_suggestions(summary)
        
    def get_metric_summary(self) -> MetricSummary:
        """Get summary of current metrics."""
        return self.metric_collector.get_summary()
        
    def get_current_thresholds(self) -> PerformanceThresholds:
        """Get current performance thresholds."""
        # Update thresholds based on recent performance
        summary = self.metric_collector.get_summary()
        
        # Adapt thresholds using exponential moving average
        alpha = 0.1  # Smoothing factor
        self.thresholds.response_time_threshold = (
            (1 - alpha) * self.thresholds.response_time_threshold +
            alpha * summary.avg_response_time
        )
        self.thresholds.throughput_threshold = (
            (1 - alpha) * self.thresholds.throughput_threshold +
            alpha * summary.avg_throughput
        )
        
        return self.thresholds