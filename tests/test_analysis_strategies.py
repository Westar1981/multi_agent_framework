"""
Tests for analysis strategies with performance benchmarks.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
import asyncio
import time
import statistics
from dataclasses import dataclass

from ..core.analysis_strategies import (
    AnalysisStrategy,
    TrendAnalysis,
    CyclicAnalysis,
    AnomalyDetection,
    CompoundAnalysis
)
from ..core.tool_management import ToolOrchestrator, ToolRegistry, Tool, ToolResult

@dataclass
class BenchmarkResult:
    """Results from performance benchmark."""
    mean_time: float
    std_dev: float
    min_time: float
    max_time: float
    num_iterations: int
    pattern_count: int

async def run_benchmark(strategy: AnalysisStrategy, 
                       data: List[float], 
                       iterations: int = 10) -> BenchmarkResult:
    """Run performance benchmark on strategy."""
    times = []
    pattern_counts = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        patterns = await strategy.analyze(data)
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
        pattern_counts.append(len(patterns))
        
    return BenchmarkResult(
        mean_time=statistics.mean(times),
        std_dev=statistics.stdev(times) if len(times) > 1 else 0,
        min_time=min(times),
        max_time=max(times),
        num_iterations=iterations,
        pattern_count=statistics.mean(pattern_counts)
    )

class MockAnalysisTool(Tool):
    """Mock tool for testing analysis strategies."""
    
    def __init__(self, name: str, results: Dict[str, Any], latency: float = 0.1):
        super().__init__(name, f"Mock {name} tool")
        self.results = results
        self.latency = latency
        
    async def execute(self, *args, **kwargs) -> ToolResult:
        """Return predefined results with simulated latency."""
        await asyncio.sleep(self.latency)
        return ToolResult(
            success=True,
            result=self.results,
            execution_time=self.latency
        )

@pytest.fixture
def sample_data_generator():
    """Create parameterized sample data generator."""
    def _generate(length: int = 100, 
                 trend_slope: float = 0.5,
                 cycle_amplitude: float = 2.0,
                 noise_std: float = 0.2,
                 num_anomalies: int = 2) -> Tuple[List[float], Dict[str, Any]]:
        """Generate synthetic data with known patterns."""
        x = np.linspace(0, 10, length)
        trend = trend_slope * x
        cycle = cycle_amplitude * np.sin(x)
        noise = np.random.normal(0, noise_std, length)
        
        data = trend + cycle + noise
        
        # Add anomalies
        anomaly_positions = np.random.choice(length, num_anomalies, replace=False)
        anomaly_values = np.random.choice([-5, 5], num_anomalies)
        for pos, val in zip(anomaly_positions, anomaly_values):
            data[pos] += val
            
        ground_truth = {
            'trend': {
                'slope': trend_slope,
                'type': 'linear'
            },
            'cycle': {
                'amplitude': cycle_amplitude,
                'frequency': 1.0
            },
            'anomalies': list(zip(anomaly_positions.tolist(), anomaly_values.tolist()))
        }
        
        return data.tolist(), ground_truth
        
    return _generate

@pytest.fixture
def tool_orchestrator():
    """Create tool orchestrator with mock tools."""
    registry = ToolRegistry()
    
    # Register mock analysis tools with different latencies
    tools = [
        MockAnalysisTool("trend_analyzer", {
            "patterns": [
                {
                    "type": "accelerating",
                    "confidence": 0.9,
                    "metrics": ["trend_advanced"],
                    "description": "Advanced trend pattern detected"
                }
            ]
        }, latency=0.05),
        
        MockAnalysisTool("cycle_detector", {
            "cycles": [
                {
                    "confidence": 0.85,
                    "affected_metrics": ["cycle_advanced"],
                    "description": "Advanced cyclic pattern detected"
                }
            ]
        }, latency=0.1),
        
        MockAnalysisTool("anomaly_detector", {
            "anomalies": [
                {
                    "confidence": 0.95,
                    "metrics": ["anomaly_advanced"],
                    "description": "Advanced anomaly detected"
                }
            ]
        }, latency=0.15)
    ]
    
    for tool in tools:
        registry.register_tool(tool)
    
    return ToolOrchestrator(registry)

class TestTrendAnalysis:
    """Test trend analysis strategy."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("trend_slope", [-1.0, -0.5, 0.0, 0.5, 1.0])
    async def test_trend_detection_sensitivity(self, sample_data_generator, trend_slope):
        """Test trend detection with different slopes."""
        data, ground_truth = sample_data_generator(trend_slope=trend_slope)
        strategy = TrendAnalysis()
        patterns = await strategy.analyze(data)
        
        if abs(trend_slope) > 0.1:
            assert len(patterns) > 0
            pattern_type = 'accelerating' if trend_slope > 0 else 'decelerating'
            assert any(p.pattern_type == pattern_type for p in patterns)
        else:
            # Should not detect significant trend for near-zero slope
            assert not any(p.pattern_type in ['accelerating', 'decelerating'] 
                         for p in patterns)
    
    @pytest.mark.asyncio
    async def test_trend_detection_with_noise(self, sample_data_generator):
        """Test trend detection robustness to noise."""
        noise_levels = [0.1, 0.5, 1.0, 2.0]
        trend_slope = 1.0
        
        for noise in noise_levels:
            data, _ = sample_data_generator(trend_slope=trend_slope, noise_std=noise)
            strategy = TrendAnalysis()
            patterns = await strategy.analyze(data)
            
            # Should detect trend despite noise, but confidence should decrease
            assert len(patterns) > 0
            confidences = [p.confidence for p in patterns]
            avg_confidence = sum(confidences) / len(confidences)
            
            # Confidence should decrease with noise
            assert avg_confidence > 0.5  # Should still be reasonably confident
            
    @pytest.mark.asyncio
    async def test_performance_scaling(self, sample_data_generator):
        """Test performance scaling with data size."""
        data_sizes = [100, 500, 1000, 5000]
        strategy = TrendAnalysis()
        
        scaling_results = []
        for size in data_sizes:
            data, _ = sample_data_generator(length=size)
            benchmark = await run_benchmark(strategy, data, iterations=5)
            scaling_results.append(benchmark)
            
        # Verify sub-quadratic time complexity
        for i in range(len(data_sizes) - 1):
            size_ratio = data_sizes[i+1] / data_sizes[i]
            time_ratio = scaling_results[i+1].mean_time / scaling_results[i].mean_time
            assert time_ratio < size_ratio ** 2

class TestCyclicAnalysis:
    """Test cyclic pattern analysis."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("amplitude", [0.5, 1.0, 2.0, 4.0])
    async def test_amplitude_sensitivity(self, sample_data_generator, amplitude):
        """Test cycle detection with different amplitudes."""
        data, _ = sample_data_generator(cycle_amplitude=amplitude)
        strategy = CyclicAnalysis()
        patterns = await strategy.analyze(data)
        
        if amplitude > 1.0:
            assert len(patterns) > 0
            assert any(p.pattern_type == 'cyclic' for p in patterns)
            # Confidence should increase with amplitude
            max_confidence = max(p.confidence for p in patterns)
            assert max_confidence > 0.3
            
    @pytest.mark.asyncio
    async def test_multiple_frequencies(self, sample_data_generator):
        """Test detection of multiple cyclic components."""
        x = np.linspace(0, 10, 1000)
        # Combine two frequencies
        data = (np.sin(x) + 0.5 * np.sin(2 * x)).tolist()
        
        strategy = CyclicAnalysis()
        patterns = await strategy.analyze(data)
        
        # Should detect multiple cycles
        assert len(patterns) >= 2
        assert all(p.pattern_type == 'cyclic' for p in patterns)
        
    @pytest.mark.asyncio
    async def test_phase_invariance(self, sample_data_generator):
        """Test that detection is phase-invariant."""
        phase_shifts = [0, np.pi/4, np.pi/2, np.pi]
        base_confidence = None
        
        for phase in phase_shifts:
            x = np.linspace(0, 10, 100)
            data = np.sin(x + phase).tolist()
            
            strategy = CyclicAnalysis()
            patterns = await strategy.analyze(data)
            
            if base_confidence is None:
                base_confidence = max(p.confidence for p in patterns)
            else:
                current_confidence = max(p.confidence for p in patterns)
                # Confidence should be similar regardless of phase
                assert abs(current_confidence - base_confidence) < 0.1

class TestAnomalyDetection:
    """Test anomaly detection."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("anomaly_size", [2, 3, 4, 5])
    async def test_anomaly_size_sensitivity(self, sample_data_generator, anomaly_size):
        """Test sensitivity to anomaly magnitude."""
        x = np.linspace(0, 10, 100)
        data = np.sin(x).tolist()
        # Add single anomaly
        data[50] += anomaly_size
        
        strategy = AnomalyDetection()
        patterns = await strategy.analyze(data)
        
        if anomaly_size >= 3:  # Should detect clear anomalies
            assert len(patterns) > 0
            assert patterns[0].confidence > 0.5
        
    @pytest.mark.asyncio
    async def test_contextual_anomalies(self, sample_data_generator):
        """Test detection of contextual anomalies."""
        x = np.linspace(0, 10, 100)
        base = np.sin(x)
        # Add contextual anomaly (normal value in wrong context)
        anomaly_idx = 25
        base[anomaly_idx] = base[anomaly_idx + 50]  # Use value from different phase
        
        strategy = AnomalyDetection()
        patterns = await strategy.analyze(base.tolist())
        
        assert len(patterns) > 0
        assert any(abs(int(p.description.split()[-2]) - anomaly_idx) < 3 
                  for p in patterns)
        
    @pytest.mark.asyncio
    async def test_collective_anomalies(self, sample_data_generator):
        """Test detection of collective anomalies."""
        x = np.linspace(0, 10, 100)
        data = np.sin(x).tolist()
        # Add sequence of unusual values
        for i in range(45, 55):
            data[i] += 2
            
        strategy = AnomalyDetection()
        patterns = await strategy.analyze(data)
        
        # Should detect anomalous sequence
        assert len(patterns) > 0
        anomaly_positions = [int(p.description.split()[-2]) for p in patterns]
        assert max(anomaly_positions) - min(anomaly_positions) >= 5

class TestCompoundAnalysis:
    """Test compound analysis strategy."""
    
    @pytest.mark.asyncio
    async def test_pattern_interaction(self, sample_data_generator):
        """Test detection of interacting patterns."""
        # Generate data with trend, cycle, and anomalies
        data, ground_truth = sample_data_generator(
            trend_slope=1.0,
            cycle_amplitude=2.0,
            num_anomalies=3
        )
        
        strategy = CompoundAnalysis()
        patterns = await strategy.analyze(data)
        
        # Should detect all pattern types
        pattern_types = {p.pattern_type for p in patterns}
        assert 'accelerating' in pattern_types or 'decelerating' in pattern_types
        assert 'cyclic' in pattern_types
        assert 'anomaly' in pattern_types
        
    @pytest.mark.asyncio
    async def test_concurrent_performance(self, sample_data_generator, tool_orchestrator):
        """Test performance of concurrent vs sequential analysis."""
        data, _ = sample_data_generator(length=1000)
        
        # Sequential execution
        strategies = [TrendAnalysis(), CyclicAnalysis(), AnomalyDetection()]
        start_time = time.perf_counter()
        sequential_patterns = []
        for strategy in strategies:
            patterns = await strategy.analyze(data)
            sequential_patterns.extend(patterns)
        sequential_time = time.perf_counter() - start_time
        
        # Concurrent execution
        compound = CompoundAnalysis()
        compound.set_tool_orchestrator(tool_orchestrator)
        start_time = time.perf_counter()
        concurrent_patterns = await compound.analyze(data)
        concurrent_time = time.perf_counter() - start_time
        
        # Concurrent should be faster
        assert concurrent_time < sequential_time
        # But should find similar patterns
        assert len(concurrent_patterns) >= len(sequential_patterns) * 0.8
        
    @pytest.mark.asyncio
    async def test_resource_usage(self, sample_data_generator):
        """Test resource usage with large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Analyze large dataset
        data, _ = sample_data_generator(length=10000)
        strategy = CompoundAnalysis()
        patterns = await strategy.analyze(data)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory usage should be reasonable
        assert memory_increase < 100  # Less than 100MB increase
        assert len(patterns) > 0