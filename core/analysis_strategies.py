"""
Analysis strategies for pattern detection and system behavior analysis.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod
import logging

from .tool_management import ToolOrchestrator
from .self_analysis import AnalysisPattern

logger = logging.getLogger(__name__)

class AnalysisStrategy(ABC):
    """Base class for analysis strategies."""
    
    def __init__(self):
        self.tool_orchestrator: Optional[ToolOrchestrator] = None
        
    def set_tool_orchestrator(self, orchestrator: ToolOrchestrator):
        """Set tool orchestrator for strategy."""
        self.tool_orchestrator = orchestrator
        
    @abstractmethod
    async def analyze(self, data: List[float]) -> List[AnalysisPattern]:
        """Analyze data and return patterns."""
        pass

class TrendAnalysis(AnalysisStrategy):
    """Analyzes trends in time series data."""
    
    async def analyze(self, data: List[float]) -> List[AnalysisPattern]:
        if len(data) < 2:
            return []
            
        # Sophisticated trend analysis
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 2)  # Quadratic fit
        trend = np.poly1d(coeffs)
        
        # Calculate confidence based on R-squared
        y_pred = trend(x)
        r_squared = 1 - np.sum((np.array(data) - y_pred) ** 2) / np.sum((np.array(data) - np.mean(data)) ** 2)
        
        patterns = []
        if abs(coeffs[0]) > 0.1:  # Significant quadratic term
            pattern_type = 'accelerating' if coeffs[0] > 0 else 'decelerating'
            patterns.append(AnalysisPattern(
                pattern_type=pattern_type,
                confidence=r_squared,
                affected_metrics=['trend'],
                description=f"Detected {pattern_type} trend with confidence {r_squared:.2f}",
                timestamp=datetime.now()
            ))
            
        # Use tools for additional analysis if available
        if self.tool_orchestrator:
            try:
                tool_result = await self.tool_orchestrator.execute_tool(
                    'trend_analyzer',
                    'system',
                    data=data
                )
                if tool_result.success:
                    patterns.extend(self._convert_tool_results(tool_result.result))
            except Exception as e:
                logger.warning(f"Tool-based trend analysis failed: {e}")
                
        return patterns
        
    def _convert_tool_results(self, tool_results: Dict[str, Any]) -> List[AnalysisPattern]:
        """Convert tool results to analysis patterns."""
        patterns = []
        for result in tool_results.get('patterns', []):
            patterns.append(AnalysisPattern(
                pattern_type=result['type'],
                confidence=result['confidence'],
                affected_metrics=result['metrics'],
                description=result['description'],
                timestamp=datetime.now()
            ))
        return patterns

class CyclicAnalysis(AnalysisStrategy):
    """Analyzes cyclic patterns in data."""
    
    async def analyze(self, data: List[float]) -> List[AnalysisPattern]:
        if len(data) < 10:  # Need sufficient data for cycle detection
            return []
            
        # FFT for cycle detection
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        
        # Find dominant frequencies
        main_freq_idx = np.argmax(np.abs(fft[1:])) + 1
        cycle_length = 1 / freqs[main_freq_idx]
        
        # Calculate confidence based on power spectrum
        power = np.abs(fft[main_freq_idx]) / len(data)
        confidence = min(power, 1.0)
        
        patterns = []
        if confidence > 0.3:  # Significant cycle detected
            patterns.append(AnalysisPattern(
                pattern_type='cyclic',
                confidence=confidence,
                affected_metrics=['cycle'],
                description=f"Detected cycle of length {cycle_length:.1f} with confidence {confidence:.2f}",
                timestamp=datetime.now()
            ))
            
        # Use tools for advanced cycle detection
        if self.tool_orchestrator:
            try:
                tool_result = await self.tool_orchestrator.execute_tool(
                    'cycle_detector',
                    'system',
                    data=data,
                    min_confidence=0.3
                )
                if tool_result.success:
                    patterns.extend(self._convert_tool_results(tool_result.result))
            except Exception as e:
                logger.warning(f"Tool-based cycle detection failed: {e}")
                
        return patterns
        
    def _convert_tool_results(self, tool_results: Dict[str, Any]) -> List[AnalysisPattern]:
        """Convert tool results to analysis patterns."""
        patterns = []
        for cycle in tool_results.get('cycles', []):
            patterns.append(AnalysisPattern(
                pattern_type='cyclic',
                confidence=cycle['confidence'],
                affected_metrics=cycle['affected_metrics'],
                description=cycle['description'],
                timestamp=datetime.now()
            ))
        return patterns

class AnomalyDetection(AnalysisStrategy):
    """Detects anomalies in system behavior."""
    
    async def analyze(self, data: List[float]) -> List[AnalysisPattern]:
        if len(data) < 3:
            return []
            
        # Z-score based anomaly detection
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((np.array(data) - mean) / std)
        
        patterns = []
        for i, z_score in enumerate(z_scores):
            if z_score > 3:  # 3 sigma rule
                confidence = min(z_score / 10, 1.0)
                patterns.append(AnalysisPattern(
                    pattern_type='anomaly',
                    confidence=confidence,
                    affected_metrics=['anomaly'],
                    description=f"Detected anomaly at position {i} with z-score {z_score:.2f}",
                    timestamp=datetime.now()
                ))
                
        # Use tools for advanced anomaly detection
        if self.tool_orchestrator:
            try:
                tool_result = await self.tool_orchestrator.execute_tool(
                    'anomaly_detector',
                    'system',
                    data=data,
                    sensitivity=0.95
                )
                if tool_result.success:
                    patterns.extend(self._convert_tool_results(tool_result.result))
            except Exception as e:
                logger.warning(f"Tool-based anomaly detection failed: {e}")
                
        return patterns
        
    def _convert_tool_results(self, tool_results: Dict[str, Any]) -> List[AnalysisPattern]:
        """Convert tool results to analysis patterns."""
        patterns = []
        for anomaly in tool_results.get('anomalies', []):
            patterns.append(AnalysisPattern(
                pattern_type='anomaly',
                confidence=anomaly['confidence'],
                affected_metrics=anomaly['metrics'],
                description=anomaly['description'],
                timestamp=datetime.now()
            ))
        return patterns

class CompoundAnalysis(AnalysisStrategy):
    """Combines multiple analysis strategies for comprehensive pattern detection."""
    
    def __init__(self):
        super().__init__()
        self.strategies = [
            TrendAnalysis(),
            CyclicAnalysis(),
            AnomalyDetection()
        ]
        
    def set_tool_orchestrator(self, orchestrator: ToolOrchestrator):
        """Set tool orchestrator for all strategies."""
        super().set_tool_orchestrator(orchestrator)
        for strategy in self.strategies:
            strategy.set_tool_orchestrator(orchestrator)
            
    async def analyze(self, data: List[float]) -> List[AnalysisPattern]:
        """Run all analysis strategies and combine results."""
        all_patterns = []
        
        # Run strategies concurrently
        import asyncio
        pattern_futures = [
            strategy.analyze(data)
            for strategy in self.strategies
        ]
        
        # Gather results
        pattern_lists = await asyncio.gather(*pattern_futures)
        
        # Combine and deduplicate patterns
        seen_patterns = set()
        for patterns in pattern_lists:
            for pattern in patterns:
                pattern_key = (
                    pattern.pattern_type,
                    pattern.description,
                    tuple(pattern.affected_metrics)
                )
                if pattern_key not in seen_patterns:
                    seen_patterns.add(pattern_key)
                    all_patterns.append(pattern)
                    
        return all_patterns 