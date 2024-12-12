"""Cache preloader for predictive caching."""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PreloaderConfig:
    """Configuration for cache preloader."""
    window_size: int = 100  # Size of sliding window for pattern detection
    min_confidence: float = 0.7  # Minimum confidence for prediction
    max_predictions: int = 10  # Maximum predictions per key
    update_interval: int = 60  # Seconds between pattern updates
    max_patterns: int = 1000  # Maximum number of patterns to track
    cleanup_threshold: int = 10000  # Access count before cleanup

class AccessPattern:
    """Pattern of cache access sequences."""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.sequences: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_occurrences: Dict[tuple, int] = defaultdict(int)
        
    def add_sequence(self, sequence: List[str]) -> None:
        """Add an access sequence."""
        if len(sequence) < 2:
            return
            
        # Use sliding window
        for i in range(len(sequence) - 1):
            window = tuple(sequence[max(0, i - self.window_size + 1):i + 1])
            next_key = sequence[i + 1]
            
            self.sequences[window][next_key] += 1
            self.total_occurrences[window] += 1
            
    def predict_next(self, sequence: List[str], min_confidence: float) -> List[tuple]:
        """Predict next likely keys."""
        if len(sequence) < 1:
            return []
            
        # Get recent window
        window = tuple(sequence[-self.window_size:])
        
        # Get predictions with confidence
        if window in self.sequences:
            total = self.total_occurrences[window]
            predictions = [
                (key, count / total)
                for key, count in self.sequences[window].items()
                if count / total >= min_confidence
            ]
            return sorted(predictions, key=lambda x: x[1], reverse=True)
            
        return []

class CachePreloader:
    """Predictive cache preloader."""
    
    def __init__(self, config: PreloaderConfig):
        self.config = config
        self.pattern = AccessPattern(config.window_size)
        self.access_history: Dict[str, List[float]] = defaultdict(list)
        self.access_sequences: List[str] = []
        self.preload_callbacks: List[Callable[[str], None]] = []
        self._setup_tasks()
        
    def _setup_tasks(self) -> None:
        """Setup background tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._update_task = asyncio.create_task(self._update_loop())
        
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old patterns."""
        while True:
            try:
                total_accesses = sum(
                    len(times) for times in self.access_history.values()
                )
                
                if total_accesses > self.config.cleanup_threshold:
                    # Remove old access history
                    current_time = time.time()
                    for key in list(self.access_history.keys()):
                        self.access_history[key] = [
                            t for t in self.access_history[key]
                            if current_time - t < 3600  # Keep last hour
                        ]
                        if not self.access_history[key]:
                            del self.access_history[key]
                            
                    # Limit pattern storage
                    if len(self.pattern.sequences) > self.config.max_patterns:
                        # Keep most frequent patterns
                        patterns = sorted(
                            self.pattern.sequences.items(),
                            key=lambda x: sum(x[1].values()),
                            reverse=True
                        )
                        self.pattern.sequences = dict(patterns[:self.config.max_patterns])
                        
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                
            await asyncio.sleep(300)  # Run every 5 minutes
            
    async def _update_loop(self) -> None:
        """Periodic pattern update and prediction."""
        while True:
            try:
                # Update patterns
                self.pattern.add_sequence(self.access_sequences[-self.config.window_size:])
                
                # Make predictions
                predictions = self.pattern.predict_next(
                    self.access_sequences[-self.config.window_size:],
                    self.config.min_confidence
                )
                
                # Trigger preloading
                for key, confidence in predictions[:self.config.max_predictions]:
                    for callback in self.preload_callbacks:
                        try:
                            callback(key)
                        except Exception as e:
                            logger.error(f"Error in preload callback: {str(e)}")
                            
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                
            await asyncio.sleep(self.config.update_interval)
            
    def record_access(self, key: str) -> None:
        """Record cache access."""
        current_time = time.time()
        self.access_history[key].append(current_time)
        self.access_sequences.append(key)
        
        # Limit sequence length
        if len(self.access_sequences) > self.config.window_size * 2:
            self.access_sequences = self.access_sequences[-self.config.window_size:]
            
    def add_preload_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for preloading predictions."""
        self.preload_callbacks.append(callback)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get preloader statistics."""
        stats = {
            'total_patterns': len(self.pattern.sequences),
            'total_keys': len(self.access_history),
            'total_accesses': sum(len(times) for times in self.access_history.values()),
            'sequence_length': len(self.access_sequences)
        }
        
        # Calculate access frequencies
        frequencies = {}
        current_time = time.time()
        for key, times in self.access_history.items():
            recent_times = [t for t in times if current_time - t < 3600]
            if recent_times:
                frequencies[key] = len(recent_times) / 3600  # Access per second
                
        if frequencies:
            stats.update({
                'max_frequency': max(frequencies.values()),
                'avg_frequency': np.mean(list(frequencies.values())),
                'min_frequency': min(frequencies.values())
            })
            
        return stats
        
    def get_predictions(self, sequence: Optional[List[str]] = None) -> List[tuple]:
        """Get predictions for a sequence."""
        if sequence is None:
            sequence = self.access_sequences
            
        return self.pattern.predict_next(
            sequence,
            self.config.min_confidence
        )
        
    async def shutdown(self) -> None:
        """Shutdown preloader."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            
        if self._update_task:
            self._update_task.cancel()
            
        # Clear data
        self.pattern.sequences.clear()
        self.access_history.clear()
        self.access_sequences.clear()
        self.preload_callbacks.clear() 