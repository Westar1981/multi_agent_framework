"""Unified cache manager integrating distributed caching, persistence and preloading."""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass

from .distributed_cache import DistributedCache, DistributedCacheConfig
from .cache_persistence import CachePersistence, PersistenceConfig
from .cache_preloader import CachePreloader, PreloaderConfig

logger = logging.getLogger(__name__)

@dataclass
class UnifiedCacheConfig:
    """Configuration for unified cache."""
    distributed: DistributedCacheConfig
    persistence: PersistenceConfig
    preloader: PreloaderConfig
    
    # Integration settings
    preload_batch_size: int = 100
    persistence_batch_size: int = 1000
    sync_interval: int = 300  # 5 minutes
    enable_stats: bool = True

class UnifiedCache:
    """Unified cache manager integrating distributed, persistent and predictive caching."""
    
    def __init__(self, config: UnifiedCacheConfig):
        self.config = config
        
        # Initialize components
        self.distributed = DistributedCache(config.distributed)
        self.persistence = CachePersistence(config.persistence)
        self.preloader = CachePreloader(config.preloader)
        
        # Setup integration
        self._setup_integration()
        
    def _setup_integration(self) -> None:
        """Setup component integration."""
        # Add preloader callback
        self.preloader.add_preload_callback(self._handle_preload)
        
        # Start sync task
        self._sync_task = asyncio.create_task(self._sync_loop())
        
    async def _sync_loop(self) -> None:
        """Synchronize components periodically."""
        while True:
            try:
                # Sync distributed cache to persistence
                await self._sync_to_persistence()
                
                # Update preloader patterns
                self._update_patterns()
                
            except Exception as e:
                logger.error(f"Error in sync loop: {str(e)}")
                
            await asyncio.sleep(self.config.sync_interval)
            
    async def _sync_to_persistence(self) -> None:
        """Sync distributed cache to persistent storage."""
        try:
            # Get all keys from distributed cache
            pattern = f"{self.config.distributed.namespace}:*"
            keys = await self.distributed.redis.keys(pattern)
            
            # Process in batches
            for i in range(0, len(keys), self.config.persistence_batch_size):
                batch_keys = keys[i:i + self.config.persistence_batch_size]
                
                # Get values
                values = await asyncio.gather(*[
                    self.distributed.get(key)
                    for key in batch_keys
                ])
                
                # Save to persistence
                with self.persistence.batch() as batch:
                    for key, value in zip(batch_keys, values):
                        if value is not None:
                            batch.add(key, value)
                            
        except Exception as e:
            logger.error(f"Error syncing to persistence: {str(e)}")
            
    def _update_patterns(self) -> None:
        """Update access patterns from persistence stats."""
        try:
            stats = self.persistence.get_stats()
            
            # Get most accessed keys
            if 'access_counts' in stats:
                top_keys = sorted(
                    stats['access_counts'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:self.config.preload_batch_size]
                
                # Add to preloader
                for key, _ in top_keys:
                    self.preloader.record_access(key)
                    
        except Exception as e:
            logger.error(f"Error updating patterns: {str(e)}")
            
    async def _handle_preload(self, key: str) -> None:
        """Handle preload prediction."""
        try:
            # Check if already in distributed cache
            if not await self.distributed.get(key):
                # Try to load from persistence
                value = self.persistence.load(key)
                if value is not None:
                    await self.distributed.put(key, value)
                    
        except Exception as e:
            logger.error(f"Error handling preload for {key}: {str(e)}")
            
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            # Try distributed cache first
            value = await self.distributed.get(key)
            if value is not None:
                # Record access for prediction
                self.preloader.record_access(key)
                return value
                
            # Try persistence
            value = self.persistence.load(key)
            if value is not None:
                # Add to distributed cache
                await self.distributed.put(key, value)
                # Record access
                self.preloader.record_access(key)
                return value
                
        except Exception as e:
            logger.error(f"Error getting {key}: {str(e)}")
            
        return None
        
    async def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        try:
            # Save to distributed cache
            await self.distributed.put(key, value)
            
            # Save to persistence
            self.persistence.save(key, value)
            
            # Record access
            self.preloader.record_access(key)
            
        except Exception as e:
            logger.error(f"Error putting {key}: {str(e)}")
            
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        try:
            # Delete from distributed cache
            await self.distributed.delete(key)
            
            # Delete from persistence
            self.persistence.delete(key)
            
        except Exception as e:
            logger.error(f"Error deleting {key}: {str(e)}")
            
    async def clear(self) -> None:
        """Clear all caches."""
        try:
            # Clear distributed cache
            await self.distributed.clear()
            
            # Clear persistence
            await self.persistence.clear()
            
            # Clear preloader
            await self.preloader.shutdown()
            self.preloader = CachePreloader(self.config.preloader)
            self.preloader.add_preload_callback(self._handle_preload)
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get unified cache statistics."""
        if not self.config.enable_stats:
            return {}
            
        try:
            stats = {
                'distributed': {},
                'persistence': self.persistence.get_stats(),
                'preloader': self.preloader.get_stats()
            }
            
            # Add distributed stats if available
            if hasattr(self.distributed, 'get_stats'):
                stats['distributed'] = self.distributed.get_stats()
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
            
    async def shutdown(self) -> None:
        """Shutdown cache manager."""
        try:
            # Cancel sync task
            if self._sync_task:
                self._sync_task.cancel()
                
            # Shutdown components
            await self.distributed.shutdown()
            await self.persistence.shutdown()
            await self.preloader.shutdown()
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown() 