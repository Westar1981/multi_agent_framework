"""Distributed cache manager using Redis."""

import json
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
try:
    import aioredis
except ImportError:
    aioredis = None
from .cache_manager import CacheConfig, CacheStats

logger = logging.getLogger(__name__)

@dataclass
class DistributedCacheConfig:
    """Configuration for distributed cache."""
    redis_url: str = "redis://localhost"
    namespace: str = "multi_agent"
    replication_factor: int = 2
    sync_interval: int = 60
    local_cache_size: int = 1000
    preload_keys: Set[str] = None
    persistence_enabled: bool = True
    backup_interval: int = 3600  # 1 hour

class DistributedCache:
    """Distributed cache implementation using Redis."""
    
    def __init__(self, config: DistributedCacheConfig):
        self.config = config
        self.redis: Optional[aioredis.Redis] = None
        self.local_cache: Dict[str, Any] = {}
        self.preload_queue: asyncio.Queue = asyncio.Queue()
        self._setup_tasks()
        
    async def _setup_tasks(self) -> None:
        """Setup background tasks."""
        # Connect to Redis
        self.redis = await aioredis.create_redis_pool(self.config.redis_url)
        
        # Start background tasks
        asyncio.create_task(self._sync_task())
        asyncio.create_task(self._backup_task())
        asyncio.create_task(self._preload_task())
        
    async def _sync_task(self) -> None:
        """Synchronize local cache with Redis."""
        while True:
            try:
                # Get all keys in namespace
                pattern = f"{self.config.namespace}:*"
                keys = await self.redis.keys(pattern)
                
                # Update local cache
                for key in keys:
                    if key not in self.local_cache:
                        value = await self.redis.get(key)
                        if value:
                            self.local_cache[key] = json.loads(value)
                            
                # Remove expired local entries
                local_keys = set(self.local_cache.keys())
                redis_keys = set(keys)
                expired = local_keys - redis_keys
                for key in expired:
                    del self.local_cache[key]
                    
            except Exception as e:
                logger.error(f"Error in sync task: {str(e)}")
                
            await asyncio.sleep(self.config.sync_interval)
            
    async def _backup_task(self) -> None:
        """Backup cache to persistent storage."""
        if not self.config.persistence_enabled:
            return
            
        while True:
            try:
                # Save current cache state
                backup = {
                    "local_cache": self.local_cache,
                    "timestamp": time.time()
                }
                
                backup_key = f"{self.config.namespace}:backup"
                await self.redis.set(
                    backup_key,
                    json.dumps(backup)
                )
                
                logger.info("Cache backup completed")
                
            except Exception as e:
                logger.error(f"Error in backup task: {str(e)}")
                
            await asyncio.sleep(self.config.backup_interval)
            
    async def _preload_task(self) -> None:
        """Preload frequently accessed keys."""
        while True:
            try:
                key = await self.preload_queue.get()
                
                # Check if already in local cache
                if key in self.local_cache:
                    continue
                    
                # Get from Redis
                value = await self.redis.get(key)
                if value:
                    self.local_cache[key] = json.loads(value)
                    
            except Exception as e:
                logger.error(f"Error in preload task: {str(e)}")
                
            finally:
                self.preload_queue.task_done()
                
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Try local cache first
        if key in self.local_cache:
            return self.local_cache[key]
            
        # Try Redis
        redis_key = f"{self.config.namespace}:{key}"
        value = await self.redis.get(redis_key)
        
        if value:
            # Update local cache
            parsed = json.loads(value)
            self.local_cache[key] = parsed
            return parsed
            
        return None
        
    async def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        # Update Redis
        redis_key = f"{self.config.namespace}:{key}"
        await self.redis.set(
            redis_key,
            json.dumps(value)
        )
        
        # Update local cache
        self.local_cache[key] = value
        
        # Replicate if needed
        if self.config.replication_factor > 1:
            await self._replicate(key, value)
            
    async def _replicate(self, key: str, value: Any) -> None:
        """Replicate value to backup nodes."""
        for i in range(1, self.config.replication_factor):
            backup_key = f"{self.config.namespace}:backup{i}:{key}"
            await self.redis.set(
                backup_key,
                json.dumps(value)
            )
            
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        # Remove from Redis
        redis_key = f"{self.config.namespace}:{key}"
        await self.redis.delete(redis_key)
        
        # Remove from local cache
        if key in self.local_cache:
            del self.local_cache[key]
            
        # Remove replicas
        for i in range(1, self.config.replication_factor):
            backup_key = f"{self.config.namespace}:backup{i}:{key}"
            await self.redis.delete(backup_key)
            
    async def clear(self) -> None:
        """Clear all cache entries."""
        # Clear Redis namespace
        pattern = f"{self.config.namespace}:*"
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)
            
        # Clear local cache
        self.local_cache.clear()
        
    def schedule_preload(self, key: str) -> None:
        """Schedule key for preloading."""
        self.preload_queue.put_nowait(key)
        
    async def restore_backup(self) -> None:
        """Restore cache from backup."""
        if not self.config.persistence_enabled:
            return
            
        try:
            backup_key = f"{self.config.namespace}:backup"
            backup_data = await self.redis.get(backup_key)
            
            if backup_data:
                backup = json.loads(backup_data)
                self.local_cache = backup["local_cache"]
                logger.info("Cache restored from backup")
                
        except Exception as e:
            logger.error(f"Error restoring cache: {str(e)}")
            
    async def shutdown(self) -> None:
        """Shutdown cache manager."""
        # Save backup
        if self.config.persistence_enabled:
            await self._backup_task()
            
        # Close Redis connection
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
            
        # Clear local cache
        self.local_cache.clear() 