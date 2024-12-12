"""Cache persistence manager."""

import os
import json
import time
import logging
import sqlite3
import threading
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PersistenceConfig:
    """Configuration for cache persistence."""
    db_path: str = "cache.db"
    auto_persist: bool = True
    persist_interval: int = 300  # 5 minutes
    max_batch_size: int = 1000
    compression_enabled: bool = True
    vacuum_threshold: int = 10000  # Entries before vacuum
    max_age_days: int = 30

class CachePersistence:
    """Manages cache persistence to disk."""
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.db_path = Path(config.db_path)
        self._setup_db()
        self._setup_tasks()
        
    def _setup_db(self) -> None:
        """Setup SQLite database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at INTEGER,
                    accessed_at INTEGER,
                    access_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_accessed_at 
                ON cache_entries(accessed_at)
            """)
            
    def _setup_tasks(self) -> None:
        """Setup background tasks."""
        if self.config.auto_persist:
            self._persist_thread = threading.Thread(
                target=self._persist_task,
                daemon=True
            )
            self._persist_thread.start()
            
    def _persist_task(self) -> None:
        """Periodic persistence task."""
        while True:
            try:
                # Clean old entries
                self.cleanup()
                
                # Optimize if needed
                self.optimize()
                
            except Exception as e:
                logger.error(f"Error in persist task: {str(e)}")
                
            time.sleep(self.config.persist_interval)
            
    def save(self, key: str, value: Any) -> None:
        """Save value to persistent storage."""
        try:
            serialized = json.dumps(value)
            if self.config.compression_enabled:
                serialized = self._compress(serialized)
                
            with sqlite3.connect(self.db_path) as conn:
                now = int(time.time())
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, value, created_at, accessed_at, access_count)
                    VALUES (?, ?, ?, ?, 0)
                """, (key, serialized, now, now))
                
        except Exception as e:
            logger.error(f"Error saving cache entry: {str(e)}")
            
    def load(self, key: str) -> Optional[Any]:
        """Load value from persistent storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = int(time.time())
                cursor = conn.execute("""
                    UPDATE cache_entries 
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE key = ?
                    RETURNING value
                """, (now, key))
                
                row = cursor.fetchone()
                if row:
                    value = row[0]
                    if self.config.compression_enabled:
                        value = self._decompress(value)
                    return json.loads(value)
                    
        except Exception as e:
            logger.error(f"Error loading cache entry: {str(e)}")
            
        return None
        
    def delete(self, key: str) -> None:
        """Delete value from persistent storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE key = ?
                """, (key,))
                
        except Exception as e:
            logger.error(f"Error deleting cache entry: {str(e)}")
            
    def cleanup(self) -> None:
        """Clean up old entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                max_age = time.time() - (self.config.max_age_days * 86400)
                conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE accessed_at < ?
                """, (int(max_age),))
                
        except Exception as e:
            logger.error(f"Error cleaning cache: {str(e)}")
            
    def optimize(self) -> None:
        """Optimize database if needed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                count = conn.execute("""
                    SELECT COUNT(*) FROM cache_entries
                """).fetchone()[0]
                
                if count > self.config.vacuum_threshold:
                    conn.execute("VACUUM")
                    logger.info("Database optimized")
                    
        except Exception as e:
            logger.error(f"Error optimizing database: {str(e)}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Total entries
                stats['total_entries'] = conn.execute("""
                    SELECT COUNT(*) FROM cache_entries
                """).fetchone()[0]
                
                # Size on disk
                stats['size_bytes'] = os.path.getsize(self.db_path)
                
                # Age statistics
                now = time.time()
                cursor = conn.execute("""
                    SELECT 
                        AVG(?) - created_at,
                        AVG(?) - accessed_at,
                        AVG(access_count)
                    FROM cache_entries
                """, (now, now))
                
                avg_age, avg_access_age, avg_access_count = cursor.fetchone()
                stats.update({
                    'avg_age_seconds': avg_age,
                    'avg_access_age_seconds': avg_access_age,
                    'avg_access_count': avg_access_count
                })
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
            
    def _compress(self, data: str) -> bytes:
        """Compress string data."""
        import zlib
        return zlib.compress(data.encode())
        
    def _decompress(self, data: bytes) -> str:
        """Decompress bytes data."""
        import zlib
        return zlib.decompress(data).decode()
        
    def shutdown(self) -> None:
        """Shutdown persistence manager."""
        try:
            # Optimize before shutdown
            self.optimize()
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            
class BatchPersistence:
    """Batch operations for cache persistence."""
    
    def __init__(self, persistence: CachePersistence):
        self.persistence = persistence
        self.batch: List[tuple] = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        
    def add(self, key: str, value: Any) -> None:
        """Add item to batch."""
        self.batch.append((key, value))
        
        if len(self.batch) >= self.persistence.config.max_batch_size:
            self.flush()
            
    def flush(self) -> None:
        """Flush batch to persistent storage."""
        if not self.batch:
            return
            
        try:
            with sqlite3.connect(self.persistence.db_path) as conn:
                now = int(time.time())
                conn.executemany("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, value, created_at, accessed_at, access_count)
                    VALUES (?, ?, ?, ?, 0)
                """, [
                    (
                        key,
                        self.persistence._compress(json.dumps(value))
                        if self.persistence.config.compression_enabled
                        else json.dumps(value),
                        now,
                        now
                    )
                    for key, value in self.batch
                ])
                
            self.batch.clear()
            
        except Exception as e:
            logger.error(f"Error flushing batch: {str(e)}")
            self.batch.clear() 