"""Tests for cache persistence."""

import os
import json
import time
import pytest
import sqlite3
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

from ..core.cache_persistence import (
    CachePersistence,
    PersistenceConfig,
    BatchPersistence
)

@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_cache.db")

@pytest.fixture
def persistence(temp_db_path):
    """Create a persistence manager."""
    config = PersistenceConfig(
        db_path=temp_db_path,
        auto_persist=False,
        compression_enabled=True
    )
    manager = CachePersistence(config)
    yield manager
    
    # Cleanup
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)

def test_save_and_load(persistence):
    """Test saving and loading values."""
    # Save value
    persistence.save("key1", "value1")
    
    # Load value
    value = persistence.load("key1")
    assert value == "value1"
    
    # Check access count
    with sqlite3.connect(persistence.db_path) as conn:
        count = conn.execute("""
            SELECT access_count 
            FROM cache_entries 
            WHERE key = ?
        """, ("key1",)).fetchone()[0]
        assert count == 1
        
def test_compression(persistence):
    """Test data compression."""
    # Large value to test compression
    large_value = "x" * 1000
    
    # Save with compression
    persistence.save("key1", large_value)
    
    # Get compressed size
    with sqlite3.connect(persistence.db_path) as conn:
        compressed = conn.execute("""
            SELECT value 
            FROM cache_entries 
            WHERE key = ?
        """, ("key1",)).fetchone()[0]
        
    # Should be smaller than original
    assert len(compressed) < len(large_value)
    
    # Should load correctly
    assert persistence.load("key1") == large_value
    
def test_cleanup(persistence):
    """Test old entry cleanup."""
    # Add old entry
    with sqlite3.connect(persistence.db_path) as conn:
        old_time = int(time.time()) - (31 * 86400)  # 31 days old
        conn.execute("""
            INSERT INTO cache_entries 
            (key, value, created_at, accessed_at)
            VALUES (?, ?, ?, ?)
        """, ("old_key", b"old_value", old_time, old_time))
        
    # Add new entry
    persistence.save("new_key", "new_value")
    
    # Run cleanup
    persistence.cleanup()
    
    # Check entries
    with sqlite3.connect(persistence.db_path) as conn:
        entries = conn.execute("SELECT key FROM cache_entries").fetchall()
        keys = [row[0] for row in entries]
        
    assert "old_key" not in keys
    assert "new_key" in keys
    
def test_optimize(persistence):
    """Test database optimization."""
    # Add many entries
    for i in range(persistence.config.vacuum_threshold + 1):
        persistence.save(f"key{i}", f"value{i}")
        
    # Get initial size
    initial_size = os.path.getsize(persistence.db_path)
    
    # Delete most entries
    with sqlite3.connect(persistence.db_path) as conn:
        conn.execute("DELETE FROM cache_entries WHERE key != 'key0'")
        
    # Run optimization
    persistence.optimize()
    
    # Size should be reduced
    final_size = os.path.getsize(persistence.db_path)
    assert final_size < initial_size
    
def test_batch_operations(persistence):
    """Test batch operations."""
    with BatchPersistence(persistence) as batch:
        for i in range(10):
            batch.add(f"key{i}", f"value{i}")
            
    # Check all values were saved
    for i in range(10):
        assert persistence.load(f"key{i}") == f"value{i}"
        
def test_concurrent_access(persistence):
    """Test concurrent access to persistence."""
    def worker(start, end):
        for i in range(start, end):
            persistence.save(f"key{i}", f"value{i}")
            assert persistence.load(f"key{i}") == f"value{i}"
            
    threads = [
        threading.Thread(target=worker, args=(i*10, (i+1)*10))
        for i in range(10)
    ]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    # Verify all values
    for i in range(100):
        assert persistence.load(f"key{i}") == f"value{i}"
        
def test_stats(persistence):
    """Test statistics collection."""
    # Add some entries
    for i in range(10):
        persistence.save(f"key{i}", f"value{i}")
        persistence.load(f"key{i}")
        
    stats = persistence.get_stats()
    
    assert stats['total_entries'] == 10
    assert stats['size_bytes'] > 0
    assert stats['avg_access_count'] == 1
    assert 0 <= stats['avg_age_seconds'] <= 1
    
def test_error_handling(persistence):
    """Test error handling."""
    # Invalid JSON
    with patch('json.dumps', side_effect=Exception("JSON error")):
        persistence.save("key1", "value1")
        assert persistence.load("key1") is None
        
    # Database error
    with patch('sqlite3.connect', side_effect=Exception("DB error")):
        persistence.save("key2", "value2")
        assert persistence.load("key2") is None
        
def test_compression_disabled(temp_db_path):
    """Test operation with compression disabled."""
    config = PersistenceConfig(
        db_path=temp_db_path,
        compression_enabled=False
    )
    persistence = CachePersistence(config)
    
    # Save and load without compression
    persistence.save("key1", "value1")
    assert persistence.load("key1") == "value1"
    
    # Check raw value
    with sqlite3.connect(temp_db_path) as conn:
        value = conn.execute("""
            SELECT value 
            FROM cache_entries 
            WHERE key = ?
        """, ("key1",)).fetchone()[0]
        
    # Should be plain JSON
    assert json.loads(value) == "value1"
    
def test_auto_persist(temp_db_path):
    """Test automatic persistence."""
    config = PersistenceConfig(
        db_path=temp_db_path,
        auto_persist=True,
        persist_interval=0.1
    )
    persistence = CachePersistence(config)
    
    # Add old entry
    with sqlite3.connect(temp_db_path) as conn:
        old_time = int(time.time()) - (31 * 86400)
        conn.execute("""
            INSERT INTO cache_entries 
            (key, value, created_at, accessed_at)
            VALUES (?, ?, ?, ?)
        """, ("old_key", b"old_value", old_time, old_time))
        
    # Wait for auto cleanup
    time.sleep(0.2)
    
    # Old entry should be removed
    assert persistence.load("old_key") is None 