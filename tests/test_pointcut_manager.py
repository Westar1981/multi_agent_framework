"""Unit tests for the pointcut manager."""

import pytest
from multi_agent_framework.core.pointcuts.manager import PointcutManager, PointcutType, Pointcut

def test_pointcut_creation():
    """Test creating a pointcut."""
    manager = PointcutManager()
    pattern = r"test_.*"
    pointcut_type = PointcutType.METHOD_EXECUTION
    
    pointcut_id = manager.add_pointcut(pattern, pointcut_type)
    assert pointcut_id.startswith("execution_")
    assert pointcut_id in manager.pointcuts
    
    pointcut = manager.get_pointcut(pointcut_id)
    assert pointcut.pattern == pattern
    assert pointcut.pointcut_type == pointcut_type
    assert pointcut.enabled is True
    assert isinstance(pointcut.metadata, dict)

def test_invalid_pattern():
    """Test creating a pointcut with invalid pattern."""
    manager = PointcutManager()
    with pytest.raises(ValueError):
        manager.add_pointcut("[invalid", PointcutType.METHOD_EXECUTION)

def test_pointcut_removal():
    """Test removing a pointcut."""
    manager = PointcutManager()
    pointcut_id = manager.add_pointcut(r"test_.*", PointcutType.METHOD_EXECUTION)
    
    manager.remove_pointcut(pointcut_id)
    assert pointcut_id not in manager.pointcuts
    
    with pytest.raises(KeyError):
        manager.get_pointcut(pointcut_id)

def test_pointcut_enable_disable():
    """Test enabling and disabling pointcuts."""
    manager = PointcutManager()
    pointcut_id = manager.add_pointcut(r"test_.*", PointcutType.METHOD_EXECUTION)
    
    manager.disable_pointcut(pointcut_id)
    assert not manager.get_pointcut(pointcut_id).enabled
    
    manager.enable_pointcut(pointcut_id)
    assert manager.get_pointcut(pointcut_id).enabled

def test_pattern_matching():
    """Test pattern matching functionality."""
    manager = PointcutManager()
    manager.add_pointcut(r"test_.*", PointcutType.METHOD_EXECUTION)
    manager.add_pointcut(r"prod_.*", PointcutType.METHOD_CALL)
    
    matches = manager.check_matches("test_function")
    assert len(matches) == 1
    assert matches[0].target_name == "test_function"
    
    matches = manager.check_matches("prod_function")
    assert len(matches) == 1
    assert matches[0].target_name == "prod_function"
    
    matches = manager.check_matches("other_function")
    assert len(matches) == 0

def test_match_caching():
    """Test match result caching."""
    manager = PointcutManager()
    pointcut_id = manager.add_pointcut(r"test_.*", PointcutType.METHOD_EXECUTION)
    
    # First match should cache result
    matches1 = manager.check_matches("test_function")
    matches2 = manager.check_matches("test_function")
    assert matches1 == matches2
    
    # Modifying pointcut should invalidate cache
    manager.disable_pointcut(pointcut_id)
    matches3 = manager.check_matches("test_function")
    assert len(matches3) == 0

def test_match_context():
    """Test matching with context information."""
    manager = PointcutManager()
    manager.add_pointcut(r"test_.*", PointcutType.METHOD_EXECUTION)
    
    context = {"module": "test_module", "class": "TestClass"}
    matches = manager.check_matches("test_function", context)
    
    assert len(matches) == 1
    assert matches[0].context == context

def test_metadata():
    """Test pointcut metadata handling."""
    manager = PointcutManager()
    metadata = {"description": "Test pointcut", "priority": "high"}
    pointcut_id = manager.add_pointcut(
        r"test_.*",
        PointcutType.METHOD_EXECUTION,
        metadata=metadata
    )
    
    pointcut = manager.get_pointcut(pointcut_id)
    assert pointcut.metadata == metadata

def test_active_matches():
    """Test active matches tracking."""
    manager = PointcutManager()
    manager.add_pointcut(r"test_.*", PointcutType.METHOD_EXECUTION)
    
    # Check matches should add to active matches
    manager.check_matches("test_function")
    active = manager.get_active_matches()
    assert len(active) == 1
    
    # Clear should remove all active matches
    manager.clear_matches()
    assert len(manager.get_active_matches()) == 0

def test_multiple_matches():
    """Test matching multiple patterns."""
    manager = PointcutManager()
    manager.add_pointcut(r"test_.*", PointcutType.METHOD_EXECUTION)
    manager.add_pointcut(r".*_test", PointcutType.METHOD_EXECUTION)
    
    matches = manager.check_matches("test_function_test")
    assert len(matches) == 2 