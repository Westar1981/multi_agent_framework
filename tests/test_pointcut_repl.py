"""Tests for pointcut REPL interface."""

import io
import pytest
from unittest.mock import patch, MagicMock

from ..repl.pointcut_repl import PointcutREPL
from ..core.pointcut_manager import PointcutType

@pytest.fixture
def repl():
    """Create test REPL instance."""
    return PointcutREPL()

def test_add_command(repl):
    """Test add command."""
    # Test basic add
    repl.do_add("METHOD_EXECUTION get_.*")
    assert len(repl.manager.pointcuts) == 1
    
    # Test add with metadata
    repl.do_add("METHOD_CALL set_.* scope=public type=method")
    assert len(repl.manager.pointcuts) == 2
    
    # Test invalid type
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        repl.do_add("INVALID get_.*")
        assert "Invalid pointcut type" in fake_out.getvalue()
        
    # Test missing arguments
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        repl.do_add("")
        assert "Error: Must provide pointcut type and pattern" in fake_out.getvalue()
        
def test_remove_command(repl):
    """Test remove command."""
    # Add and remove pointcut
    pointcut_id = repl.manager.add_pointcut(
        pattern="test_.*",
        pointcut_type=PointcutType.METHOD_EXECUTION
    )
    
    repl.do_remove(pointcut_id)
    assert len(repl.manager.pointcuts) == 0
    
    # Test missing argument
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        repl.do_remove("")
        assert "Error: Must provide pointcut ID" in fake_out.getvalue()
        
def test_list_command(repl):
    """Test list command."""
    # Add test pointcuts
    repl.manager.add_pointcut(
        pattern="test1_.*",
        pointcut_type=PointcutType.METHOD_EXECUTION
    )
    repl.manager.add_pointcut(
        pattern="test2_.*",
        pointcut_type=PointcutType.METHOD_CALL
    )
    
    # Test list all
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        repl.do_list("")
        output = fake_out.getvalue()
        assert "test1_.*" in output
        assert "test2_.*" in output
        
    # Test list enabled
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        repl.do_list("enabled")
        output = fake_out.getvalue()
        assert "test1_.*" in output
        assert "test2_.*" in output
        
def test_check_command(repl):
    """Test check command."""
    # Add test pointcut
    repl.manager.add_pointcut(
        pattern="test_.*",
        pointcut_type=PointcutType.METHOD_EXECUTION
    )
    
    # Test matching
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        repl.do_check("test_method")
        output = fake_out.getvalue()
        assert "test_.*" in output
        
    # Test non-matching
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        repl.do_check("other_method")
        assert "No matches found" in fake_out.getvalue()
        
    # Test missing argument
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        repl.do_check("")
        assert "Error: Must provide target name" in fake_out.getvalue()
        
def test_enable_disable_commands(repl):
    """Test enable and disable commands."""
    # Add test pointcut
    pointcut_id = repl.manager.add_pointcut(
        pattern="test_.*",
        pointcut_type=PointcutType.METHOD_EXECUTION
    )
    
    # Test disable
    repl.do_disable(pointcut_id)
    assert not repl.manager.pointcuts[pointcut_id].enabled
    
    # Test enable
    repl.do_enable(pointcut_id)
    assert repl.manager.pointcuts[pointcut_id].enabled
    
    # Test missing arguments
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        repl.do_enable("")
        assert "Error: Must provide pointcut ID" in fake_out.getvalue()
        
    with patch('sys.stdout', new=io.StringIO()) as fake_out:
        repl.do_disable("")
        assert "Error: Must provide pointcut ID" in fake_out.getvalue()
        
def test_clear_command(repl):
    """Test clear command."""
    # Add test pointcut and create match
    repl.manager.add_pointcut(
        pattern="test_.*",
        pointcut_type=PointcutType.METHOD_EXECUTION
    )
    repl.manager.check_matches("test_method")
    assert len(repl.manager.get_active_matches()) > 0
    
    # Test clear
    repl.do_clear("")
    assert len(repl.manager.get_active_matches()) == 0
    
def test_quit_command(repl):
    """Test quit command."""
    assert repl.do_quit("") is True
    
def test_EOF_command(repl):
    """Test EOF handling."""
    assert repl.do_EOF("") is True 