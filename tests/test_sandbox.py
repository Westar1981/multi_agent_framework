"""Tests for code sandbox functionality."""

import pytest
import asyncio
import time
from ..repl.code_sandbox import CodeSandbox, SandboxConfig

@pytest.mark.asyncio
async def test_basic_code_execution():
    sandbox = CodeSandbox()
    result = await sandbox.run_code("x = 1 + 1; print(x)")
    
    assert result['success']
    assert result['output'].strip() == "2"
    assert result['locals']['x'] == 2
    
@pytest.mark.asyncio
async def test_memory_limit():
    config = SandboxConfig(max_memory_mb=1)  # 1MB limit
    sandbox = CodeSandbox(config)
    
    # Try to allocate large list
    result = await sandbox.run_code("""
    x = [0] * (1024 * 1024 * 10)  # Try to allocate 10MB
    """)
    
    assert not result['success']
    assert 'MemoryError' in str(result['error'])
    
@pytest.mark.asyncio
async def test_time_limit():
    config = SandboxConfig(max_time_seconds=1)
    sandbox = CodeSandbox(config)
    
    result = await sandbox.run_code("""
    import time
    time.sleep(2)
    """)
    
    assert not result['success']
    assert 'TimeoutError' in str(result['error'])
    
@pytest.mark.asyncio
async def test_blocked_imports():
    sandbox = CodeSandbox()
    
    result = await sandbox.run_code("""
    import os
    os.system('echo "hack"')
    """)
    
    assert not result['success']
    assert "Import of module 'os' not allowed" in str(result['error'])
    
@pytest.mark.asyncio
async def test_blocked_builtins():
    sandbox = CodeSandbox()
    
    result = await sandbox.run_code("""
    eval('print("hack")')
    """)
    
    assert not result['success']
    assert "Call to 'eval' not allowed" in str(result['error'])
    
@pytest.mark.asyncio
async def test_allowed_imports():
    sandbox = CodeSandbox()
    
    result = await sandbox.run_code("""
    import math
    print(math.pi)
    """)
    
    assert result['success']
    assert '3.14' in result['output']
    
@pytest.mark.asyncio
async def test_context_variables():
    sandbox = CodeSandbox()
    context = {'x': 42, 'name': 'test'}
    
    result = await sandbox.run_code("""
    print(f'x = {x}, name = {name}')
    y = x * 2
    """, context=context)
    
    assert result['success']
    assert 'x = 42, name = test' in result['output']
    assert result['locals']['y'] == 84
    
@pytest.mark.asyncio
async def test_syntax_error():
    sandbox = CodeSandbox()
    
    result = await sandbox.run_code("""
    if True
        print("invalid syntax")
    """)
    
    assert not result['success']
    assert 'SyntaxError' in result['error']['type']
    
@pytest.mark.asyncio
async def test_runtime_error():
    sandbox = CodeSandbox()
    
    result = await sandbox.run_code("""
    1/0
    """)
    
    assert not result['success']
    assert 'ZeroDivisionError' in result['error']['type']
    
@pytest.mark.asyncio
async def test_cleanup():
    sandbox = CodeSandbox()
    original_builtins = dict(globals()['__builtins__'].__dict__)
    
    # Run some code that could modify builtins
    await sandbox.run_code("print('test')")
    
    # Cleanup
    sandbox.cleanup()
    
    # Check builtins are restored
    current_builtins = dict(globals()['__builtins__'].__dict__)
    assert set(original_builtins.keys()) == set(current_builtins.keys())
    
@pytest.mark.asyncio
async def test_custom_config():
    config = SandboxConfig(
        max_memory_mb=256,
        max_time_seconds=2,
        allowed_modules={'datetime'},
        blocked_attributes={'eval', 'exec'}
    )
    sandbox = CodeSandbox(config)
    
    # Test allowed module
    result = await sandbox.run_code("""
    from datetime import datetime
    print(datetime.now().year)
    """)
    
    assert result['success']
    assert str(time.localtime().tm_year) in result['output']
    
    # Test blocked module
    result = await sandbox.run_code("""
    import math
    """)
    
    assert not result['success']
    assert "Import of module 'math' not allowed" in str(result['error'])
    
@pytest.mark.asyncio
async def test_async_code():
    sandbox = CodeSandbox()
    
    result = await sandbox.run_code("""
    async def foo():
        return 42
        
    result = await foo()
    print(result)
    """)
    
    assert result['success']
    assert '42' in result['output'] 