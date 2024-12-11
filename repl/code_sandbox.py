"""
Secure sandbox for code execution and compilation.
"""

import ast
import sys
import builtins
import inspect
from typing import Dict, Any, Optional, Set, List
import asyncio
from contextlib import contextmanager
import resource
import signal
from io import StringIO
import traceback
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SandboxConfig:
    """Configuration for code sandbox."""
    max_memory_mb: int = 512
    max_time_seconds: int = 5
    max_cpu_time_seconds: int = 3
    allowed_modules: Set[str] = None
    blocked_attributes: Set[str] = None
    
    def __post_init__(self):
        if self.allowed_modules is None:
            self.allowed_modules = {
                'asyncio', 'datetime', 'json', 'math', 'random',
                'typing', 'collections', 'itertools', 'functools'
            }
        if self.blocked_attributes is None:
            self.blocked_attributes = {
                '__import__', 'eval', 'exec', 'compile',
                'globals', 'locals', 'vars', 'dir',
                'open', 'file', 'input', 'raw_input',
                'reload', '__builtins__', '__import__',
                'system', 'popen', 'spawn', 'fork',
                'subprocess', 'os', 'sys', 'shutil'
            }

class CodeSandbox:
    """Secure sandbox for code execution."""
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._original_builtins = dict(builtins.__dict__)
        self._setup_security()
        
    def _setup_security(self):
        """Setup security measures."""
        # Restrict builtins
        restricted_builtins = {
            name: func for name, func in builtins.__dict__.items()
            if name not in self.config.blocked_attributes
        }
        builtins.__dict__.clear()
        builtins.__dict__.update(restricted_builtins)
        
    def _restore_builtins(self):
        """Restore original builtins."""
        builtins.__dict__.clear()
        builtins.__dict__.update(self._original_builtins)
        
    @contextmanager
    def _resource_limits(self):
        """Set resource limits for execution."""
        # Set memory limit
        mem_limit = self.config.max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
        
        # Set CPU time limit
        cpu_limit = self.config.max_cpu_time_seconds
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        
        # Setup alarm for total time limit
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution exceeded time limit")
            
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.config.max_time_seconds)
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
    def _validate_ast(self, code: str) -> bool:
        """Validate code AST for security."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in code: {str(e)}")
            
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, sandbox):
                self.sandbox = sandbox
                self.errors = []
                
            def visit_Import(self, node):
                for name in node.names:
                    if name.name.split('.')[0] not in self.sandbox.config.allowed_modules:
                        self.errors.append(
                            f"Import of module '{name.name}' not allowed"
                        )
                        
            def visit_ImportFrom(self, node):
                if node.module.split('.')[0] not in self.sandbox.config.allowed_modules:
                    self.errors.append(
                        f"Import from module '{node.module}' not allowed"
                    )
                    
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.sandbox.config.blocked_attributes:
                        self.errors.append(
                            f"Call to '{node.func.id}' not allowed"
                        )
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in self.sandbox.config.blocked_attributes:
                        self.errors.append(
                            f"Call to attribute '{node.func.attr}' not allowed"
                        )
                self.generic_visit(node)
                
        visitor = SecurityVisitor(self)
        visitor.visit(tree)
        
        if visitor.errors:
            raise ValueError("\n".join(visitor.errors))
            
        return True
        
    async def compile_code(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Compile code with security checks."""
        # Validate code
        self._validate_ast(code)
        
        # Create safe globals
        safe_globals = {
            '__builtins__': builtins.__dict__.copy(),
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sum': sum,
            'min': min,
            'max': max,
            'sorted': sorted,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'bool': bool,
            'int': int,
            'float': float,
            'str': str,
            'type': type,
            'isinstance': isinstance,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'Exception': Exception,
            'asyncio': asyncio
        }
        
        if context:
            safe_globals.update(context)
            
        try:
            # Compile code
            compiled = compile(code, '<string>', 'exec')
            return compiled
        except Exception as e:
            raise ValueError(f"Compilation error: {str(e)}")
            
    async def execute_code(
        self,
        compiled_code: Any,
        context: Optional[Dict[str, Any]] = None,
        capture_output: bool = True
    ) -> Dict[str, Any]:
        """Execute compiled code in sandbox."""
        # Setup execution environment
        local_vars = {}
        if context:
            local_vars.update(context)
            
        # Capture output
        stdout = StringIO() if capture_output else sys.stdout
        stderr = StringIO() if capture_output else sys.stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout
        sys.stderr = stderr
        
        result = {
            'success': False,
            'output': '',
            'error': None,
            'locals': {},
            'return_value': None
        }
        
        try:
            with self._resource_limits():
                # Execute code
                exec(compiled_code, local_vars)
                
                # Collect results
                result.update({
                    'success': True,
                    'output': stdout.getvalue() if capture_output else '',
                    'locals': {
                        k: v for k, v in local_vars.items()
                        if not k.startswith('__')
                    }
                })
                
        except Exception as e:
            result.update({
                'success': False,
                'error': {
                    'type': type(e).__name__,
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }
            })
            
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            if capture_output:
                stdout.close()
                stderr.close()
                
        return result
        
    async def run_code(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        capture_output: bool = True
    ) -> Dict[str, Any]:
        """Compile and execute code in one step."""
        try:
            compiled = await self.compile_code(code, context)
            return await self.execute_code(compiled, context, capture_output)
        except Exception as e:
            return {
                'success': False,
                'error': {
                    'type': type(e).__name__,
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }
            }
            
    def cleanup(self):
        """Clean up sandbox resources."""
        self._restore_builtins() 