"""
Tool management system for coordinating and tracking tool usage across agents.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging
from abc import ABC, abstractmethod
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ToolUsageMetrics:
    """Metrics for tool usage."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_latency: float = 0.0
    last_used: Optional[datetime] = None
    error_types: Dict[str, int] = None
    
    def __post_init__(self):
        if self.error_types is None:
            self.error_types = {}

@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

class Tool(ABC):
    """Base class for tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.metrics = ToolUsageMetrics()
        self.parameters: Dict[str, Any] = {}
        self.required_permissions: List[str] = []
        
    @abstractmethod
    async def execute(self, *args, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass
        
    def update_metrics(self, result: ToolResult):
        """Update usage metrics."""
        self.metrics.total_calls += 1
        self.metrics.last_used = datetime.now()
        
        if result.success:
            self.metrics.successful_calls += 1
        else:
            self.metrics.failed_calls += 1
            error_type = type(result.error).__name__
            self.metrics.error_types[error_type] = self.metrics.error_types.get(error_type, 0) + 1
            
        # Update average latency
        if self.metrics.total_calls == 1:
            self.metrics.average_latency = result.execution_time
        else:
            self.metrics.average_latency = (
                (self.metrics.average_latency * (self.metrics.total_calls - 1) + result.execution_time) / 
                self.metrics.total_calls
            )

class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.tool_configs = self._load_tool_configs()
        self.usage_patterns: Dict[str, List[Dict[str, Any]]] = {}
        
    def _load_tool_configs(self) -> Dict[str, Any]:
        """Load tool configurations."""
        config_path = Path(__file__).parent / "tool_configs.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}
        
    def register_tool(self, tool: Tool):
        """Register a new tool."""
        if tool.name in self.tools:
            logger.warning(f"Tool {tool.name} already registered. Updating registration.")
        self.tools[tool.name] = tool
        self.usage_patterns[tool.name] = []
        
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a registered tool by name."""
        return self.tools.get(name)
        
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        return [
            {
                'name': tool.name,
                'description': tool.description,
                'metrics': tool.metrics.__dict__,
                'parameters': tool.parameters,
                'required_permissions': tool.required_permissions
            }
            for tool in self.tools.values()
        ]
        
    def analyze_usage_patterns(self) -> Dict[str, Any]:
        """Analyze tool usage patterns."""
        patterns = {}
        
        for tool_name, tool in self.tools.items():
            metrics = tool.metrics
            
            # Calculate success rate
            success_rate = (metrics.successful_calls / metrics.total_calls 
                          if metrics.total_calls > 0 else 0)
            
            # Analyze error patterns
            common_errors = sorted(
                metrics.error_types.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            patterns[tool_name] = {
                'success_rate': success_rate,
                'average_latency': metrics.average_latency,
                'common_errors': common_errors,
                'usage_frequency': metrics.total_calls,
                'last_used': metrics.last_used
            }
            
        return patterns

class ToolOrchestrator:
    """Orchestrates tool execution and manages permissions."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.permission_cache: Dict[str, List[str]] = {}
        self.execution_queue = asyncio.Queue()
        self.max_concurrent_executions = 5
        
    async def execute_tool(self, 
                          tool_name: str,
                          agent_id: str,
                          *args,
                          **kwargs) -> ToolResult:
        """Execute a tool with permission checking."""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                result=None,
                error=f"Tool {tool_name} not found"
            )
            
        # Check permissions
        if not self._check_permissions(agent_id, tool):
            return ToolResult(
                success=False,
                result=None,
                error=f"Agent {agent_id} does not have permission to use {tool_name}"
            )
            
        # Add to execution queue
        await self.execution_queue.put((tool, args, kwargs))
        
        # Process queue
        return await self._process_execution()
        
    async def _process_execution(self) -> ToolResult:
        """Process tool execution from queue."""
        tool, args, kwargs = await self.execution_queue.get()
        try:
            start_time = datetime.now()
            result = await tool.execute(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result.execution_time = execution_time
            tool.update_metrics(result)
            
            return result
        except Exception as e:
            error_result = ToolResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            tool.update_metrics(error_result)
            return error_result
        finally:
            self.execution_queue.task_done()
            
    def _check_permissions(self, agent_id: str, tool: Tool) -> bool:
        """Check if agent has permission to use tool."""
        if agent_id not in self.permission_cache:
            self.permission_cache[agent_id] = self._load_agent_permissions(agent_id)
            
        agent_permissions = self.permission_cache[agent_id]
        return all(perm in agent_permissions for perm in tool.required_permissions)
        
    def _load_agent_permissions(self, agent_id: str) -> List[str]:
        """Load agent permissions from configuration."""
        # This could be expanded to load from a database or permission service
        return ["basic_tools", "file_access", "network_access"]  # Default permissions

class ToolAnalytics:
    """Analytics for tool usage patterns."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        
    def generate_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        patterns = self.registry.analyze_usage_patterns()
        
        return {
            'usage_patterns': patterns,
            'system_recommendations': self._generate_recommendations(patterns),
            'optimization_opportunities': self._identify_optimizations(patterns)
        }
        
    def _generate_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate tool usage recommendations."""
        recommendations = []
        
        for tool_name, stats in patterns.items():
            if stats['success_rate'] < 0.8:
                recommendations.append(
                    f"Consider reviewing {tool_name} implementation - low success rate"
                )
                
            if stats['average_latency'] > 1.0:  # More than 1 second
                recommendations.append(
                    f"Optimize {tool_name} for better performance"
                )
                
            if len(stats['common_errors']) > 0:
                recommendations.append(
                    f"Address common errors in {tool_name}: {stats['common_errors'][0][0]}"
                )
                
        return recommendations
        
    def _identify_optimizations(self, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential optimization opportunities."""
        optimizations = []
        
        # Look for tools that could be batched
        high_frequency_tools = [
            name for name, stats in patterns.items()
            if stats['usage_frequency'] > 100  # Arbitrary threshold
        ]
        if high_frequency_tools:
            optimizations.append({
                'type': 'batching',
                'tools': high_frequency_tools,
                'suggestion': 'Consider implementing batch processing'
            })
            
        # Look for tools with high latency
        slow_tools = [
            name for name, stats in patterns.items()
            if stats['average_latency'] > 1.0
        ]
        if slow_tools:
            optimizations.append({
                'type': 'performance',
                'tools': slow_tools,
                'suggestion': 'Implement caching or optimize execution'
            })
            
        return optimizations 