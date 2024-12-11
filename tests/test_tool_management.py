"""
Tests for tool management system.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from ..core.tool_management import (
    Tool,
    ToolResult,
    ToolRegistry,
    ToolOrchestrator,
    ToolAnalytics,
    ToolUsageMetrics
)

class MockTool(Tool):
    """Mock tool for testing."""
    
    def __init__(self, name: str, success: bool = True, latency: float = 0.1):
        super().__init__(name, f"Mock tool {name}")
        self.should_succeed = success
        self.latency = latency
        
    async def execute(self, *args, **kwargs) -> ToolResult:
        """Execute mock tool."""
        await asyncio.sleep(self.latency)
        if self.should_succeed:
            return ToolResult(
                success=True,
                result="mock_result",
                execution_time=self.latency
            )
        return ToolResult(
            success=False,
            result=None,
            error="mock_error",
            execution_time=self.latency
        )

@pytest.fixture
def registry():
    """Create tool registry."""
    return ToolRegistry()

@pytest.fixture
def orchestrator(registry):
    """Create tool orchestrator."""
    return ToolOrchestrator(registry)

@pytest.fixture
def analytics(registry):
    """Create tool analytics."""
    return ToolAnalytics(registry)

class TestToolRegistry:
    """Test tool registry functionality."""
    
    def test_register_tool(self, registry):
        """Test tool registration."""
        tool = MockTool("test_tool")
        registry.register_tool(tool)
        assert "test_tool" in registry.tools
        
    def test_get_tool(self, registry):
        """Test getting registered tool."""
        tool = MockTool("test_tool")
        registry.register_tool(tool)
        retrieved = registry.get_tool("test_tool")
        assert retrieved == tool
        
    def test_list_tools(self, registry):
        """Test listing registered tools."""
        tools = [
            MockTool("tool1"),
            MockTool("tool2"),
            MockTool("tool3")
        ]
        for tool in tools:
            registry.register_tool(tool)
            
        tool_list = registry.list_tools()
        assert len(tool_list) == 3
        assert all(t['name'] in ["tool1", "tool2", "tool3"] for t in tool_list)
        
    def test_analyze_usage_patterns(self, registry):
        """Test usage pattern analysis."""
        tool = MockTool("test_tool")
        registry.register_tool(tool)
        
        # Simulate some usage
        tool.metrics.total_calls = 10
        tool.metrics.successful_calls = 8
        tool.metrics.average_latency = 0.5
        
        patterns = registry.analyze_usage_patterns()
        assert "test_tool" in patterns
        assert patterns["test_tool"]["success_rate"] == 0.8
        assert patterns["test_tool"]["average_latency"] == 0.5

class TestToolOrchestrator:
    """Test tool orchestrator functionality."""
    
    @pytest.mark.asyncio
    async def test_execute_tool(self, orchestrator):
        """Test tool execution."""
        tool = MockTool("test_tool")
        orchestrator.registry.register_tool(tool)
        
        result = await orchestrator.execute_tool("test_tool", "test_agent")
        assert result.success
        assert result.result == "mock_result"
        
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, orchestrator):
        """Test executing non-existent tool."""
        result = await orchestrator.execute_tool("nonexistent", "test_agent")
        assert not result.success
        assert "not found" in result.error
        
    @pytest.mark.asyncio
    async def test_permission_checking(self, orchestrator):
        """Test tool permission checking."""
        tool = MockTool("test_tool")
        tool.required_permissions = ["special_permission"]
        orchestrator.registry.register_tool(tool)
        
        result = await orchestrator.execute_tool("test_tool", "test_agent")
        assert not result.success
        assert "permission" in result.error
        
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, orchestrator):
        """Test concurrent tool execution."""
        tools = [
            MockTool(f"tool{i}", latency=0.1)
            for i in range(3)
        ]
        for tool in tools:
            orchestrator.registry.register_tool(tool)
            
        results = await asyncio.gather(*[
            orchestrator.execute_tool(f"tool{i}", "test_agent")
            for i in range(3)
        ])
        
        assert all(r.success for r in results)
        assert len(results) == 3

class TestToolAnalytics:
    """Test tool analytics functionality."""
    
    def test_generate_analytics_report(self, analytics):
        """Test analytics report generation."""
        tool = MockTool("test_tool")
        analytics.registry.register_tool(tool)
        
        # Simulate usage
        tool.metrics.total_calls = 100
        tool.metrics.successful_calls = 90
        tool.metrics.average_latency = 1.5
        tool.metrics.error_types = {"ValueError": 5, "TypeError": 5}
        
        report = analytics.generate_analytics_report()
        assert "usage_patterns" in report
        assert "system_recommendations" in report
        assert "optimization_opportunities" in report
        
    def test_recommendations_generation(self, analytics):
        """Test recommendation generation."""
        tool = MockTool("test_tool")
        analytics.registry.register_tool(tool)
        
        # Simulate poor performance
        tool.metrics.total_calls = 100
        tool.metrics.successful_calls = 70  # 70% success rate
        tool.metrics.average_latency = 2.0  # High latency
        
        report = analytics.generate_analytics_report()
        recommendations = report["system_recommendations"]
        
        assert any("success rate" in r.lower() for r in recommendations)
        assert any("performance" in r.lower() for r in recommendations)
        
    def test_optimization_identification(self, analytics):
        """Test optimization opportunity identification."""
        tools = [
            MockTool("frequent_tool", latency=0.1),
            MockTool("slow_tool", latency=2.0)
        ]
        for tool in tools:
            analytics.registry.register_tool(tool)
            
        # Simulate usage patterns
        tools[0].metrics.total_calls = 200  # High frequency
        tools[1].metrics.average_latency = 2.0  # High latency
        
        report = analytics.generate_analytics_report()
        optimizations = report["optimization_opportunities"]
        
        assert any(opt["type"] == "batching" for opt in optimizations)
        assert any(opt["type"] == "performance" for opt in optimizations)

@pytest.mark.asyncio
async def test_end_to_end_tool_usage(registry, orchestrator, analytics):
    """Test end-to-end tool usage workflow."""
    # Register tools
    tools = [
        MockTool("tool1", success=True, latency=0.1),
        MockTool("tool2", success=False, latency=0.2),
        MockTool("tool3", success=True, latency=1.5)
    ]
    for tool in tools:
        registry.register_tool(tool)
        
    # Execute tools
    results = await asyncio.gather(*[
        orchestrator.execute_tool(t.name, "test_agent")
        for t in tools
    ])
    
    # Verify results
    assert len(results) == 3
    assert results[0].success
    assert not results[1].success
    assert results[2].success
    
    # Check analytics
    report = analytics.generate_analytics_report()
    patterns = report["usage_patterns"]
    
    assert len(patterns) == 3
    assert patterns["tool1"]["success_rate"] == 1.0
    assert patterns["tool2"]["success_rate"] == 0.0
    assert patterns["tool3"]["average_latency"] > 1.0 