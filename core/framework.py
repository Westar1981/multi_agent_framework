from typing import Dict, Any, List, Optional, Set
from loguru import logger
from ..agents.base_agent import BaseAgent, AgentCapability
import asyncio
from collections import defaultdict
from pathlib import Path
import logging
import time
from dataclasses import dataclass
import psutil
import threading
from queue import Queue

from .self_analysis import SelfAnalyzer, SystemState, PerformanceMetric
from .coordinator import Coordinator
from .orchestrator import Orchestrator
from ..utils.visualizer import Visualizer

logger = logging.getLogger(__name__)

@dataclass
class FrameworkConfig:
    """Framework configuration settings."""
    enable_self_analysis: bool = True
    monitoring_interval: float = 1.0  # seconds
    performance_thresholds: Dict[str, float] = None
    visualization_enabled: bool = True
    log_level: str = "INFO"

class LogicEngine:
    """Handles reasoning and decision making for the framework."""
    
    def __init__(self):
        self.rules = []
        self.facts = []
        self.capability_graph = defaultdict(set)
        
    def add_rule(self, rule: str):
        """Add a new reasoning rule."""
        self.rules.append(rule)
        
    def add_fact(self, fact: str):
        """Add a new fact."""
        self.facts.append(fact)
        
    def update_capability_graph(self, agent: BaseAgent):
        """Update the capability dependency graph."""
        for capability in agent.get_capabilities():
            for input_type in capability.input_types:
                self.capability_graph[input_type].add(capability.name)
            for output_type in capability.output_types:
                self.capability_graph[capability.name].add(output_type)

    def find_capability_chain(self, input_type: str, output_type: str) -> List[str]:
        """Find a chain of capabilities that can transform input to output."""
        visited = set()
        path = []
        
        def dfs(current: str) -> bool:
            if current == output_type:
                return True
            if current in visited:
                return False
                
            visited.add(current)
            for next_node in self.capability_graph[current]:
                path.append(next_node)
                if dfs(next_node):
                    return True
                path.pop()
            return False
        
        dfs(input_type)
        return path

class HotPluggableAgentFramework:
    """Framework that supports dynamic agent addition and removal."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.logic_engine = LogicEngine()
        self.running_tasks: Set[asyncio.Task] = set()
        
    def register_agent(self, agent: BaseAgent):
        """Hot-plug a new agent into the framework."""
        self.agents[agent.name] = agent
        self.logic_engine.update_capability_graph(agent)
        logger.info(f"Registered agent: {agent.name}")
        
    def unregister_agent(self, agent_name: str):
        """Remove an agent without disrupting the system."""
        if agent_name in self.agents:
            agent = self.agents.pop(agent_name)
            # Cancel agent's tasks gracefully
            for task in self.running_tasks:
                if task.get_name() == agent_name:
                    task.cancel()
            logger.info(f"Unregistered agent: {agent_name}")
            
    async def start_agent(self, agent_name: str):
        """Start an agent's processing loop."""
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            task = asyncio.create_task(agent.start(), name=agent_name)
            self.running_tasks.add(task)
            task.add_done_callback(self.running_tasks.discard)
            logger.info(f"Started agent: {agent_name}")
            
    async def stop_agent(self, agent_name: str):
        """Stop an agent's processing loop."""
        if agent_name in self.agents:
            await self.agents[agent_name].stop()
            logger.info(f"Stopped agent: {agent_name}")
            
    def get_agent_capabilities(self, agent_name: str) -> List[AgentCapability]:
        """Get the capabilities of a specific agent."""
        if agent_name in self.agents:
            return self.agents[agent_name].get_capabilities()
        return []
        
    def get_agent_metrics(self, agent_name: str) -> Optional[Dict[str, float]]:
        """Get performance metrics for a specific agent."""
        if agent_name in self.agents:
            return self.agents[agent_name].get_performance_metrics()
        return None
        
    def get_system_state(self) -> Dict[str, Any]:
        """Get the current state of the entire system."""
        return {
            "agents": {
                name: agent.get_state()
                for name, agent in self.agents.items()
            },
            "capability_graph": dict(self.logic_engine.capability_graph),
            "active_tasks": len(self.running_tasks)
        }
        
    async def find_capable_agent(self, capability_name: str) -> Optional[BaseAgent]:
        """Find an agent that can handle a specific capability."""
        for agent in self.agents.values():
            if any(cap.name == capability_name for cap in agent.get_capabilities()):
                return agent
        return None
        
    async def execute_capability_chain(self, input_data: Any, input_type: str,
                                     output_type: str) -> Optional[Any]:
        """Execute a chain of capabilities to transform input to desired output."""
        chain = self.logic_engine.find_capability_chain(input_type, output_type)
        if not chain:
            return None
            
        current_data = input_data
        for capability in chain:
            agent = await self.find_capable_agent(capability)
            if not agent:
                continue
                
            try:
                current_data = await agent._execute_task({
                    "type": capability,
                    "data": current_data
                })
            except Exception as e:
                logger.error(f"Error in capability chain at {capability}: {str(e)}")
                return None
                
        return current_data
        
    async def broadcast_message(self, content: Dict[str, Any], message_type: str,
                              exclude_agents: Optional[List[str]] = None):
        """Broadcast a message to all agents."""
        exclude_agents = exclude_agents or []
        for agent_name, agent in self.agents.items():
            if agent_name not in exclude_agents:
                await agent.send_message(
                    receiver="broadcast",
                    content=content,
                    message_type=message_type
                )
                
    def add_agent_dependency(self, agent_name: str, dependency: str):
        """Add a dependency between agents."""
        if agent_name in self.agents:
            self.agents[agent_name].dependencies.add(dependency)
            
    def remove_agent_dependency(self, agent_name: str, dependency: str):
        """Remove a dependency between agents."""
        if agent_name in self.agents:
            self.agents[agent_name].dependencies.discard(dependency)
            
    def get_agent_dependencies(self, agent_name: str) -> Set[str]:
        """Get all dependencies for an agent."""
        if agent_name in self.agents:
            return self.agents[agent_name].dependencies
        return set()
        
    def validate_system_state(self) -> bool:
        """Validate the current system state."""
        # Check for circular dependencies
        visited = set()
        path = set()
        
        def has_cycle(agent_name: str) -> bool:
            if agent_name in path:
                return True
            if agent_name in visited:
                return False
                
            visited.add(agent_name)
            path.add(agent_name)
            
            for dep in self.get_agent_dependencies(agent_name):
                if has_cycle(dep):
                    return True
                    
            path.remove(agent_name)
            return False
            
        for agent_name in self.agents:
            if has_cycle(agent_name):
                return False
                
        return True 

class Framework:
    """Core framework with self-analysis and monitoring."""
    
    def __init__(self, config: Optional[FrameworkConfig] = None):
        self.config = config or FrameworkConfig()
        self.coordinator = Coordinator()
        self.orchestrator = Orchestrator()
        self.visualizer = Visualizer() if self.config.visualization_enabled else None
        
        # Initialize self-analysis components
        self.analyzer = SelfAnalyzer() if self.config.enable_self_analysis else None
        self.monitoring_queue = Queue()
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
    def start(self):
        """Start the framework and monitoring."""
        logger.info("Starting framework...")
        
        # Start core components
        self.coordinator.start()
        self.orchestrator.start()
        if self.visualizer:
            self.visualizer.start()
            
        # Start monitoring if enabled
        if self.analyzer:
            self._start_monitoring()
            
        logger.info("Framework started successfully")
        
    def stop(self):
        """Stop the framework and cleanup."""
        logger.info("Stopping framework...")
        
        # Stop monitoring
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join()
                
        # Stop core components
        if self.visualizer:
            self.visualizer.stop()
        self.orchestrator.stop()
        self.coordinator.stop()
        
        logger.info("Framework stopped successfully")
        
    def _start_monitoring(self):
        """Start system monitoring thread."""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_state = self._collect_system_state()
                
                # Update analyzer
                self.analyzer.capture_state(system_state)
                
                # Process any pending analysis tasks
                self._process_analysis_queue()
                
                # Generate and handle improvement suggestions
                self._handle_improvements()
                
                # Update visualization if enabled
                if self.visualizer:
                    self._update_visualization()
                    
                # Wait for next monitoring interval
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
    def _collect_system_state(self) -> SystemState:
        """Collect current system state metrics."""
        # Get active agents
        active_agents = self.coordinator.get_active_agents()
        
        # Collect memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage = {
            'framework': memory_info.rss / (1024 * 1024),  # MB
            'agents': self.coordinator.get_agent_memory_usage()
        }
        
        # Collect processing load
        cpu_percent = process.cpu_percent()
        processing_load = {
            'framework': cpu_percent,
            'agents': self.coordinator.get_agent_processing_load()
        }
        
        # Collect knowledge stats
        knowledge_stats = {
            'rules': len(self.orchestrator.get_knowledge_base()),
            'patterns': len(self.orchestrator.get_patterns()),
            'active_queries': len(self.orchestrator.get_active_queries())
        }
        
        # Collect error rates
        error_rates = {
            agent: stats.get('error_rate', 0.0)
            for agent, stats in self.coordinator.get_agent_stats().items()
        }
        
        return SystemState(
            active_agents=active_agents,
            memory_usage=memory_usage,
            processing_load=processing_load,
            knowledge_stats=knowledge_stats,
            error_rates=error_rates
        )
        
    def _process_analysis_queue(self):
        """Process pending analysis tasks."""
        while not self.monitoring_queue.empty():
            task = self.monitoring_queue.get()
            try:
                if task['type'] == 'analyze_performance':
                    self._analyze_performance(task['target'])
                elif task['type'] == 'optimize_component':
                    self._optimize_component(task['component'], task['params'])
            except Exception as e:
                logger.error(f"Error processing analysis task: {e}")
                
    def _analyze_performance(self, target: str):
        """Analyze performance of specific target."""
        if target == 'agents':
            stats = self.coordinator.get_agent_stats()
            for agent, agent_stats in stats.items():
                self._check_performance_thresholds(agent, agent_stats)
        elif target == 'system':
            system_stats = self.analyzer.get_analysis_report()
            self._handle_system_analysis(system_stats)
            
    def _check_performance_thresholds(self, agent: str, stats: Dict[str, Any]):
        """Check if performance metrics exceed thresholds."""
        if self.config.performance_thresholds:
            for metric, threshold in self.config.performance_thresholds.items():
                if metric in stats and stats[metric] > threshold:
                    self._handle_threshold_violation(agent, metric, stats[metric])
                    
    def _handle_threshold_violation(self, agent: str, metric: str, value: float):
        """Handle performance threshold violation."""
        logger.warning(f"Performance threshold violated - Agent: {agent}, Metric: {metric}, Value: {value}")
        
        # Add to optimization queue
        self.monitoring_queue.put({
            'type': 'optimize_component',
            'component': agent,
            'params': {'metric': metric, 'value': value}
        })
        
    def _optimize_component(self, component: str, params: Dict[str, Any]):
        """Optimize specific component based on analysis."""
        logger.info(f"Optimizing component: {component} with params: {params}")
        
        # Get optimization strategy
        strategy = self.analyzer.suggest_improvements()
        if strategy:
            # Apply optimization
            self.coordinator.optimize_agent(component, strategy)
            
    def _handle_improvements(self):
        """Handle improvement suggestions from analyzer."""
        suggestions = self.analyzer.suggest_improvements()
        for suggestion in suggestions:
            if "Priority: High" in suggestion:
                self._handle_high_priority_improvement(suggestion)
            else:
                logger.info(f"Improvement suggestion: {suggestion}")
                
    def _handle_high_priority_improvement(self, suggestion: str):
        """Handle high priority improvement suggestion."""
        logger.warning(f"High priority improvement needed: {suggestion}")
        
        # Add to optimization queue
        self.monitoring_queue.put({
            'type': 'optimize_component',
            'component': 'system',
            'params': {'suggestion': suggestion}
        })
        
    def _handle_system_analysis(self, stats: Dict[str, Any]):
        """Handle system-wide analysis results."""
        if 'metrics' in stats:
            for metric_name, metric_data in stats['metrics'].items():
                if metric_data[-1].value > metric_data[-1].threshold:
                    logger.warning(f"System metric threshold exceeded: {metric_name}")
                    self._handle_system_metric_violation(metric_name, metric_data[-1])
                    
    def _handle_system_metric_violation(self, metric: str, data: PerformanceMetric):
        """Handle system metric threshold violation."""
        logger.warning(f"System metric violation - {metric}: {data.value}")
        
        # Add to optimization queue
        self.monitoring_queue.put({
            'type': 'optimize_component',
            'component': 'system',
            'params': {'metric': metric, 'data': data.__dict__}
        })
        
    def _update_visualization(self):
        """Update system visualization."""
        if self.visualizer:
            analysis_report = self.analyzer.get_analysis_report()
            self.visualizer.update_system_metrics(analysis_report)
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""
        if not self.analyzer:
            return {}
            
        return {
            'analysis': self.analyzer.get_analysis_report(),
            'active_agents': self.coordinator.get_active_agents(),
            'system_health': self._get_system_health()
        }
        
    def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        return {
            'memory_usage': psutil.Process().memory_percent(),
            'cpu_usage': psutil.Process().cpu_percent(),
            'active_threads': threading.active_count(),
            'queue_size': self.monitoring_queue.qsize()
        }