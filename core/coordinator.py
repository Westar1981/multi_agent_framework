"""
Coordinator module for managing agent interactions and optimization.
"""

from collections import defaultdict
from queue import Queue
import asyncio
import logging
import psutil
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Type, Union
from dataclasses import dataclass

from ..agents.base_agent import BaseAgent, Message
from ..agents.neural_symbolic_agent import NeuralSymbolicAgent
from ..agents.meta_reasoner import MetaReasoner
from ..agents.learner_agent import LearnerAgent
try:
    from .optimization import OptimizationManager, OptimizationConfig
except ImportError:
    OptimizationManager = None
    OptimizationConfig = None
from .integration import AgentIntegration

# Handle optional PrologReasoner import
try:
    from ..agents.prolog_reasoner import PrologReasoner
except ImportError:
    PrologReasoner = None  # type: Optional[Type[BaseAgent]]

logger = logging.getLogger(__name__)

@dataclass
class CoordinatorConfig:
    """Configuration for agent coordination."""
    enable_optimization: bool = True
    enable_integration: bool = True
    optimization_config: Optional[Dict[str, Any]] = None
    integration_config: Optional[Dict[str, Any]] = None

class Coordinator:
    """Manages agent interactions and system optimization."""
    
    def __init__(self, config: Optional[CoordinatorConfig] = None) -> None:
        """Initialize coordinator with optional configuration.
        
        Args:
            config: Optional coordinator configuration
        """
        self.config = config or CoordinatorConfig()
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: List[asyncio.Task[Any]] = []
        self.agent_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.optimization_queue: Queue[Dict[str, Any]] = Queue()
        self.is_running: bool = False
        self.optimization_thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()
        
        # Initialize components
        self.optimization_manager: Optional[OptimizationManager] = None
        self.integration: Optional[AgentIntegration] = None
        self._setup_components()
        
        logger.info("Coordinator initialized")
        
    def _setup_components(self) -> None:
        """Setup optimization and integration components."""
        if self.config.enable_optimization and OptimizationManager is not None:
            opt_config = OptimizationConfig(**self.config.optimization_config) if self.config.optimization_config else OptimizationConfig()
            self.optimization_manager = OptimizationManager(opt_config)
            logger.info("Optimization manager initialized")
        else:
            logger.warning("Optimization disabled or OptimizationManager not available")
            
        if self.config.enable_integration:
            self.integration = AgentIntegration(self.config.integration_config)
            logger.info("Agent integration initialized")
            
    async def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the coordinator.
        
        Args:
            agent: Agent instance to register
        """
        self.agents[agent.id] = agent
        if self.integration:
            await self.integration.register_agent(agent)
            
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent.
        
        Args:
            agent_id: ID of agent to unregister
        """
        if agent_id in self.agents:
            if self.integration:
                await self.integration.unregister_agent(agent_id)
            del self.agents[agent_id]
            
    async def send_message(self, message: Message) -> None:
        """Send a message to an agent.
        
        Args:
            message: Message to send
        """
        if message.receiver in self.agents:
            await self.agents[message.receiver].handle_message(message)
        elif self.integration:
            await self.integration.send_message(message)
            
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID.
        
        Args:
            agent_id: ID of agent to retrieve
            
        Returns:
            Optional[BaseAgent]: Agent if found, None otherwise
        """
        return self.agents.get(agent_id)
        
    def get_agents_by_type(self, agent_type: Type[BaseAgent]) -> List[BaseAgent]:
        """Get all agents of a specific type.
        
        Args:
            agent_type: Type of agents to retrieve
            
        Returns:
            List[BaseAgent]: List of matching agents
        """
        return [
            agent for agent in self.agents.values()
            if isinstance(agent, agent_type)
        ]
        
    async def optimize_performance(self) -> None:
        """Optimize system performance."""
        if self.optimization_manager:
            try:
                await self.optimization_manager.optimize()
            except Exception as e:
                logger.error(f"Optimization failed: {str(e)}")
                
    async def shutdown(self) -> None:
        """Shutdown the coordinator and all components."""
        if self.optimization_manager:
            await self.optimization_manager.shutdown()
            
        if self.integration:
            await self.integration.shutdown()
            
        # Shutdown all agents
        for agent in self.agents.values():
            await agent.shutdown()
            
        self.agents.clear()
