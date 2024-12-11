"""
Coordinator module for managing agent interactions and optimization.
"""

from typing import Dict, Any, List, Optional
import logging
import asyncio
import uuid
from collections import defaultdict
import threading
import time
import psutil
from queue import Queue

from ..agents.base_agent import BaseAgent, Message
from ..agents.neural_symbolic_agent import NeuralSymbolicAgent
from ..agents.prolog_reasoner import PrologReasoner
from ..agents.meta_reasoner import MetaReasoner
from ..agents.learner_agent import LearnerAgent
from .optimization import OptimizationManager, OptimizationConfig
from .integration import AgentIntegration, IntegrationConfig

logger = logging.getLogger(__name__)

class Coordinator:
    """Manages agent interactions and system optimization."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: List[asyncio.Task] = []
        self.agent_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.optimization_queue = Queue()
        self.is_running = False
        self.optimization_thread = None
        self._lock = threading.Lock()
        
        # Initialize integration and optimization
        self.integration = AgentIntegration()
        self.optimizer = OptimizationManager()
        
        logger.info("Coordinator initialized")
        
    async def start(self):
        """Start coordinator and optimization thread."""
        logger.info("Starting coordinator...")
        self.is_running = True
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
        
        # Start all agents
        for agent in self.agents.values():
            task = asyncio.create_task(agent.start())
            self.tasks.append(task)
            
        logger.info("Coordinator started successfully")
        
    async def stop(self):
        """Stop coordinator and cleanup."""
        logger.info("Stopping coordinator...")
        self.is_running = False
        
        # Stop optimization thread
        if self.optimization_thread:
            self.optimization_thread.join()
            
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
            
        # Wait for all tasks to complete
        if self.tasks:
            await asyncio.wait(self.tasks)
            
        self.tasks.clear()
        logger.info("Coordinator stopped successfully")
        
    def register_agent(self, agent_id: str, agent: BaseAgent):
        """Register a new agent."""
        with self._lock:
            if agent_id in self.agents:
                raise ValueError(f"Agent {agent_id} already registered")
                
            self.agents[agent_id] = agent
            self.agent_stats[agent_id] = self._initialize_agent_stats()
            
            # Register with integration module
            self.integration.message_queues[agent_id] = asyncio.Queue()
            self.integration.agent_states[agent_id] = {
                'type': self._get_agent_type(agent),
                'capabilities': agent.get_capabilities()
            }
            
            logger.info(f"Agent {agent_id} registered successfully")
            
    async def send_message(self, 
                          sender_id: Optional[str],
                          receiver_id: str,
                          content: Any,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send a direct message to a specific agent."""
        message = Message(
            id=str(uuid.uuid4()),
            sender=sender_id or "coordinator",
            receiver=receiver_id,
            content=content,
            metadata=metadata or {}
        )
        
        return await self.integration.route_message(message, 'direct')
        
    async def broadcast_message(self,
                              sender_id: Optional[str],
                              content: Any,
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Broadcast message to all agents."""
        message = Message(
            id=str(uuid.uuid4()),
            sender=sender_id or "coordinator",
            receiver="broadcast",
            content=content,
            metadata=metadata or {}
        )
        
        return await self.integration.route_message(message, 'broadcast')
        
    async def sync_agents(self,
                         source_id: str,
                         target_id: str,
                         sync_type: str = 'knowledge') -> bool:
        """Synchronize state between agents."""
        if source_id not in self.agents or target_id not in self.agents:
            logger.error(f"Invalid agent IDs for sync: {source_id}, {target_id}")
            return False
            
        return await self.integration.sync_agents(
            self.agents[source_id],
            self.agents[target_id],
            sync_type
        )
        
    def get_active_agents(self) -> List[str]:
        """Get list of active agent IDs."""
        return list(self.agents.keys())
        
    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get current agent statistics."""
        return self.agent_stats
        
    def get_agent_memory_usage(self) -> Dict[str, float]:
        """Get memory usage for each agent."""
        memory_usage = {}
        for agent_id, agent in self.agents.items():
            try:
                memory_usage[agent_id] = self.agent_stats[agent_id]['memory_usage']
            except Exception as e:
                logger.error(f"Error getting memory usage for agent {agent_id}: {e}")
                memory_usage[agent_id] = 0.0
        return memory_usage
        
    def get_agent_processing_load(self) -> Dict[str, float]:
        """Get processing load for each agent."""
        processing_load = {}
        for agent_id in self.agents:
            try:
                times = self.agent_stats[agent_id]['processing_time']
                if times:
                    processing_load[agent_id] = sum(times) / len(times)
                else:
                    processing_load[agent_id] = 0.0
            except Exception as e:
                logger.error(f"Error getting processing load for agent {agent_id}: {e}")
                processing_load[agent_id] = 0.0
        return processing_load
        
    def update_agent_stats(self, 
                          agent_id: str,
                          stats_update: Dict[str, Any]):
        """Update statistics for an agent."""
        with self._lock:
            if agent_id not in self.agent_stats:
                logger.warning(f"No stats found for agent {agent_id}")
                return
                
            for key, value in stats_update.items():
                if key == 'processing_time':
                    self.agent_stats[agent_id]['processing_time'].append(value)
                    if len(self.agent_stats[agent_id]['processing_time']) > 100:
                        self.agent_stats[agent_id]['processing_time'].pop(0)
                else:
                    self.agent_stats[agent_id][key] = value
                    
            # Update integration state
            self.integration.agent_states[agent_id].update(stats_update)
            
    def optimize_agent(self, 
                      agent_id: str,
                      strategy: List[str]):
        """Queue agent optimization task."""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found for optimization")
            return
            
        self.optimization_queue.put({
            'agent_id': agent_id,
            'strategy': strategy
        })
        
    def _optimization_loop(self):
        """Main optimization loop."""
        while self.is_running:
            try:
                if self.optimization_queue.empty():
                    time.sleep(1)
                    continue
                    
                task = self.optimization_queue.get()
                agent_id = task['agent_id']
                strategy = task['strategy']
                
                self._apply_optimization(agent_id, strategy)
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                
    def _apply_optimization(self, 
                          agent_id: str,
                          strategy: List[str]):
        """Apply optimization strategy to agent."""
        logger.info(f"Applying optimization to agent {agent_id}: {strategy}")
        
        try:
            agent = self.agents[agent_id]
            
            for step in strategy:
                if "memory" in step.lower():
                    self._optimize_memory(agent)
                elif "processing" in step.lower():
                    self._optimize_processing(agent)
                elif "accuracy" in step.lower():
                    self._optimize_accuracy(agent)
                    
            self.agent_stats[agent_id]['optimization_count'] += 1
            
        except Exception as e:
            logger.error(f"Error optimizing agent {agent_id}: {e}")
            
    def _optimize_memory(self, agent: BaseAgent):
        """Optimize agent memory usage."""
        if isinstance(agent, NeuralSymbolicAgent):
            agent.clear_cache()
            agent.optimize_knowledge_base()
        elif isinstance(agent, PrologReasoner):
            agent.cleanup_knowledge_base()
            
    def _optimize_processing(self, agent: BaseAgent):
        """Optimize agent processing efficiency."""
        if isinstance(agent, NeuralSymbolicAgent):
            agent.optimize_batch_size()
            agent.update_processing_strategy()
        elif isinstance(agent, MetaReasoner):
            agent.optimize_reasoning_patterns()
            
    def _optimize_accuracy(self, agent: BaseAgent):
        """Optimize agent accuracy."""
        if isinstance(agent, NeuralSymbolicAgent):
            agent.enhance_validation()
            agent.adjust_learning_rate()
        elif isinstance(agent, PrologReasoner):
            agent.refine_inference_rules()
            
    def get_agent_optimization_status(self, 
                                    agent_id: str) -> Dict[str, Any]:
        """Get optimization status for an agent."""
        if agent_id not in self.agent_stats:
            return {}
            
        return {
            'optimization_count': self.agent_stats[agent_id]['optimization_count'],
            'current_performance': {
                'error_rate': self.agent_stats[agent_id]['error_rate'],
                'success_rate': self.agent_stats[agent_id]['success_rate'],
                'memory_usage': self.agent_stats[agent_id]['memory_usage']
            }
        }
        
    def _initialize_agent_stats(self) -> Dict[str, Any]:
        """Initialize agent statistics."""
        return {
            'error_rate': 0.0,
            'processing_time': [],
            'memory_usage': 0.0,
            'success_rate': 1.0,
            'optimization_count': 0
        }
        
    def _get_agent_type(self, agent: BaseAgent) -> str:
        """Get agent type string."""
        if isinstance(agent, NeuralSymbolicAgent):
            return 'neural_symbolic'
        elif isinstance(agent, PrologReasoner):
            return 'prolog'
        elif isinstance(agent, MetaReasoner):
            return 'meta'
        elif isinstance(agent, LearnerAgent):
            return 'learner'
        else:
            return 'base'
