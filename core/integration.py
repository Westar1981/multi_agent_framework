"""
Integration module for agent coordination and communication.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import threading
from queue import Queue
import json

try:
    import aioredis
except ImportError:
    aioredis = None

try:
    import psutil
except ImportError:
    psutil = None

from ..agents.base_agent import BaseAgent, Message
from .optimization import OptimizationManager, OptimizationConfig

logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Configuration for agent integration."""
    redis_url: str = "redis://localhost"
    namespace: str = "multi_agent"
    message_timeout: int = 30
    batch_size: int = 100
    max_retries: int = 3
    enable_monitoring: bool = True
    resource_limits: Dict[str, float] = None
    
    def __post_init__(self):
        if self.resource_limits is None:
            self.resource_limits = {
                'memory': 0.8,  # 80% of available memory
                'cpu': 0.9,     # 90% of CPU
                'messages': 1000 # Maximum queued messages
            }

class AgentIntegration:
    """Manages integration between agents."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.redis = None
        self.message_queue: Queue = Queue()
        self.agents: Dict[str, BaseAgent] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self._setup_integration()
        
    def _setup_integration(self) -> None:
        """Setup integration components."""
        # Setup Redis if available
        if aioredis is not None:
            self._setup_redis_task = asyncio.create_task(self._setup_redis())
            
        # Setup monitoring if enabled
        if self.config.enable_monitoring and psutil is not None:
            self._monitoring_task = threading.Thread(
                target=self._monitor_resources,
                daemon=True
            )
            self._monitoring_task.start()
            
    async def _setup_redis(self) -> None:
        """Setup Redis connection."""
        try:
            self.redis = await aioredis.create_redis_pool(self.config.redis_url)
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            
    def _monitor_resources(self) -> None:
        """Monitor system resources."""
        while True:
            try:
                if psutil is not None:
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.metrics['memory_usage'].append(memory.percent / 100.0)
                    
                    # CPU usage
                    cpu = psutil.cpu_percent(interval=1) / 100.0
                    self.metrics['cpu_usage'].append(cpu)
                    
                    # Check resource limits
                    self._check_resource_limits()
                    
            except Exception as e:
                logger.error(f"Error monitoring resources: {str(e)}")
                
            time.sleep(5)  # Check every 5 seconds
            
    def _check_resource_limits(self) -> None:
        """Check and handle resource limit violations."""
        if not self.metrics:
            return
            
        memory_usage = self.metrics['memory_usage'][-1]
        cpu_usage = self.metrics['cpu_usage'][-1]
        queue_size = self.message_queue.qsize()
        
        if memory_usage > self.config.resource_limits['memory']:
            logger.warning(f"Memory usage ({memory_usage:.1%}) exceeds limit")
            self._handle_resource_violation('memory')
            
        if cpu_usage > self.config.resource_limits['cpu']:
            logger.warning(f"CPU usage ({cpu_usage:.1%}) exceeds limit")
            self._handle_resource_violation('cpu')
            
        if queue_size > self.config.resource_limits['messages']:
            logger.warning(f"Message queue ({queue_size}) exceeds limit")
            self._handle_resource_violation('messages')
            
    def _handle_resource_violation(self, resource: str) -> None:
        """Handle resource limit violation."""
        if resource == 'messages':
            # Clear old messages
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except:
                    break
        else:
            # Notify agents to reduce load
            self._notify_resource_pressure(resource)
            
    def _notify_resource_pressure(self, resource: str) -> None:
        """Notify agents of resource pressure."""
        message = Message(
            sender="integration",
            receiver="all",
            content={
                'type': 'resource_pressure',
                'resource': resource,
                'action': 'reduce_load'
            }
        )
        self.broadcast_message(message)
            
    async def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for integration."""
        self.agents[agent.id] = agent
        
        # Setup Redis subscription if available
        if self.redis is not None:
            channel = f"{self.config.namespace}:{agent.id}"
            await self.redis.subscribe(channel)
            
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            
            # Clean up subscriptions
            for topic in list(self.subscriptions.keys()):
                self.subscriptions[topic].discard(agent_id)
                
            # Unsubscribe from Redis if available
            if self.redis is not None:
                channel = f"{self.config.namespace}:{agent_id}"
                await self.redis.unsubscribe(channel)
                
    async def send_message(self, message: Message) -> None:
        """Send a message to an agent."""
        try:
            if message.receiver in self.agents:
                # Direct message
                await self.agents[message.receiver].handle_message(message)
                
            elif self.redis is not None:
                # Try Redis delivery
                channel = f"{self.config.namespace}:{message.receiver}"
                await self.redis.publish(channel, json.dumps(message.__dict__))
                
            else:
                # Queue for later delivery
                self.message_queue.put(message)
                
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            
    def broadcast_message(self, message: Message) -> None:
        """Broadcast message to all agents."""
        for agent_id in self.agents:
            message.receiver = agent_id
            asyncio.create_task(self.send_message(message))
            
    async def subscribe(self, agent_id: str, topics: List[str]) -> None:
        """Subscribe agent to topics."""
        for topic in topics:
            self.subscriptions[topic].add(agent_id)
            
    async def unsubscribe(self, agent_id: str, topics: List[str]) -> None:
        """Unsubscribe agent from topics."""
        for topic in topics:
            self.subscriptions[topic].discard(agent_id)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            'agents': len(self.agents),
            'subscriptions': {
                topic: len(subscribers)
                for topic, subscribers in self.subscriptions.items()
            },
            'queue_size': self.message_queue.qsize(),
            'metrics': {
                name: {
                    'current': values[-1] if values else 0,
                    'average': sum(values) / len(values) if values else 0
                }
                for name, values in self.metrics.items()
            }
        }
        
    async def shutdown(self) -> None:
        """Shutdown integration."""
        # Close Redis connection
        if self.redis is not None:
            self.redis.close()
            await self.redis.wait_closed()
            
        # Clear queues and metrics
        while not self.message_queue.empty():
            self.message_queue.get_nowait()
            
        self.metrics.clear()
        self.subscriptions.clear()
        self.agents.clear() 