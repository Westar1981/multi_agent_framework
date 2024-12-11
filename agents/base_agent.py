from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Set
from pydantic import BaseModel
from loguru import logger
import asyncio
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime

from ..core.memory_manager import MemoryManager, MemoryConfig
from ..core.optimization import OptimizationConfig

@dataclass
class AgentCapability:
    """Represents a capability that an agent can perform."""
    name: str
    description: str
    input_types: Set[str]
    output_types: Set[str]
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class Message(BaseModel):
    """Message class for inter-agent communication."""
    id: str
    sender: str
    receiver: str
    content: Dict[str, Any]
    message_type: str

class PerformanceMetrics:
    """Tracks agent performance metrics."""
    def __init__(self):
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_time = 0.0
        self.avg_response_time = 0.0
        self.error_rate = 0.0
        
    def update(self, execution_time: float, success: bool):
        """Update metrics with new task execution data."""
        self.total_tasks += 1
        if success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
        
        self.total_time += execution_time
        self.avg_response_time = self.total_time / self.total_tasks
        self.error_rate = self.failed_tasks / self.total_tasks if self.total_tasks > 0 else 0.0

@dataclass
class AgentConfig:
    """Agent configuration."""
    name: str
    capabilities: Set[str]
    memory_config: Optional[MemoryConfig] = None
    optimization_config: Optional[OptimizationConfig] = None

class BaseAgent(ABC):
    """Enhanced base class for all agents in the system."""
    
    def __init__(self, config: AgentConfig):
        self.name = config.name
        self.capabilities = config.capabilities
        self.memory_manager = MemoryManager(config.memory_config)
        self.optimization_config = config.optimization_config
        self.is_initialized = False
        self.metrics = {
            'processed_messages': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'avg_processing_time': 0.0
        }
        
    async def initialize(self):
        """Initialize agent and start memory monitoring."""
        if self.is_initialized:
            return
            
        # Start memory monitoring
        asyncio.create_task(self.memory_manager.start_monitoring())
        self.is_initialized = True
        
    async def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process incoming message with memory management."""
        if not self.is_initialized:
            await self.initialize()
            
        self.metrics['processed_messages'] += 1
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(message)
            cached_response = self.memory_manager.get(cache_key)
            
            if cached_response is not None:
                return cached_response
                
            # Process message
            response = await self._process_message_internal(message)
            
            if response is not None:
                # Cache successful response
                await self.memory_manager.set(cache_key, response)
                self.metrics['successful_operations'] += 1
            else:
                self.metrics['failed_operations'] += 1
                
            # Update processing time metric
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_processing_time(processing_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message in {self.name}: {str(e)}")
            self.metrics['failed_operations'] += 1
            return None
            
    async def _process_message_internal(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Internal message processing - override in subclasses."""
        raise NotImplementedError
        
    def _get_cache_key(self, message: Dict[str, Any]) -> str:
        """Generate cache key for message."""
        # Basic implementation - override for more sophisticated key generation
        return f"{self.name}:{message.get('type', '')}:{str(message.get('content', ''))}"
        
    def _update_processing_time(self, processing_time: float):
        """Update average processing time metric."""
        current_avg = self.metrics['avg_processing_time']
        total_messages = self.metrics['processed_messages']
        
        self.metrics['avg_processing_time'] = (
            (current_avg * (total_messages - 1) + processing_time) / total_messages
        )
        
    async def optimize(self):
        """Optimize agent performance."""
        # Optimize memory usage
        memory_metrics = await self.memory_manager.optimize()
        
        # Get current performance metrics
        performance_metrics = self.get_metrics()
        
        return {
            'memory': memory_metrics,
            'performance': performance_metrics
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        return {
            **self.metrics,
            'memory': self.memory_manager.get_metrics()
        }
        
    async def shutdown(self):
        """Cleanup resources."""
        # Cleanup will happen automatically through garbage collection
        self.is_initialized = False

    async def start(self):
        """Start the agent's main processing loop."""
        self.running = True
        logger.info(f"Starting agent: {self.name}")
        while self.running:
            try:
                message = await self.message_queue.get()
                start_time = time.time()
                try:
                    await self.process_message(message)
                    execution_time = time.time() - start_time
                    self.performance_metrics.update(execution_time, True)
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.performance_metrics.update(execution_time, False)
                    logger.error(f"Error processing message in {self.name}: {str(e)}")
                finally:
                    self.message_queue.task_done()
            except Exception as e:
                logger.error(f"Critical error in {self.name}'s message loop: {str(e)}")

    async def stop(self):
        """Stop the agent's processing loop."""
        self.running = False
        logger.info(f"Stopping agent: {self.name}")

    async def send_message(self, receiver: str, content: Dict[str, Any], message_type: str):
        """Send a message to another agent."""
        message = Message(
            id=str(uuid.uuid4()),
            sender=self.agent_id,
            receiver=receiver,
            content=content,
            message_type=message_type
        )
        await self.message_queue.put(message)
        logger.debug(f"Message sent from {self.name} to {receiver}")

    @abstractmethod
    async def handle_task(self, task: Dict[str, Any]):
        """Handle a specific task. Must be implemented by subclasses."""
        pass

    async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with performance tracking."""
        start_time = time.time()
        try:
            result = await self.handle_task(task)
            execution_time = time.time() - start_time
            self.performance_metrics.update(execution_time, True)
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            self.performance_metrics.update(execution_time, False)
            logger.error(f"Task execution failed in {self.name}: {str(e)}")
            raise

    def get_capabilities(self) -> List[AgentCapability]:
        """Get the agent's current capabilities."""
        return self.capabilities

    def add_capability(self, capability: AgentCapability):
        """Add a new capability to the agent."""
        self.capabilities.append(capability)
        logger.info(f"Added capability {capability.name} to {self.name}")

    def remove_capability(self, capability_name: str):
        """Remove a capability from the agent."""
        self.capabilities = [c for c in self.capabilities if c.name != capability_name]
        logger.info(f"Removed capability {capability_name} from {self.name}")

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get the agent's current performance metrics."""
        return {
            "total_tasks": self.performance_metrics.total_tasks,
            "successful_tasks": self.performance_metrics.successful_tasks,
            "failed_tasks": self.performance_metrics.failed_tasks,
            "avg_response_time": self.performance_metrics.avg_response_time,
            "error_rate": self.performance_metrics.error_rate
        }

    def get_state(self) -> Dict[str, Any]:
        """Get the agent's current state."""
        return {
            "id": self.agent_id,
            "name": self.name,
            "running": self.running,
            "capabilities": [vars(c) for c in self.capabilities],
            "performance": self.get_performance_metrics(),
            "dependencies": list(self.dependencies),
            "state": self.state
        }

    def __str__(self):
        return f"{self.name} ({self.agent_id})"
