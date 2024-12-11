"""
Agent coordination module for managing multi-agent interactions.
"""

from enum import Enum, auto
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import asyncio
import heapq
from collections import defaultdict
import numpy as np

class TaskPriority(Enum):
    """Priority levels for agent tasks."""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()

class CoordinationStrategy(Enum):
    """Strategies for coordinating agent interactions."""
    ROUND_ROBIN = auto()
    LOAD_BALANCED = auto()
    PRIORITY_BASED = auto()
    SPECIALIZED = auto()
    ADAPTIVE = auto()  # Dynamically adjusts based on performance
    HIERARCHICAL = auto()  # Uses agent hierarchy for delegation
    COLLABORATIVE = auto()  # Enables multi-agent collaboration
    AUCTION_BASED = auto()  # Agents bid for tasks

@dataclass
class AgentCapability:
    """Represents an agent's capabilities and expertise."""
    specialization: str
    expertise_level: float  # 0.0 to 1.0
    supported_tasks: Set[str]
    performance_score: float = 0.8

@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    recipient: str
    content: Any
    priority: TaskPriority
    timestamp: datetime
    task_type: Optional[str] = None
    requires_collaboration: bool = False
    max_processing_time: Optional[float] = None
    
    def __lt__(self, other):
        """Enable priority queue ordering."""
        return (self.priority.value, self.timestamp) < (other.priority.value, other.timestamp)

@dataclass
class CollaborationGroup:
    """Group of agents working together on a task."""
    primary_agent: str
    supporting_agents: List[str]
    task_type: str
    formation_time: datetime
    performance_history: List[float] = None

class AgentCoordinator:
    """Coordinates interactions between multiple agents."""
    
    def __init__(self, strategy: CoordinationStrategy = CoordinationStrategy.ADAPTIVE):
        self.agents = {}  # name -> agent mapping
        self.agent_capabilities = {}  # name -> capabilities mapping
        self.message_queue = []  # priority queue of messages
        self.strategy = strategy
        self.agent_loads = defaultdict(int)  # Track agent workloads
        self.error_count = 0
        self.collaboration_groups = []  # Active collaboration groups
        self.performance_history = defaultdict(list)  # Track agent performance
        self.adaptive_thresholds = {
            'load_threshold': 0.8,
            'performance_threshold': 0.7,
            'collaboration_threshold': 0.85
        }
        self.metrics = {
            'messages_processed': 0,
            'errors': 0,
            'avg_response_time': 0.0,
            'collaboration_success_rate': 1.0
        }
        
    def register_agent(self, agent, capabilities: Optional[AgentCapability] = None):
        """Register an agent with the coordinator."""
        self.agents[agent.name] = agent
        if capabilities:
            self.agent_capabilities[agent.name] = capabilities
        
    async def submit_message(self, message: AgentMessage):
        """Submit a message for processing."""
        if message.requires_collaboration:
            await self._setup_collaboration(message)
        heapq.heappush(self.message_queue, message)
        
    async def _setup_collaboration(self, message: AgentMessage):
        """Setup a collaboration group for complex tasks."""
        primary_agent = self._select_primary_agent(message.task_type)
        if not primary_agent:
            return
            
        # Select supporting agents based on complementary capabilities
        supporting_agents = self._select_supporting_agents(
            primary_agent,
            message.task_type,
            max_supporters=3
        )
        
        group = CollaborationGroup(
            primary_agent=primary_agent,
            supporting_agents=supporting_agents,
            task_type=message.task_type,
            formation_time=datetime.now(),
            performance_history=[]
        )
        self.collaboration_groups.append(group)
        
    def _select_primary_agent(self, task_type: str) -> Optional[str]:
        """Select the best primary agent for a task type."""
        candidates = []
        for name, capabilities in self.agent_capabilities.items():
            if task_type in capabilities.supported_tasks:
                score = capabilities.expertise_level * (1 - self.agent_loads[name]/10)
                candidates.append((name, score))
                
        return max(candidates, key=lambda x: x[1])[0] if candidates else None
        
    def _select_supporting_agents(self, primary: str, task_type: str, max_supporters: int) -> List[str]:
        """Select supporting agents with complementary capabilities."""
        candidates = []
        primary_capabilities = self.agent_capabilities[primary]
        
        for name, capabilities in self.agent_capabilities.items():
            if name != primary and task_type in capabilities.supported_tasks:
                # Calculate complementarity score
                score = (capabilities.expertise_level * 
                        (1 - self.agent_loads[name]/10) *
                        (1 - self._capability_overlap(capabilities, primary_capabilities)))
                candidates.append((name, score))
                
        # Select top scoring candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in candidates[:max_supporters]]
        
    def _capability_overlap(self, cap1: AgentCapability, cap2: AgentCapability) -> float:
        """Calculate capability overlap between two agents."""
        common_tasks = cap1.supported_tasks.intersection(cap2.supported_tasks)
        all_tasks = cap1.supported_tasks.union(cap2.supported_tasks)
        return len(common_tasks) / len(all_tasks) if all_tasks else 0
        
    def _select_agent(self, specialization: str = None) -> Optional[str]:
        """Select an agent based on current strategy."""
        if not self.agents:
            return None
            
        if self.strategy == CoordinationStrategy.ROUND_ROBIN:
            agents = list(self.agents.keys())
            return agents[self.metrics['messages_processed'] % len(agents)]
            
        elif self.strategy == CoordinationStrategy.LOAD_BALANCED:
            return min(self.agent_loads.items(), key=lambda x: x[1])[0]
            
        elif self.strategy == CoordinationStrategy.SPECIALIZED:
            if specialization:
                matching = [
                    name for name, agent in self.agents.items()
                    if agent.specialization == specialization
                ]
                if matching:
                    return min(
                        matching,
                        key=lambda x: self.agent_loads[x]
                    )
                    
        elif self.strategy == CoordinationStrategy.ADAPTIVE:
            return self._adaptive_selection(specialization)
            
        elif self.strategy == CoordinationStrategy.AUCTION_BASED:
            return self._auction_selection(specialization)
            
        return list(self.agents.keys())[0]
        
    def _adaptive_selection(self, specialization: str = None) -> str:
        """Adaptively select agent based on performance history."""
        candidates = {}
        for name, agent in self.agents.items():
            if specialization and agent.specialization != specialization:
                continue
                
            # Calculate adaptive score
            performance = np.mean(self.performance_history[name]) if self.performance_history[name] else 0.8
            load_factor = self.agent_loads[name] / 10
            expertise = self.agent_capabilities[name].expertise_level if name in self.agent_capabilities else 0.5
            
            score = (0.4 * performance + 
                    0.3 * (1 - load_factor) + 
                    0.3 * expertise)
            candidates[name] = score
            
        return max(candidates.items(), key=lambda x: x[1])[0]
        
    def _auction_selection(self, specialization: str = None) -> str:
        """Select agent through auction mechanism."""
        bids = {}
        for name, agent in self.agents.items():
            if specialization and agent.specialization != specialization:
                continue
                
            # Agents bid based on their current state
            load_factor = self.agent_loads[name] / 10
            expertise = self.agent_capabilities[name].expertise_level if name in self.agent_capabilities else 0.5
            performance = np.mean(self.performance_history[name]) if self.performance_history[name] else 0.8
            
            # Higher bid means more suitable for task
            bid = (0.4 * (1 - load_factor) + 
                  0.3 * expertise + 
                  0.3 * performance)
            bids[name] = bid
            
        return max(bids.items(), key=lambda x: x[1])[0]
        
    async def process_queue(self) -> List[AgentMessage]:
        """Process all messages in the queue."""
        results = []
        start_time = datetime.now()
        
        while self.message_queue:
            message = heapq.heappop(self.message_queue)
            
            try:
                # Get target agent
                agent = self.agents.get(message.recipient)
                if not agent:
                    agent = self.agents[self._select_agent(
                        message.task_type if hasattr(message, 'task_type') else None
                    )]
                
                # Update load tracking
                self.agent_loads[agent.name] += 1
                
                # Process message
                result = await agent.process_message(message)
                if result:
                    results.append(result)
                    # Update performance history
                    processing_time = (datetime.now() - start_time).total_seconds()
                    if message.max_processing_time:
                        performance = max(0, 1 - processing_time / message.max_processing_time)
                        self.performance_history[agent.name].append(performance)
                
                self.metrics['messages_processed'] += 1
                
            except Exception as e:
                self.error_count += 1
                self.metrics['errors'] += 1
                print(f"Error processing message: {str(e)}")
                
            finally:
                # Update load tracking
                if agent:
                    self.agent_loads[agent.name] -= 1
                
        # Update metrics
        if results:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            self.metrics['avg_response_time'] = processing_time / len(results)
            
        return results
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current coordination metrics."""
        return {
            **self.metrics,
            'queue_size': len(self.message_queue),
            'agent_loads': dict(self.agent_loads),
            'error_rate': self.metrics['errors'] / max(1, self.metrics['messages_processed']),
            'active_collaborations': len(self.collaboration_groups),
            'agent_performance': {
                name: np.mean(scores) if scores else 0
                for name, scores in self.performance_history.items()
            }
        }
        
    def update_adaptive_thresholds(self):
        """Update adaptive thresholds based on system performance."""
        if not self.performance_history:
            return
            
        # Calculate system-wide performance metrics
        avg_performance = np.mean([
            np.mean(scores) for scores in self.performance_history.values()
        ])
        
        # Adjust thresholds
        adjustment = 0.1 * (avg_performance - 0.8)  # 0.8 is the target performance
        self.adaptive_thresholds['load_threshold'] += adjustment
        self.adaptive_thresholds['performance_threshold'] += adjustment
        
        # Ensure thresholds stay within reasonable bounds
        for key in self.adaptive_thresholds:
            self.adaptive_thresholds[key] = max(0.5, min(0.95, self.adaptive_thresholds[key])) 