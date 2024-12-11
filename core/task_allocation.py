"""
Hierarchical task allocation system with dynamic workload balancing.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from datetime import datetime
import numpy as np

from .memory_manager import MemoryManager
from ..agents.base_agent import BaseAgent, AgentCapability
from .optimization import OptimizationConfig

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """Represents a task to be allocated to agents."""
    id: str
    name: str
    priority: TaskPriority
    required_capabilities: Set[AgentCapability]
    estimated_complexity: float  # 0.0 to 1.0
    dependencies: Set[str]  # Task IDs this task depends on
    created_at: datetime
    deadline: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    
class TaskAllocationManager:
    """Manages hierarchical task allocation and dynamic workload balancing."""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager or MemoryManager()
        self.tasks: Dict[str, Task] = {}
        self.agent_workloads: Dict[str, float] = {}  # 0.0 to 1.0
        self.agent_capabilities: Dict[str, Set[AgentCapability]] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        
    async def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with its capabilities."""
        self.agent_capabilities[agent.id] = agent.capabilities
        self.agent_workloads[agent.id] = 0.0
        self.performance_history[agent.id] = []
        
    async def submit_task(self, task: Task) -> bool:
        """Submit a new task for allocation."""
        if task.id in self.tasks:
            logger.warning(f"Task {task.id} already exists")
            return False
            
        self.tasks[task.id] = task
        # Priority tuple: (priority value, complexity, creation time)
        priority_tuple = (
            -task.priority.value,  # Negative so higher priority comes first
            task.estimated_complexity,
            task.created_at.timestamp()
        )
        await self.task_queue.put((priority_tuple, task))
        return True
        
    async def allocate_tasks(self) -> List[Tuple[str, str]]:
        """Allocate pending tasks to agents based on capabilities and workload."""
        allocations: List[Tuple[str, str]] = []  # (task_id, agent_id)
        
        while not self.task_queue.empty():
            _, task = await self.task_queue.get()
            
            if not self._are_dependencies_completed(task):
                # Put back in queue with slightly lower priority
                priority_tuple = (
                    -task.priority.value + 0.1,
                    task.estimated_complexity,
                    task.created_at.timestamp()
                )
                await self.task_queue.put((priority_tuple, task))
                continue
                
            best_agent = await self._find_best_agent(task)
            if best_agent:
                task.assigned_agent = best_agent
                self.agent_workloads[best_agent] += task.estimated_complexity
                allocations.append((task.id, best_agent))
                
        return allocations
        
    async def update_task_progress(self, task_id: str, progress: float) -> None:
        """Update task progress and adjust workload calculations."""
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found")
            return
            
        task = self.tasks[task_id]
        old_progress = task.progress
        task.progress = progress
        
        if task.assigned_agent:
            # Adjust workload based on progress
            progress_delta = progress - old_progress
            self.agent_workloads[task.assigned_agent] -= (
                task.estimated_complexity * progress_delta
            )
            
        if progress >= 1.0:
            await self._handle_task_completion(task)
            
    async def optimize_allocations(self) -> None:
        """Optimize task allocations based on performance metrics."""
        # Calculate agent performance scores
        performance_scores = self._calculate_performance_scores()
        
        # Find overloaded agents
        overloaded_agents = {
            agent_id: workload
            for agent_id, workload in self.agent_workloads.items()
            if workload > 0.8  # 80% workload threshold
        }
        
        # Redistribute tasks from overloaded agents
        for agent_id in overloaded_agents:
            agent_tasks = [
                task for task in self.tasks.values()
                if task.assigned_agent == agent_id and task.progress < 1.0
            ]
            
            # Sort tasks by priority and progress
            agent_tasks.sort(key=lambda t: (
                -t.priority.value,
                -t.progress
            ))
            
            # Try to reallocate some tasks
            for task in agent_tasks[:len(agent_tasks)//2]:  # Reallocate half
                new_agent = await self._find_best_agent(task, exclude={agent_id})
                if new_agent:
                    # Update allocations
                    self.agent_workloads[agent_id] -= (
                        task.estimated_complexity * (1 - task.progress)
                    )
                    self.agent_workloads[new_agent] += (
                        task.estimated_complexity * (1 - task.progress)
                    )
                    task.assigned_agent = new_agent
                    
    def _calculate_performance_scores(self) -> Dict[str, float]:
        """Calculate performance scores for each agent."""
        scores = {}
        for agent_id, history in self.performance_history.items():
            if not history:
                scores[agent_id] = 0.5  # Default score
                continue
                
            # Calculate weighted average of recent performance
            weights = np.exp(-np.arange(len(history)) / 5)  # Exponential decay
            weights /= weights.sum()
            scores[agent_id] = np.average(history, weights=weights)
            
        return scores
        
    async def _find_best_agent(
        self, task: Task, exclude: Optional[Set[str]] = None
    ) -> Optional[str]:
        """Find the best agent for a task based on capabilities and workload."""
        exclude = exclude or set()
        candidates = []
        
        for agent_id, capabilities in self.agent_capabilities.items():
            if agent_id in exclude:
                continue
                
            if not task.required_capabilities.issubset(capabilities):
                continue
                
            # Calculate agent score
            workload_score = 1 - self.agent_workloads[agent_id]
            performance_score = np.mean(self.performance_history[agent_id]) \
                              if self.performance_history[agent_id] else 0.5
            
            # Combine scores (weighted sum)
            score = 0.7 * workload_score + 0.3 * performance_score
            candidates.append((score, agent_id))
            
        if not candidates:
            return None
            
        # Return agent with highest score
        candidates.sort(reverse=True)
        return candidates[0][1]
        
    def _are_dependencies_completed(self, task: Task) -> bool:
        """Check if all task dependencies are completed."""
        return all(
            self.tasks[dep_id].progress >= 1.0
            for dep_id in task.dependencies
            if dep_id in self.tasks
        )
        
    async def _handle_task_completion(self, task: Task) -> None:
        """Handle task completion and update metrics."""
        if not task.assigned_agent:
            return
            
        # Update performance history
        if task.deadline:
            on_time = datetime.now() <= task.deadline
            performance = 1.0 if on_time else 0.5
        else:
            performance = 0.8  # Default good performance for tasks without deadline
            
        self.performance_history[task.assigned_agent].append(performance)
        
        # Keep performance history bounded
        if len(self.performance_history[task.assigned_agent]) > 100:
            self.performance_history[task.assigned_agent] = \
                self.performance_history[task.assigned_agent][-100:]
                
        # Clean up task from workload
        self.agent_workloads[task.assigned_agent] = max(
            0.0,
            self.agent_workloads[task.assigned_agent] - task.estimated_complexity
        ) 