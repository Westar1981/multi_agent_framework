"""
Integration module for task allocation and collaborative learning systems.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import asyncio
import logging
from datetime import datetime
import json
from pathlib import Path

from .task_allocation import TaskAllocationManager, Task, TaskPriority
from ..learning.collaborative_learning import (
    CollaborativeLearningSystem,
    Experience,
    LearningPreference
)
from ..agents.base_agent import BaseAgent, AgentCapability
from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class LearningEnabledTaskManager:
    """Task manager with integrated collaborative learning capabilities."""
    
    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        config_path: Optional[str] = None
    ):
        self.memory_manager = memory_manager or MemoryManager()
        self.task_manager = TaskAllocationManager(self.memory_manager)
        self.learning_system = CollaborativeLearningSystem(self.memory_manager)
        self.config = self._load_config(config_path)
        self.task_history: Dict[str, List[Task]] = {}
        self.learning_enabled = True
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if not config_path:
            config_path = str(Path(__file__).parent / "task_allocation_config.json")
            
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
            
    async def register_agent(
        self,
        agent: BaseAgent,
        initial_preferences: Optional[LearningPreference] = None
    ) -> None:
        """Register an agent with both systems."""
        # Register with task manager
        await self.task_manager.register_agent(agent)
        
        # Create default preferences if none provided
        if not initial_preferences:
            initial_preferences = LearningPreference(
                preferred_tasks=set(),
                expertise_areas={cap: 0.5 for cap in agent.capabilities},
                learning_rate=self.config.get("learning_parameters", {}).get(
                    "base_learning_rate", 0.1
                ),
                exploration_rate=self.config.get("learning_parameters", {}).get(
                    "base_exploration_rate", 0.2
                )
            )
            
        # Register with learning system
        await self.learning_system.register_agent(agent, initial_preferences)
        
    async def submit_task(self, task: Task) -> bool:
        """Submit a task with learning-enhanced allocation."""
        if not self.learning_enabled:
            return await self.task_manager.submit_task(task)
            
        # Get recommendations for task handling
        recommendations = []
        for agent_id in self.task_manager.agent_capabilities:
            agent_recommendations = await self.learning_system.get_recommendations(
                agent_id, task
            )
            if agent_recommendations:
                recommendations.append((agent_id, agent_recommendations))
                
        # Adjust task parameters based on learning
        if recommendations:
            await self._enhance_task_with_learning(task, recommendations)
            
        # Submit to task manager
        success = await self.task_manager.submit_task(task)
        if success:
            self.task_history.setdefault(task.id, []).append(task)
            
        return success
        
    async def update_task_progress(
        self,
        task_id: str,
        progress: float,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Update task progress with learning feedback."""
        if task_id not in self.task_manager.tasks:
            logger.warning(f"Task {task_id} not found")
            return
            
        task = self.task_manager.tasks[task_id]
        old_progress = task.progress
        
        # Update task progress
        await self.task_manager.update_task_progress(task_id, progress)
        
        if not self.learning_enabled:
            return
            
        # Record experience if task completed
        if progress >= 1.0 and task.assigned_agent:
            await self._record_task_experience(task, performance_metrics or {})
            
        # Update agent preferences
        if task.assigned_agent:
            await self.learning_system.update_preferences(
                task.assigned_agent,
                progress,
                task.name
            )
            
    async def optimize_allocations(self) -> None:
        """Optimize task allocations using learning insights."""
        if not self.learning_enabled:
            await self.task_manager.optimize_allocations()
            return
            
        # Get learning insights for all agents
        insights = {}
        for agent_id in self.task_manager.agent_capabilities:
            agent_insights = await self.learning_system.get_learning_insights(agent_id)
            if agent_insights:
                insights[agent_id] = agent_insights
                
        # Use insights to optimize allocations
        await self._optimize_with_insights(insights)
        
    async def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get combined metrics for an agent."""
        metrics = {
            'workload': self.task_manager.agent_workloads.get(agent_id, 0.0),
            'capabilities': list(self.task_manager.agent_capabilities.get(agent_id, set()))
        }
        
        if self.learning_enabled:
            insights = await self.learning_system.get_learning_insights(agent_id)
            metrics.update({
                'learning_rate': insights.get('learning_rate', 0.0),
                'exploration_rate': insights.get('exploration_rate', 0.0),
                'performance_trend': insights.get('performance_trend', 0.0),
                'strengths': insights.get('strengths', []),
                'improvement_areas': insights.get('improvement_areas', [])
            })
            
        return metrics
        
    async def _enhance_task_with_learning(
        self,
        task: Task,
        recommendations: List[Tuple[str, List[Tuple[str, float]]]]
    ) -> None:
        """Enhance task parameters using learning recommendations."""
        if not recommendations:
            return
            
        # Aggregate recommendations
        action_scores: Dict[str, List[float]] = {}
        for _, agent_recommendations in recommendations:
            for action, score in agent_recommendations:
                action_scores.setdefault(action, []).append(score)
                
        # Calculate average scores
        avg_scores = {
            action: sum(scores) / len(scores)
            for action, scores in action_scores.items()
        }
        
        # Adjust task complexity based on recommended actions
        complexity_factor = sum(avg_scores.values()) / len(avg_scores)
        task.estimated_complexity *= max(0.5, min(1.5, complexity_factor))
        
    async def _record_task_experience(
        self,
        task: Task,
        performance_metrics: Dict[str, float]
    ) -> None:
        """Record task completion experience."""
        if not task.assigned_agent:
            return
            
        # Create experience record
        experience = Experience(
            agent_id=task.assigned_agent,
            task_id=task.id,
            task_type=task.name,
            capabilities_used=task.required_capabilities,
            context={
                'priority': task.priority.name,
                'complexity': task.estimated_complexity,
                'dependencies': list(task.dependencies)
            },
            actions_taken=self._extract_task_actions(task),
            outcome=self._calculate_task_outcome(task, performance_metrics),
            timestamp=datetime.now(),
            execution_time=performance_metrics.get('execution_time', 0.0),
            resource_usage={
                'memory': performance_metrics.get('memory_usage', 0.0),
                'cpu': performance_metrics.get('cpu_usage', 0.0),
                'complexity': task.estimated_complexity
            }
        )
        
        await self.learning_system.record_experience(experience)
        
    def _extract_task_actions(self, task: Task) -> List[str]:
        """Extract actions taken during task execution."""
        # This would be enhanced based on actual task execution tracking
        return [
            f"execute_{task.name.lower()}",
            f"handle_{task.priority.name.lower()}_priority"
        ]
        
    def _calculate_task_outcome(
        self,
        task: Task,
        metrics: Dict[str, float]
    ) -> float:
        """Calculate task outcome score."""
        # Base score
        if task.deadline and datetime.now() > task.deadline:
            base_score = 0.5  # Penalty for missing deadline
        else:
            base_score = 0.8
            
        # Adjust based on performance metrics
        metric_scores = []
        if 'error_rate' in metrics:
            metric_scores.append(1.0 - metrics['error_rate'])
        if 'quality_score' in metrics:
            metric_scores.append(metrics['quality_score'])
        if 'efficiency_score' in metrics:
            metric_scores.append(metrics['efficiency_score'])
            
        # Combine scores
        if metric_scores:
            return (base_score + sum(metric_scores) / len(metric_scores)) / 2
        return base_score
        
    async def _optimize_with_insights(
        self,
        insights: Dict[str, Dict[str, Any]]
    ) -> None:
        """Optimize allocations using learning insights."""
        # First, perform standard optimization
        await self.task_manager.optimize_allocations()
        
        # Then, apply learning-based adjustments
        for agent_id, agent_insights in insights.items():
            performance_trend = agent_insights.get('performance_trend', 0.0)
            
            if performance_trend < 0.5:  # Poor performance
                # Reduce workload for struggling agents
                current_workload = self.task_manager.agent_workloads.get(agent_id, 0.0)
                if current_workload > 0.5:
                    # Find tasks to reallocate
                    agent_tasks = [
                        task for task in self.task_manager.tasks.values()
                        if task.assigned_agent == agent_id and task.progress < 1.0
                    ]
                    
                    # Sort by priority and progress
                    agent_tasks.sort(key=lambda t: (
                        -t.priority.value,
                        -t.progress
                    ))
                    
                    # Try to reallocate some tasks
                    for task in agent_tasks[:len(agent_tasks)//2]:
                        await self._try_reallocate_task(task, agent_id, insights)
                        
    async def _try_reallocate_task(
        self,
        task: Task,
        current_agent: str,
        insights: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Try to reallocate a task to a better-suited agent."""
        best_score = 0.0
        best_agent = None
        
        for agent_id, agent_insights in insights.items():
            if agent_id == current_agent:
                continue
                
            # Calculate agent suitability score
            score = self._calculate_agent_suitability(
                task, agent_id, agent_insights
            )
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
                
        if best_agent and best_score > 0.7:  # Threshold for reallocation
            # Update task assignment
            task.assigned_agent = best_agent
            
            # Update workloads
            remaining_work = task.estimated_complexity * (1 - task.progress)
            self.task_manager.agent_workloads[current_agent] -= remaining_work
            self.task_manager.agent_workloads[best_agent] += remaining_work
            
            return True
            
        return False
        
    def _calculate_agent_suitability(
        self,
        task: Task,
        agent_id: str,
        agent_insights: Dict[str, Any]
    ) -> float:
        """Calculate how suitable an agent is for a task."""
        # Check capabilities
        if not task.required_capabilities.issubset(
            self.task_manager.agent_capabilities.get(agent_id, set())
        ):
            return 0.0
            
        # Base score on performance trend
        score = agent_insights.get('performance_trend', 0.0)
        
        # Adjust based on strengths
        if task.name in agent_insights.get('strengths', []):
            score *= 1.2
            
        # Adjust based on workload
        workload = self.task_manager.agent_workloads.get(agent_id, 0.0)
        if workload > 0.8:
            score *= 0.5
            
        return min(1.0, score) 