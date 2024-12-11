"""
Collaborative learning system for sharing and learning from agent experiences.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
import numpy as np
import logging
from datetime import datetime
import asyncio
from collections import defaultdict

from ..core.memory_manager import MemoryManager
from ..agents.base_agent import BaseAgent, AgentCapability
from ..core.task_allocation import Task, TaskPriority

logger = logging.getLogger(__name__)

@dataclass
class Experience:
    """Represents a learning experience."""
    agent_id: str
    task_id: str
    task_type: str
    capabilities_used: Set[AgentCapability]
    context: Dict[str, Any]
    actions_taken: List[str]
    outcome: float  # 0.0 to 1.0 success metric
    timestamp: datetime
    execution_time: float
    resource_usage: Dict[str, float]
    
@dataclass
class LearningPreference:
    """Agent's learning preferences and strengths."""
    preferred_tasks: Set[str]
    expertise_areas: Dict[AgentCapability, float]  # Capability -> Proficiency
    learning_rate: float
    exploration_rate: float
    
class CollaborativeLearningSystem:
    """Manages collaborative learning and experience sharing between agents."""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager or MemoryManager()
        self.experiences: Dict[str, List[Experience]] = defaultdict(list)
        self.agent_preferences: Dict[str, LearningPreference] = {}
        self.similarity_matrix: Dict[Tuple[str, str], float] = {}
        self.task_clusters: Dict[str, List[str]] = defaultdict(list)
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        
    async def register_agent(self, agent: BaseAgent, preferences: LearningPreference) -> None:
        """Register an agent with its learning preferences."""
        self.agent_preferences[agent.id] = preferences
        await self._update_similarity_matrix(agent.id)
        
    async def record_experience(self, experience: Experience) -> None:
        """Record a new learning experience."""
        self.experiences[experience.agent_id].append(experience)
        
        # Update performance history
        self.performance_history[experience.agent_id].append(experience.outcome)
        
        # Keep history bounded
        if len(self.performance_history[experience.agent_id]) > 100:
            self.performance_history[experience.agent_id] = \
                self.performance_history[experience.agent_id][-100:]
                
        # Update task clusters
        await self._update_task_clusters(experience)
        
        # Clean up old experiences
        await self._cleanup_old_experiences()
        
    async def get_recommendations(
        self, agent_id: str, task: Task
    ) -> List[Tuple[str, float]]:
        """Get action recommendations for a task based on similar experiences."""
        if agent_id not in self.agent_preferences:
            return []
            
        # Find similar experiences
        similar_experiences = await self._find_similar_experiences(agent_id, task)
        
        if not similar_experiences:
            return []
            
        # Calculate action success rates
        action_scores: Dict[str, List[float]] = defaultdict(list)
        for exp in similar_experiences:
            for action in exp.actions_taken:
                # Weight by experience similarity and recency
                similarity = self._calculate_experience_similarity(task, exp)
                recency_weight = self._calculate_recency_weight(exp.timestamp)
                weighted_score = exp.outcome * similarity * recency_weight
                action_scores[action].append(weighted_score)
                
        # Calculate final recommendations
        recommendations = [
            (action, np.mean(scores))
            for action, scores in action_scores.items()
            if len(scores) >= 3  # Require minimum number of samples
        ]
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
        
    async def update_preferences(
        self, agent_id: str, task_outcome: float, task_type: str
    ) -> None:
        """Update agent preferences based on task outcomes."""
        if agent_id not in self.agent_preferences:
            return
            
        preferences = self.agent_preferences[agent_id]
        
        # Update task preferences
        if task_outcome >= 0.8:  # Successful outcome
            preferences.preferred_tasks.add(task_type)
            
        # Adjust learning rate based on recent performance
        recent_performance = self.performance_history[agent_id][-10:]
        if recent_performance:
            performance_trend = np.mean(recent_performance)
            if performance_trend > 0.8:
                preferences.learning_rate *= 0.95  # Reduce learning rate when performing well
            elif performance_trend < 0.5:
                preferences.learning_rate *= 1.05  # Increase learning rate when struggling
                
        # Adjust exploration rate
        preferences.exploration_rate = max(
            0.1,  # Minimum exploration
            preferences.exploration_rate * 0.99  # Gradually reduce exploration
        )
        
    async def get_learning_insights(
        self, agent_id: str
    ) -> Dict[str, Any]:
        """Get learning insights for an agent."""
        if agent_id not in self.agent_preferences:
            return {}
            
        recent_experiences = self.experiences[agent_id][-20:]
        if not recent_experiences:
            return {}
            
        # Calculate performance trends
        performance_trend = np.mean([exp.outcome for exp in recent_experiences])
        
        # Identify strengths and weaknesses
        task_performance = defaultdict(list)
        for exp in recent_experiences:
            task_performance[exp.task_type].append(exp.outcome)
            
        strengths = [
            task_type for task_type, outcomes in task_performance.items()
            if np.mean(outcomes) >= 0.8 and len(outcomes) >= 3
        ]
        
        weaknesses = [
            task_type for task_type, outcomes in task_performance.items()
            if np.mean(outcomes) < 0.6 and len(outcomes) >= 3
        ]
        
        # Find improvement opportunities
        improvement_areas = []
        for capability, proficiency in self.agent_preferences[agent_id].expertise_areas.items():
            if proficiency < 0.7:  # Below desired proficiency
                similar_agents = self._find_similar_agents(agent_id)
                better_agents = [
                    agent for agent in similar_agents
                    if self.agent_preferences[agent].expertise_areas.get(capability, 0) > proficiency
                ]
                if better_agents:
                    improvement_areas.append({
                        'capability': capability,
                        'current_proficiency': proficiency,
                        'target_proficiency': max(
                            self.agent_preferences[agent].expertise_areas[capability]
                            for agent in better_agents
                        ),
                        'recommended_mentors': better_agents[:2]  # Top 2 mentors
                    })
                    
        return {
            'performance_trend': performance_trend,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'improvement_areas': improvement_areas,
            'learning_rate': self.agent_preferences[agent_id].learning_rate,
            'exploration_rate': self.agent_preferences[agent_id].exploration_rate
        }
        
    async def _update_similarity_matrix(self, agent_id: str) -> None:
        """Update agent similarity matrix."""
        for other_id, other_prefs in self.agent_preferences.items():
            if other_id == agent_id:
                continue
                
            agent_prefs = self.agent_preferences[agent_id]
            
            # Calculate similarity components
            capability_similarity = self._calculate_capability_similarity(
                agent_prefs.expertise_areas,
                other_prefs.expertise_areas
            )
            
            task_similarity = len(
                agent_prefs.preferred_tasks & other_prefs.preferred_tasks
            ) / max(
                len(agent_prefs.preferred_tasks | other_prefs.preferred_tasks),
                1
            )
            
            # Combine similarities
            similarity = 0.7 * capability_similarity + 0.3 * task_similarity
            
            # Update matrix symmetrically
            self.similarity_matrix[(agent_id, other_id)] = similarity
            self.similarity_matrix[(other_id, agent_id)] = similarity
            
    def _calculate_capability_similarity(
        self,
        capabilities1: Dict[AgentCapability, float],
        capabilities2: Dict[AgentCapability, float]
    ) -> float:
        """Calculate similarity between capability sets."""
        all_capabilities = set(capabilities1.keys()) | set(capabilities2.keys())
        if not all_capabilities:
            return 0.0
            
        similarity = 0.0
        for cap in all_capabilities:
            prof1 = capabilities1.get(cap, 0.0)
            prof2 = capabilities2.get(cap, 0.0)
            similarity += 1.0 - abs(prof1 - prof2)
            
        return similarity / len(all_capabilities)
        
    async def _update_task_clusters(self, experience: Experience) -> None:
        """Update task clustering based on new experience."""
        task_type = experience.task_type
        
        # Find similar tasks based on capability requirements and outcomes
        similar_tasks = []
        for other_exp in self.experiences[experience.agent_id]:
            if other_exp.task_type == task_type:
                continue
                
            # Calculate task similarity
            cap_similarity = len(
                experience.capabilities_used & other_exp.capabilities_used
            ) / len(
                experience.capabilities_used | other_exp.capabilities_used
            )
            
            outcome_similarity = 1.0 - abs(experience.outcome - other_exp.outcome)
            
            similarity = 0.6 * cap_similarity + 0.4 * outcome_similarity
            
            if similarity >= 0.8:  # High similarity threshold
                similar_tasks.append(other_exp.task_type)
                
        # Update clusters
        self.task_clusters[task_type].extend(similar_tasks)
        for similar_task in similar_tasks:
            self.task_clusters[similar_task].append(task_type)
            
        # Keep clusters bounded
        for task_type in self.task_clusters:
            self.task_clusters[task_type] = list(set(self.task_clusters[task_type]))[:5]
            
    async def _find_similar_experiences(
        self, agent_id: str, task: Task
    ) -> List[Experience]:
        """Find experiences similar to the given task."""
        if not self.experiences[agent_id]:
            return []
            
        # Get experiences from similar agents
        similar_agents = self._find_similar_agents(agent_id)
        all_experiences = []
        for agent in similar_agents:
            all_experiences.extend(self.experiences[agent])
            
        # Filter and score experiences
        scored_experiences = []
        for exp in all_experiences:
            similarity = self._calculate_experience_similarity(task, exp)
            if similarity >= 0.6:  # Similarity threshold
                scored_experiences.append((similarity, exp))
                
        # Sort by similarity and return top experiences
        scored_experiences.sort(reverse=True)
        return [exp for _, exp in scored_experiences[:10]]
        
    def _find_similar_agents(self, agent_id: str, threshold: float = 0.6) -> List[str]:
        """Find agents similar to the given agent."""
        similar_agents = [
            other_id
            for other_id in self.agent_preferences.keys()
            if other_id != agent_id and
            self.similarity_matrix.get((agent_id, other_id), 0) >= threshold
        ]
        return similar_agents
        
    def _calculate_experience_similarity(self, task: Task, experience: Experience) -> float:
        """Calculate similarity between a task and an experience."""
        # Calculate capability overlap
        capability_similarity = len(
            task.required_capabilities & experience.capabilities_used
        ) / len(
            task.required_capabilities | experience.capabilities_used
        )
        
        # Calculate complexity similarity
        complexity_similarity = 1.0 - abs(
            task.estimated_complexity - experience.resource_usage.get('complexity', 0.5)
        )
        
        # Combine similarities
        return 0.7 * capability_similarity + 0.3 * complexity_similarity
        
    def _calculate_recency_weight(self, timestamp: datetime) -> float:
        """Calculate recency weight for an experience."""
        age = (datetime.now() - timestamp).total_seconds() / 3600  # Hours
        return np.exp(-age / 24)  # Exponential decay over 24 hours
        
    async def _cleanup_old_experiences(self) -> None:
        """Clean up old experiences to maintain memory bounds."""
        for agent_id in self.experiences:
            if len(self.experiences[agent_id]) > 1000:  # Max experiences per agent
                # Sort by timestamp and keep most recent
                self.experiences[agent_id].sort(key=lambda x: x.timestamp)
                self.experiences[agent_id] = self.experiences[agent_id][-1000:] 