from typing import Dict, Any, List, Optional, Set
from .base_agent import BaseAgent, AgentCapability, Message
from loguru import logger
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import asyncio

@dataclass
class Experience:
    """Represents a learning experience."""
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    done: bool

class CapabilityNetwork(nn.Module):
    """Neural network for capability learning."""
    
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MetaLearner:
    """Handles meta-learning for capability enhancement."""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = CapabilityNetwork(state_size, action_size).to(self.device)
        self.target_model = CapabilityNetwork(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def remember(self, experience: Experience):
        """Store experience in memory."""
        self.memory.append(experience)
        
    def act(self, state: Dict[str, Any]) -> str:
        """Choose an action based on state."""
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
            
        state_tensor = torch.FloatTensor(self._encode_state(state)).to(self.device)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return torch.argmax(action_values).item()
        
    def replay(self, batch_size: int):
        """Train on past experiences."""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([self._encode_state(e.state) for e in minibatch]).to(self.device)
        actions = torch.LongTensor([e.action for e in minibatch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in minibatch]).to(self.device)
        next_states = torch.FloatTensor([self._encode_state(e.next_state) for e in minibatch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in minibatch]).to(self.device)
        
        # Get Q values for current states
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Get Q values for next states
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss and update
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_model(self):
        """Update target network."""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def _encode_state(self, state: Dict[str, Any]) -> List[float]:
        """Encode state dictionary into vector."""
        encoded = []
        for key, value in sorted(state.items()):
            if isinstance(value, (int, float)):
                encoded.append(float(value))
            elif isinstance(value, bool):
                encoded.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # Simple hash-based encoding
                encoded.append(float(hash(value)) % 100)
        return encoded

class CapabilityEvolution:
    """Handles capability evolution through genetic algorithms."""
    
    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.population: List[AgentCapability] = []
        self.fitness_scores: List[float] = []
        
    def initialize_population(self, template: AgentCapability):
        """Initialize population with variations of template capability."""
        self.population = [self._mutate_capability(template) for _ in range(self.population_size)]
        
    def evolve(self, generations: int = 10):
        """Evolve capabilities over generations."""
        for _ in range(generations):
            # Select parents based on fitness
            parents = self._select_parents()
            
            # Create new population through crossover and mutation
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self._crossover(parent1, parent2)
                child = self._mutate_capability(child)
                new_population.append(child)
                
            self.population = new_population
            
    def _select_parents(self) -> List[AgentCapability]:
        """Select parents using tournament selection."""
        tournament_size = 3
        parents = []
        for _ in range(self.population_size // 2):
            tournament = random.sample(list(enumerate(self.population)), tournament_size)
            winner = max(tournament, key=lambda x: self.fitness_scores[x[0]])
            parents.append(winner[1])
        return parents
        
    def _crossover(self, parent1: AgentCapability, parent2: AgentCapability) -> AgentCapability:
        """Create new capability by combining parents."""
        return AgentCapability(
            name=f"{parent1.name}_{parent2.name}",
            description=f"Hybrid of {parent1.name} and {parent2.name}",
            input_types=parent1.input_types.union(parent2.input_types),
            output_types=parent1.output_types.union(parent2.output_types),
            performance_metrics={}
        )
        
    def _mutate_capability(self, capability: AgentCapability) -> AgentCapability:
        """Randomly modify capability attributes."""
        return AgentCapability(
            name=capability.name + "_mutated",
            description=capability.description,
            input_types=set(list(capability.input_types)[:random.randint(1, len(capability.input_types))]),
            output_types=set(list(capability.output_types)[:random.randint(1, len(capability.output_types))]),
            performance_metrics={}
        )

class LearnerAgent(BaseAgent):
    """Agent specialized in learning and evolving capabilities."""
    
    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id=agent_id, name="LearnerAgent")
        self.meta_learner = MetaLearner(state_size=10, action_size=5)
        self.capability_evolution = CapabilityEvolution()
        self.learning_tasks: asyncio.Queue = asyncio.Queue()
        self.evolution_tasks: asyncio.Queue = asyncio.Queue()
        
    async def process_message(self, message: Message):
        """Process incoming messages."""
        if message.message_type == "learn_capability":
            await self.learn_capability(message.content, message.sender)
        elif message.message_type == "evolve_capabilities":
            await self.evolve_capabilities(message.content, message.sender)
        else:
            logger.warning(f"Unknown message type received: {message.message_type}")
            
    async def handle_task(self, task: Dict[str, Any]):
        """Handle specific learning tasks."""
        task_type = task.get("type")
        if task_type == "learn":
            await self.learning_tasks.put(task)
        elif task_type == "evolve":
            await self.evolution_tasks.put(task)
            
    async def learn_capability(self, content: Dict[str, Any], requester: str):
        """Learn a new capability or improve existing one."""
        try:
            state = content.get("state", {})
            action = content.get("action", "")
            reward = content.get("reward", 0.0)
            next_state = content.get("next_state", {})
            done = content.get("done", False)
            
            # Store experience
            experience = Experience(state, action, reward, next_state, done)
            self.meta_learner.remember(experience)
            
            # Train on batch
            self.meta_learner.replay(32)
            
            if done:
                self.meta_learner.update_target_model()
                
            await self.send_message(
                receiver=requester,
                content={"status": "Learning completed"},
                message_type="learning_complete"
            )
            
        except Exception as e:
            logger.error(f"Error learning capability: {str(e)}")
            await self.send_message(
                receiver=requester,
                content={"error": str(e)},
                message_type="learning_error"
            )
            
    async def evolve_capabilities(self, content: Dict[str, Any], requester: str):
        """Evolve capabilities through genetic algorithm."""
        try:
            template = content.get("template")
            if not template:
                raise ValueError("Template capability required for evolution")
                
            # Initialize population
            self.capability_evolution.initialize_population(template)
            
            # Set fitness scores based on performance
            self.capability_evolution.fitness_scores = [
                cap.performance_metrics.get("success_rate", 0.0)
                for cap in self.capability_evolution.population
            ]
            
            # Evolve capabilities
            self.capability_evolution.evolve()
            
            # Return best capability
            best_idx = max(range(len(self.capability_evolution.population)),
                          key=lambda i: self.capability_evolution.fitness_scores[i])
            best_capability = self.capability_evolution.population[best_idx]
            
            await self.send_message(
                receiver=requester,
                content={"evolved_capability": vars(best_capability)},
                message_type="evolution_complete"
            )
            
        except Exception as e:
            logger.error(f"Error evolving capabilities: {str(e)}")
            await self.send_message(
                receiver=requester,
                content={"error": str(e)},
                message_type="evolution_error"
            )
            
    async def _process_learning_queue(self):
        """Process queued learning tasks."""
        while True:
            try:
                task = await self.learning_tasks.get()
                await self.learn_capability(task, task.get("requester"))
                self.learning_tasks.task_done()
            except Exception as e:
                logger.error(f"Error processing learning task: {str(e)}")
            await asyncio.sleep(0.1)
            
    async def _process_evolution_queue(self):
        """Process queued evolution tasks."""
        while True:
            try:
                task = await self.evolution_tasks.get()
                await self.evolve_capabilities(task, task.get("requester"))
                self.evolution_tasks.task_done()
            except Exception as e:
                logger.error(f"Error processing evolution task: {str(e)}")
            await asyncio.sleep(0.1) 