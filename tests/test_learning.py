import asyncio
import pytest
from ..agents.learner_agent import LearnerAgent, AgentCapability
from ..core.orchestrator import AgentOrchestrator
import numpy as np
from typing import Dict, Any

@pytest.mark.asyncio
async def test_capability_learning():
    """Test the learning agent's capability to improve through experience."""
    orchestrator = AgentOrchestrator()
    learner = LearnerAgent()
    
    # Register learner agent
    await orchestrator.register_agent(learner, "learning_pool")
    
    # Create a simple capability template
    template_capability = AgentCapability(
        name="code_analysis",
        description="Analyzes code structure",
        input_types={"source_code"},
        output_types={"analysis_result"},
        performance_metrics={"success_rate": 0.5}
    )
    
    # Generate synthetic learning experiences
    states = []
    actions = []
    rewards = []
    next_states = []
    
    for _ in range(100):
        # Simulate code analysis scenarios
        state = {
            "code_complexity": np.random.uniform(0, 10),
            "num_functions": np.random.randint(1, 20),
            "cyclomatic_complexity": np.random.uniform(1, 15)
        }
        
        action = np.random.randint(0, 5)  # Different analysis strategies
        
        # Simulate reward based on action appropriateness
        reward = calculate_reward(state, action)
        
        next_state = {
            "code_complexity": state["code_complexity"] + np.random.normal(0, 0.1),
            "num_functions": state["num_functions"],
            "cyclomatic_complexity": state["cyclomatic_complexity"] + np.random.normal(0, 0.2)
        }
        
        # Store experience
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        
        # Send learning experience to agent
        await orchestrator.framework.execute_capability_chain(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": False
            },
            "state",
            "learning_result"
        )
    
    # Test capability evolution
    evolution_task = {
        "type": "evolve",
        "template": template_capability,
        "requester": "test"
    }
    
    await learner.handle_task(evolution_task)
    
    # Verify learning progress
    final_performance = np.mean(rewards[-10:])  # Average of last 10 rewards
    initial_performance = np.mean(rewards[:10])  # Average of first 10 rewards
    
    assert final_performance > initial_performance, "Learning should improve performance"
    
def calculate_reward(state: Dict[str, Any], action: int) -> float:
    """Calculate reward based on state-action pair."""
    reward = 0.0
    
    # Action 0: Basic analysis
    if action == 0:
        if state["code_complexity"] < 5:
            reward += 1.0
        else:
            reward -= 0.5
            
    # Action 1: Deep analysis
    elif action == 1:
        if state["code_complexity"] > 5:
            reward += 1.5
        else:
            reward -= 1.0
            
    # Action 2: Function analysis
    elif action == 2:
        if state["num_functions"] > 10:
            reward += 1.2
        else:
            reward -= 0.3
            
    # Action 3: Complexity analysis
    elif action == 3:
        if state["cyclomatic_complexity"] > 10:
            reward += 1.3
        else:
            reward -= 0.4
            
    # Action 4: Combined analysis
    elif action == 4:
        avg_complexity = (state["code_complexity"] + state["cyclomatic_complexity"]) / 2
        if avg_complexity > 7 and state["num_functions"] > 5:
            reward += 2.0
        else:
            reward -= 0.8
            
    return reward

@pytest.mark.asyncio
async def test_capability_evolution():
    """Test the evolution of capabilities through genetic algorithm."""
    orchestrator = AgentOrchestrator()
    learner = LearnerAgent()
    
    # Register learner agent
    await orchestrator.register_agent(learner, "learning_pool")
    
    # Create initial capabilities
    capabilities = [
        AgentCapability(
            name=f"capability_{i}",
            description=f"Test capability {i}",
            input_types={"type_a", "type_b"},
            output_types={"result"},
            performance_metrics={"success_rate": np.random.uniform(0.3, 0.7)}
        )
        for i in range(5)
    ]
    
    # Evolve capabilities
    for capability in capabilities:
        evolution_task = {
            "type": "evolve",
            "template": capability,
            "requester": "test"
        }
        await learner.handle_task(evolution_task)
        
    # Verify evolution results
    evolved_capabilities = learner.capability_evolution.population
    assert len(evolved_capabilities) > 0, "Should have evolved capabilities"
    
    # Check for improved performance
    original_performance = np.mean([cap.performance_metrics["success_rate"] for cap in capabilities])
    evolved_performance = np.mean([
        cap.performance_metrics.get("success_rate", 0.0)
        for cap in evolved_capabilities
    ])
    
    assert evolved_performance >= original_performance, "Evolution should improve or maintain performance"

@pytest.mark.asyncio
async def test_meta_learning():
    """Test meta-learning capabilities."""
    orchestrator = AgentOrchestrator()
    learner = LearnerAgent()
    
    # Register learner agent
    await orchestrator.register_agent(learner, "learning_pool")
    
    # Train on different capability types
    capability_types = ["analysis", "generation", "transformation"]
    
    for cap_type in capability_types:
        # Generate synthetic training data
        for _ in range(50):
            state = {
                "capability_type": cap_type,
                "complexity": np.random.uniform(0, 10),
                "success_rate": np.random.uniform(0, 1)
            }
            
            action = learner.meta_learner.act(state)
            
            # Simulate environment response
            reward = 0.0
            if cap_type == "analysis":
                reward = 1.0 if action in [0, 3] else -0.5
            elif cap_type == "generation":
                reward = 1.0 if action in [1, 4] else -0.5
            else:  # transformation
                reward = 1.0 if action in [2, 4] else -0.5
                
            next_state = {
                "capability_type": cap_type,
                "complexity": state["complexity"] + np.random.normal(0, 0.1),
                "success_rate": min(1.0, state["success_rate"] + reward * 0.1)
            }
            
            # Learn from experience
            await learner.learn_capability(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": False
                },
                "test"
            )
    
    # Test learned behavior
    test_states = [
        {"capability_type": "analysis", "complexity": 5.0, "success_rate": 0.5},
        {"capability_type": "generation", "complexity": 5.0, "success_rate": 0.5},
        {"capability_type": "transformation", "complexity": 5.0, "success_rate": 0.5}
    ]
    
    actions = [learner.meta_learner.act(state) for state in test_states]
    
    # Verify that different states lead to different actions
    assert len(set(actions)) > 1, "Meta-learner should learn different strategies for different capabilities"

if __name__ == "__main__":
    pytest.main([__file__]) 