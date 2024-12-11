from typing import Dict, Any, List, Optional, Set
from loguru import logger
from .framework import HotPluggableAgentFramework
from .coordinator import Coordinator
from ..agents.base_agent import BaseAgent, AgentCapability
from ..utils.visualizer import SystemVisualizer
import asyncio
import time

class AgentOrchestrator:
    """Orchestrates agent lifecycle, scaling, and interactions."""
    
    def __init__(self, visualization_enabled: bool = True):
        self.framework = HotPluggableAgentFramework()
        self.coordinator = Coordinator()
        self.scaling_policies: Dict[str, Dict[str, Any]] = {}
        self.health_checks: Dict[str, Dict[str, Any]] = {}
        self.agent_pools: Dict[str, List[BaseAgent]] = {}
        self.visualization_enabled = visualization_enabled
        if visualization_enabled:
            self.visualizer = SystemVisualizer()
        
    async def start(self):
        """Start the orchestrator and its components."""
        logger.info("Starting orchestrator")
        asyncio.create_task(self._monitor_system())
        asyncio.create_task(self._perform_health_checks())
        if self.visualization_enabled:
            asyncio.create_task(self._update_visualizations())
        
    async def stop(self):
        """Stop the orchestrator and all agents."""
        logger.info("Stopping orchestrator")
        for agents in self.agent_pools.values():
            for agent in agents:
                await self.framework.stop_agent(agent.name)
        if self.visualization_enabled:
            self.visualizer.generate_report(
                self.framework.get_system_state(),
                self.agent_pools
            )
                
    async def register_agent(self, agent: BaseAgent, pool_name: str = "default"):
        """Register an agent with scaling and health monitoring."""
        try:
            # Register with framework and coordinator
            self.framework.register_agent(agent)
            self.coordinator.register_agent(agent)
            
            # Add to agent pool
            if pool_name not in self.agent_pools:
                self.agent_pools[pool_name] = []
            self.agent_pools[pool_name].append(agent)
            
            # Set up default scaling policy
            self.scaling_policies[agent.name] = {
                "min_instances": 1,
                "max_instances": 3,
                "scale_up_threshold": 0.8,  # CPU utilization
                "scale_down_threshold": 0.2,
                "cooldown_period": 300  # seconds
            }
            
            # Set up health checks
            self.health_checks[agent.name] = {
                "last_check": time.time(),
                "check_interval": 60,  # seconds
                "failures": 0,
                "max_failures": 3
            }
            
            # Start the agent
            await self.framework.start_agent(agent.name)
            logger.info(f"Registered and started agent {agent.name} in pool {pool_name}")
            
            # Update visualizations
            if self.visualization_enabled:
                for other_agent in self.framework.agents.values():
                    if other_agent.name != agent.name:
                        self.visualizer.update_interaction_graph(
                            agent.name,
                            other_agent.name,
                            "potential_interaction"
                        )
            
        except Exception as e:
            logger.error(f"Error registering agent {agent.name}: {str(e)}")
            raise
            
    async def unregister_agent(self, agent_name: str):
        """Unregister an agent and clean up its resources."""
        try:
            # Stop the agent
            await self.framework.stop_agent(agent_name)
            
            # Remove from framework and coordinator
            self.framework.unregister_agent(agent_name)
            self.coordinator.unregister_agent(agent_name)
            
            # Remove from agent pool
            for pool in self.agent_pools.values():
                pool[:] = [a for a in pool if a.name != agent_name]
                
            # Clean up policies and checks
            self.scaling_policies.pop(agent_name, None)
            self.health_checks.pop(agent_name, None)
            
            logger.info(f"Unregistered agent {agent_name}")
            
        except Exception as e:
            logger.error(f"Error unregistering agent {agent_name}: {str(e)}")
            raise
            
    async def scale_agent_pool(self, pool_name: str, scale_factor: int):
        """Scale the number of agents in a pool."""
        if pool_name not in self.agent_pools:
            logger.warning(f"Agent pool {pool_name} not found")
            return
            
        try:
            pool = self.agent_pools[pool_name]
            if not pool:
                return
                
            # Create new agent instances based on the template
            template_agent = pool[0]
            current_count = len(pool)
            target_count = max(1, current_count + scale_factor)
            
            if scale_factor > 0:
                # Scale up
                for i in range(scale_factor):
                    new_agent = type(template_agent)(
                        agent_id=None,  # Will generate new ID
                        name=f"{template_agent.name}_{current_count + i + 1}",
                        capabilities=template_agent.get_capabilities()
                    )
                    await self.register_agent(new_agent, pool_name)
                    
            elif scale_factor < 0:
                # Scale down
                agents_to_remove = pool[target_count:]
                for agent in agents_to_remove:
                    await self.unregister_agent(agent.name)
                    
            logger.info(f"Scaled pool {pool_name} by factor {scale_factor}")
            
        except Exception as e:
            logger.error(f"Error scaling agent pool {pool_name}: {str(e)}")
            raise
            
    async def update_agent_capability(self, agent_name: str, capability: AgentCapability):
        """Update or add a capability to an agent."""
        try:
            agent = None
            for pool in self.agent_pools.values():
                for a in pool:
                    if a.name == agent_name:
                        agent = a
                        break
                if agent:
                    break
                    
            if not agent:
                logger.warning(f"Agent {agent_name} not found")
                return
                
            # Update capability
            agent.add_capability(capability)
            
            # Update framework's capability graph
            self.framework.logic_engine.update_capability_graph(agent)
            
            logger.info(f"Updated capability {capability.name} for agent {agent_name}")
            
        except Exception as e:
            logger.error(f"Error updating agent capability: {str(e)}")
            raise
            
    async def _monitor_system(self):
        """Monitor system performance and scale as needed."""
        while True:
            try:
                for pool_name, agents in self.agent_pools.items():
                    if not agents:
                        continue
                        
                    # Calculate pool metrics
                    total_load = sum(
                        agent.performance_metrics.avg_response_time
                        for agent in agents
                    ) / len(agents)
                    
                    # Check scaling policies
                    template_agent = agents[0]
                    policy = self.scaling_policies[template_agent.name]
                    
                    if total_load > policy["scale_up_threshold"]:
                        if len(agents) < policy["max_instances"]:
                            await self.scale_agent_pool(pool_name, 1)
                            
                    elif total_load < policy["scale_down_threshold"]:
                        if len(agents) > policy["min_instances"]:
                            await self.scale_agent_pool(pool_name, -1)
                            
            except Exception as e:
                logger.error(f"Error in system monitoring: {str(e)}")
                
            await asyncio.sleep(60)  # Check every minute
            
    async def _perform_health_checks(self):
        """Perform health checks on all agents."""
        while True:
            try:
                current_time = time.time()
                
                for agent_name, check in self.health_checks.items():
                    if current_time - check["last_check"] < check["check_interval"]:
                        continue
                        
                    # Find the agent
                    agent = None
                    for pool in self.agent_pools.values():
                        for a in pool:
                            if a.name == agent_name:
                                agent = a
                                break
                        if agent:
                            break
                            
                    if not agent:
                        continue
                        
                    # Check agent health
                    metrics = agent.get_performance_metrics()
                    if metrics["error_rate"] > 0.5:  # More than 50% errors
                        check["failures"] += 1
                    else:
                        check["failures"] = max(0, check["failures"] - 1)
                        
                    # Take action if needed
                    if check["failures"] >= check["max_failures"]:
                        logger.warning(f"Agent {agent_name} failing health checks, restarting")
                        await self.framework.stop_agent(agent_name)
                        await self.framework.start_agent(agent_name)
                        check["failures"] = 0
                        
                    check["last_check"] = current_time
                    
            except Exception as e:
                logger.error(f"Error in health checks: {str(e)}")
                
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def _update_visualizations(self):
        """Update system visualizations periodically."""
        if not self.visualization_enabled:
            return
            
        while True:
            try:
                # Update system metrics
                system_state = self.framework.get_system_state()
                self.visualizer.update_metrics(system_state)
                
                # Update interaction graph based on recent communications
                for agent in self.framework.agents.values():
                    for message in agent.message_queue._queue:
                        self.visualizer.update_interaction_graph(
                            message.sender,
                            message.receiver,
                            message.message_type
                        )
                
                # Generate intermediate report
                self.visualizer.generate_report(system_state, self.agent_pools)
                
            except Exception as e:
                logger.error(f"Error updating visualizations: {str(e)}")
                
            await asyncio.sleep(10)  # Update every 10 seconds