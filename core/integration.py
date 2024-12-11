"""
Integration module for managing agent interactions and communication.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
import logging
import asyncio
from dataclasses import dataclass
from collections import defaultdict
import time
import json
from pathlib import Path

from ..agents.base_agent import BaseAgent, Message
from ..agents.neural_symbolic_agent import NeuralSymbolicAgent
from ..agents.prolog_reasoner import PrologReasoner
from ..agents.meta_reasoner import MetaReasoner
from .optimization import OptimizationConfig
from .transformations import StateTransformer

logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Configuration for agent integration."""
    communication_timeout: float = 5.0  # seconds
    max_message_size: int = 1024 * 1024  # 1MB
    batch_size: int = 10
    sync_interval: float = 1.0  # seconds
    retry_attempts: int = 3

class AgentIntegration:
    """Manages integration between different agent types."""
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.message_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.agent_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.active_syncs: Set[Tuple[str, str]] = set()
        self.sync_history: List[Dict[str, Any]] = []
        self.transformer = StateTransformer()
        
    async def sync_agents(self, 
                         source_agent: BaseAgent,
                         target_agent: BaseAgent,
                         sync_type: str = 'knowledge') -> bool:
        """Synchronize knowledge/state between agents."""
        sync_id = (source_agent.agent_id, target_agent.agent_id)
        
        if sync_id in self.active_syncs:
            logger.warning(f"Sync already in progress for {sync_id}")
            return False
            
        try:
            self.active_syncs.add(sync_id)
            
            # Get source state
            source_state = await self._get_agent_state(source_agent, sync_type)
            
            # Transform state for target
            transformed_state = self.transformer.transform(
                source_state,
                self._get_agent_format(source_agent),
                self._get_agent_format(target_agent)
            )
            
            # Update target
            success = await self._update_agent_state(
                target_agent,
                transformed_state,
                sync_type
            )
            
            # Record sync
            self._record_sync(source_agent, target_agent, sync_type, success)
            
            return success
            
        except Exception as e:
            logger.error(f"Error syncing agents: {e}")
            return False
            
        finally:
            self.active_syncs.remove(sync_id)
            
    async def route_message(self,
                          message: Message,
                          routing_strategy: str = 'direct') -> bool:
        """Route message between agents with specified strategy."""
        try:
            if routing_strategy == 'direct':
                return await self._direct_route(message)
            elif routing_strategy == 'broadcast':
                return await self._broadcast_route(message)
            elif routing_strategy == 'filtered':
                return await self._filtered_route(message)
            else:
                logger.error(f"Unknown routing strategy: {routing_strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Error routing message: {e}")
            return False
            
    async def batch_process(self,
                          source_agent: BaseAgent,
                          target_agent: BaseAgent,
                          messages: List[Message]) -> List[bool]:
        """Process multiple messages in batch."""
        results = []
        current_batch = []
        
        for message in messages:
            current_batch.append(message)
            
            if len(current_batch) >= self.config.batch_size:
                batch_results = await self._process_batch(
                    source_agent,
                    target_agent,
                    current_batch
                )
                results.extend(batch_results)
                current_batch = []
                
        # Process remaining messages
        if current_batch:
            batch_results = await self._process_batch(
                source_agent,
                target_agent,
                current_batch
            )
            results.extend(batch_results)
            
        return results
        
    async def _get_agent_state(self,
                             agent: BaseAgent,
                             state_type: str) -> Dict[str, Any]:
        """Get agent state based on type."""
        try:
            message = Message(
                id=f"get_state_{time.time()}",
                sender="integration",
                receiver=agent.agent_id,
                content={
                    'type': 'get_state',
                    'state_type': state_type
                }
            )
            
            response = await agent.process_message(message)
            return response.content if response else {}
            
        except Exception as e:
            logger.error(f"Error getting agent state: {e}")
            return {}
            
    def _get_agent_format(self, agent: BaseAgent) -> str:
        """Get agent's state format."""
        if isinstance(agent, NeuralSymbolicAgent):
            return 'neural'
        elif isinstance(agent, PrologReasoner):
            return 'symbolic'
        elif isinstance(agent, MetaReasoner):
            return 'meta'
        else:
            return 'base'
            
    async def _update_agent_state(self,
                                agent: BaseAgent,
                                state: Dict[str, Any],
                                state_type: str) -> bool:
        """Update agent state."""
        try:
            message = Message(
                id=f"state_update_{time.time()}",
                sender="integration",
                receiver=agent.agent_id,
                content={
                    'type': 'state_update',
                    'state_type': state_type,
                    'state': state
                }
            )
            
            response = await agent.process_message(message)
            return response is not None and response.content.get('success', False)
            
        except Exception as e:
            logger.error(f"Error updating agent state: {e}")
            return False
            
    async def _direct_route(self, message: Message) -> bool:
        """Direct message routing."""
        try:
            queue = self.message_queues[message.receiver]
            await queue.put(message)
            return True
        except Exception as e:
            logger.error(f"Error in direct routing: {e}")
            return False
            
    async def _broadcast_route(self, message: Message) -> bool:
        """Broadcast message to all agents."""
        success = True
        for queue in self.message_queues.values():
            try:
                await queue.put(message)
            except Exception as e:
                logger.error(f"Error in broadcast routing: {e}")
                success = False
        return success
        
    async def _filtered_route(self, message: Message) -> bool:
        """Route message based on content filters."""
        if not isinstance(message.content, dict):
            return False
            
        filters = message.content.get('filters', {})
        success = True
        
        for agent_id, queue in self.message_queues.items():
            if self._matches_filters(agent_id, filters):
                try:
                    await queue.put(message)
                except Exception as e:
                    logger.error(f"Error in filtered routing: {e}")
                    success = False
                    
        return success
        
    def _matches_filters(self, agent_id: str, filters: Dict[str, Any]) -> bool:
        """Check if agent matches message filters."""
        agent_state = self.agent_states.get(agent_id, {})
        
        for key, value in filters.items():
            if agent_state.get(key) != value:
                return False
        return True
        
    async def _process_batch(self,
                          source_agent: BaseAgent,
                          target_agent: BaseAgent,
                          batch: List[Message]) -> List[bool]:
        """Process a batch of messages."""
        results = []
        
        try:
            # Transform batch content if needed
            transformed_batch = []
            source_format = self._get_agent_format(source_agent)
            target_format = self._get_agent_format(target_agent)
            
            for message in batch:
                if isinstance(message.content, dict):
                    transformed_content = self.transformer.transform(
                        message.content,
                        source_format,
                        target_format
                    )
                else:
                    transformed_content = message.content
                    
                transformed_batch.append(Message(
                    id=message.id,
                    sender=message.sender,
                    receiver=message.receiver,
                    content=transformed_content
                ))
                
            # Prepare batch message
            batch_message = Message(
                id=f"batch_{time.time()}",
                sender=source_agent.agent_id,
                receiver=target_agent.agent_id,
                content={
                    'type': 'batch',
                    'messages': [msg.content for msg in transformed_batch]
                }
            )
            
            # Process batch
            response = await target_agent.process_message(batch_message)
            
            if response and isinstance(response.content, dict):
                results = response.content.get('results', [False] * len(batch))
            else:
                results = [False] * len(batch)
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            results = [False] * len(batch)
            
        return results
        
    def _record_sync(self,
                    source_agent: BaseAgent,
                    target_agent: BaseAgent,
                    sync_type: str,
                    success: bool):
        """Record synchronization attempt."""
        self.sync_history.append({
            'timestamp': time.time(),
            'source_id': source_agent.agent_id,
            'target_id': target_agent.agent_id,
            'source_format': self._get_agent_format(source_agent),
            'target_format': self._get_agent_format(target_agent),
            'sync_type': sync_type,
            'success': success
        }) 