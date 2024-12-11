"""Load balancer for service orchestration."""

import asyncio
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ServiceHealth:
    """Health status of a service."""
    is_healthy: bool
    last_check: datetime
    error_count: int
    response_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0

@dataclass
class ServiceStats:
    """Statistics for a service."""
    request_count: int = 0
    error_count: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    last_used: datetime = None
    avg_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0

class LoadBalancer:
    """Load balancer for distributing work across services."""
    
    def __init__(self):
        self.services: Dict[str, str] = {}  # service_id -> service_type
        self.service_loads: Dict[str, int] = {}  # service_id -> current_load
        self.max_loads: Dict[str, int] = {}  # service_id -> max_load
        self.service_stats: Dict[str, ServiceStats] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
        self.circuit_open: Set[str] = set()  # Circuit breaker pattern
        self.response_times: Dict[str, List[float]] = defaultdict(list)
        
    def register_service(
        self,
        service_id: str,
        service_type: str,
        max_load: int
    ):
        """Register a new service."""
        self.services[service_id] = service_type
        self.service_loads[service_id] = 0
        self.max_loads[service_id] = max_load
        self.service_stats[service_id] = ServiceStats(
            last_used=datetime.now()
        )
        self.service_health[service_id] = ServiceHealth(
            is_healthy=True,
            last_check=datetime.now(),
            error_count=0
        )
        logger.info(f"Registered service {service_id} of type {service_type}")
        
    def deregister_service(self, service_id: str):
        """Deregister a service."""
        self.services.pop(service_id, None)
        self.service_loads.pop(service_id, None)
        self.max_loads.pop(service_id, None)
        self.service_stats.pop(service_id, None)
        self.service_health.pop(service_id, None)
        self.circuit_open.discard(service_id)
        logger.info(f"Deregistered service {service_id}")
        
    async def get_service(
        self,
        service_type: str,
        strategy: str = "least_loaded"
    ) -> Optional[str]:
        """Get the best service based on the strategy."""
        available_services = [
            sid for sid, stype in self.services.items()
            if (stype == service_type and
                sid not in self.circuit_open and
                self.service_health[sid].is_healthy)
        ]
        
        if not available_services:
            return None
            
        if strategy == "round_robin":
            return self._round_robin_select(available_services)
        elif strategy == "least_loaded":
            return self._least_loaded_select(available_services)
        elif strategy == "fastest_response":
            return self._fastest_response_select(available_services)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
    def _round_robin_select(self, services: List[str]) -> str:
        """Select service using round-robin."""
        # Sort by last used time
        services.sort(
            key=lambda x: self.service_stats[x].last_used
        )
        return services[0]
        
    def _least_loaded_select(self, services: List[str]) -> str:
        """Select least loaded service."""
        return min(
            services,
            key=lambda x: self.service_loads[x] / self.max_loads[x]
        )
        
    def _fastest_response_select(self, services: List[str]) -> str:
        """Select service with fastest response time."""
        return min(
            services,
            key=lambda x: (
                statistics.mean(self.response_times[x])
                if self.response_times[x]
                else float('inf')
            )
        )
        
    async def update_service_stats(
        self,
        service_id: str,
        stats: ServiceStats
    ):
        """Update service statistics."""
        if service_id in self.service_stats:
            self.service_stats[service_id] = stats
            
            # Update response times history
            if len(self.response_times[service_id]) > 100:
                self.response_times[service_id] = self.response_times[service_id][-100:]
            if stats.max_response_time > 0:
                self.response_times[service_id].append(stats.max_response_time)
                
            # Check circuit breaker
            error_rate = stats.error_count / max(stats.request_count, 1)
            if error_rate > 0.5:  # 50% error rate threshold
                self.circuit_open.add(service_id)
                logger.warning(f"Circuit breaker opened for service {service_id}")
            elif service_id in self.circuit_open:
                # Check if we should close the circuit
                if error_rate < 0.1:  # 10% error rate threshold for recovery
                    self.circuit_open.discard(service_id)
                    logger.info(f"Circuit breaker closed for service {service_id}")
                    
    async def update_service_health(
        self,
        service_id: str,
        health: ServiceHealth
    ):
        """Update service health status."""
        if service_id in self.service_health:
            self.service_health[service_id] = health
            
            if not health.is_healthy:
                logger.warning(
                    f"Service {service_id} health check failed: "
                    f"error_count={health.error_count}"
                )
                
    def update_load(self, service_id: str, load_change: int):
        """Update service load."""
        if service_id in self.service_loads:
            self.service_loads[service_id] = max(
                0,
                min(
                    self.service_loads[service_id] + load_change,
                    self.max_loads[service_id]
                )
            )
            
    def get_metrics(self) -> Dict:
        """Get current metrics for all services."""
        return {
            'services': {
                service_id: {
                    'type': self.services[service_id],
                    'load': self.service_loads[service_id],
                    'max_load': self.max_loads[service_id],
                    'health': self.service_health[service_id],
                    'stats': self.service_stats[service_id],
                    'circuit_status': (
                        'open' if service_id in self.circuit_open
                        else 'closed'
                    )
                }
                for service_id in self.services
            },
            'total_services': len(self.services),
            'healthy_services': sum(
                1 for h in self.service_health.values()
                if h.is_healthy
            ),
            'circuits_open': len(self.circuit_open)
        } 