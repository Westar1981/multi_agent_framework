"""Service manager for microservices architecture."""

import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, asdict
from aiohttp import web
import aioredis
import aiormq
from prometheus_client import Counter, Histogram, Gauge
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from .load_balancer import LoadBalancer, ServiceHealth, ServiceStats
from ..utils.visualizer import SystemVisualizer

logger = logging.getLogger(__name__)

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(ConsoleSpanExporter())
)

# Instrument libraries
AioHttpClientInstrumentor().instrument()
RedisInstrumentor().instrument()
RequestsInstrumentor().instrument()

# Prometheus metrics
REQUEST_COUNT = Counter(
    'service_request_total',
    'Total number of requests by service',
    ['service_id', 'service_type']
)
RESPONSE_TIME = Histogram(
    'service_response_time_seconds',
    'Response time in seconds',
    ['service_id', 'service_type']
)
SERVICE_HEALTH = Gauge(
    'service_health',
    'Service health status (1=healthy, 0=unhealthy)',
    ['service_id', 'service_type']
)

@dataclass
class ServiceConfig:
    """Configuration for a service."""
    service_id: str
    service_type: str
    host: str
    port: int
    dependencies: Set[str]
    max_load: int = 100
    health_check_interval: int = 30
    circuit_breaker_threshold: float = 0.5
    retry_timeout: int = 60
    cache_ttl: int = 300

@dataclass
class ServiceRegistration:
    """Service registration information."""
    config: ServiceConfig
    status: str = "starting"
    last_heartbeat: float = 0.0
    metadata: Dict[str, Any] = None

class ServiceManager:
    """Manages microservices and event-driven communication."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.services: Dict[str, ServiceRegistration] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.load_balancer = LoadBalancer()
        self.visualizer = SystemVisualizer()
        self.redis: Optional[aioredis.Redis] = None
        self.rmq_connection: Optional[aiormq.Connection] = None
        self.rmq_channel: Optional[aiormq.Channel] = None
        self._shutdown = False
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self._shutdown = True
        
    async def _connect_redis(self):
        """Connect to Redis for service discovery and caching."""
        with tracer.start_as_current_span("connect_redis") as span:
            self.redis = await aioredis.create_redis_pool(
                'redis://localhost',
                encoding='utf-8'
            )
            span.set_attribute("redis.connection", "localhost")
        
    async def _connect_rabbitmq(self):
        """Connect to RabbitMQ for event-driven communication."""
        with tracer.start_as_current_span("connect_rabbitmq") as span:
            self.rmq_connection = await aiormq.connect("amqp://guest:guest@localhost/")
            self.rmq_channel = await self.rmq_connection.channel()
            
            # Declare exchanges
            await self.rmq_channel.exchange_declare(
                exchange='agent_events',
                exchange_type='topic'
            )
            await self.rmq_channel.exchange_declare(
                exchange='system_events',
                exchange_type='fanout'
            )
            span.set_attribute("rabbitmq.connection", "localhost")
        
    async def start(self):
        """Start the service manager."""
        with tracer.start_as_current_span("service_manager_start") as span:
            logger.info("Starting service manager...")
            span.set_attribute("event", "start")
            
            # Load configuration
            with open(self.config_path) as f:
                config = json.load(f)
                
            # Initialize connections
            await self._connect_redis()
            await self._connect_rabbitmq()
            
            # Start HTTP server for service registration
            app = web.Application()
            app.router.add_post('/register', self.handle_registration)
            app.router.add_post('/heartbeat', self.handle_heartbeat)
            app.router.add_get('/services', self.handle_service_list)
            app.router.add_post('/event', self.handle_event)
            
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', 8080)
            await site.start()
            
            # Start background tasks
            asyncio.create_task(self._health_check_loop())
            asyncio.create_task(self._metrics_collection_loop())
            
            logger.info("Service manager started")
            span.set_attribute("status", "running")
            
            while not self._shutdown:
                await asyncio.sleep(1)
                
            # Graceful shutdown
            await self.shutdown()
        
    async def shutdown(self):
        """Shutdown the service manager."""
        with tracer.start_as_current_span("service_manager_shutdown") as span:
            logger.info("Shutting down service manager...")
            span.set_attribute("event", "shutdown")
            
            # Close connections
            if self.redis:
                self.redis.close()
                await self.redis.wait_closed()
                
            if self.rmq_channel:
                await self.rmq_channel.close()
            if self.rmq_connection:
                await self.rmq_connection.close()
                
            # Update service statuses
            for service in self.services.values():
                service.status = "stopped"
                
            logger.info("Service manager shutdown complete")
            span.set_attribute("status", "stopped")
        
    async def register_service(self, config: ServiceConfig) -> str:
        """Register a new service."""
        with tracer.start_as_current_span("register_service") as span:
            service_id = str(uuid.uuid4())
            span.set_attribute("service_id", service_id)
            span.set_attribute("service_type", config.service_type)
            
            registration = ServiceRegistration(
                config=config,
                status="starting",
                last_heartbeat=time.time(),
                metadata={}
            )
            
            self.services[service_id] = registration
            
            # Register with load balancer
            self.load_balancer.register_service(
                service_id,
                config.service_type,
                config.max_load
            )
            
            # Store in Redis for service discovery
            await self.redis.hset(
                f"service:{service_id}",
                mapping=asdict(registration)
            )
            
            # Initialize metrics
            REQUEST_COUNT.labels(service_id=service_id, service_type=config.service_type)
            RESPONSE_TIME.labels(service_id=service_id, service_type=config.service_type)
            SERVICE_HEALTH.labels(
                service_id=service_id,
                service_type=config.service_type
            ).set(1)
            
            logger.info(f"Registered service {service_id} of type {config.service_type}")
            return service_id
        
    async def handle_registration(self, request: web.Request) -> web.Response:
        """Handle service registration requests."""
        with tracer.start_as_current_span("handle_registration") as span:
            data = await request.json()
            config = ServiceConfig(**data)
            service_id = await self.register_service(config)
            span.set_attribute("service_id", service_id)
            return web.json_response({"service_id": service_id})
        
    async def handle_heartbeat(self, request: web.Request) -> web.Response:
        """Handle service heartbeat requests."""
        with tracer.start_as_current_span("handle_heartbeat") as span:
            data = await request.json()
            service_id = data["service_id"]
            span.set_attribute("service_id", service_id)
            
            if service_id in self.services:
                self.services[service_id].last_heartbeat = time.time()
                self.services[service_id].status = "running"
                
                # Update metrics
                if "metrics" in data:
                    await self.load_balancer.update_service_stats(
                        service_id,
                        ServiceStats(**data["metrics"])
                    )
                    
                    # Update Prometheus metrics
                    SERVICE_HEALTH.labels(
                        service_id=service_id,
                        service_type=self.services[service_id].config.service_type
                    ).set(1)
                    
                return web.json_response({"status": "ok"})
            return web.json_response(
                {"status": "error", "message": "Service not found"},
                status=404
            )
        
    async def handle_service_list(self, request: web.Request) -> web.Response:
        """Handle service list requests."""
        with tracer.start_as_current_span("handle_service_list") as span:
            services = {
                id: asdict(reg) for id, reg in self.services.items()
                if reg.status == "running"
            }
            span.set_attribute("service_count", len(services))
            return web.json_response(services)
        
    async def handle_event(self, request: web.Request) -> web.Response:
        """Handle event publishing requests."""
        with tracer.start_as_current_span("handle_event") as span:
            data = await request.json()
            event_type = data["type"]
            payload = data["payload"]
            span.set_attribute("event_type", event_type)
            
            # Publish to RabbitMQ
            await self.rmq_channel.basic_publish(
                exchange='agent_events',
                routing_key=event_type,
                body=json.dumps(payload).encode()
            )
            
            return web.json_response({"status": "ok"})
        
    async def _health_check_loop(self):
        """Periodic health check of services."""
        while not self._shutdown:
            with tracer.start_as_current_span("health_check_loop"):
                current_time = time.time()
                
                for service_id, registration in list(self.services.items()):
                    if (current_time - registration.last_heartbeat >
                        registration.config.health_check_interval):
                        # Service might be down
                        if registration.status == "running":
                            registration.status = "unhealthy"
                            logger.warning(f"Service {service_id} is unhealthy")
                            
                            # Update load balancer
                            await self.load_balancer.update_service_health(
                                service_id,
                                ServiceHealth(
                                    is_healthy=False,
                                    last_check=datetime.now(),
                                    error_count=1
                                )
                            )
                            
                            # Update Prometheus metrics
                            SERVICE_HEALTH.labels(
                                service_id=service_id,
                                service_type=registration.config.service_type
                            ).set(0)
                            
                await asyncio.sleep(10)
        
    async def _metrics_collection_loop(self):
        """Periodic collection and aggregation of metrics."""
        while not self._shutdown:
            with tracer.start_as_current_span("metrics_collection_loop"):
                metrics = self.load_balancer.get_metrics()
                
                # Update visualization
                self.visualizer.update_metrics(metrics)
                
                # Store in Redis for historical analysis
                await self.redis.zadd(
                    "metrics_history",
                    time.time(),
                    json.dumps(metrics)
                )
                
                # Cleanup old metrics
                await self.redis.zremrangebyscore(
                    "metrics_history",
                    0,
                    time.time() - 86400  # Keep last 24 hours
                )
                
                await asyncio.sleep(60)  # Collect metrics every minute