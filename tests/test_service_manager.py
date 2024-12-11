"""Tests for service manager functionality."""

import pytest
import asyncio
from ..core.service_manager import (
    ServiceManager,
    ServiceConfig,
    ServiceType,
    ServiceRegistry,
    EventBus
)

@pytest.fixture
def redis_url():
    return "redis://localhost:6379"
    
@pytest.fixture
def rabbitmq_url():
    return "amqp://guest:guest@localhost:5672/"
    
@pytest.fixture
def service_config():
    return ServiceConfig(
        service_type=ServiceType.AGENT,
        host="localhost",
        port=8000,
        dependencies={ServiceType.COORDINATOR}
    )
    
@pytest.mark.asyncio
async def test_service_registry(redis_url):
    registry = ServiceRegistry(redis_url)
    await registry.connect()
    
    # Test service registration
    service_id = "test_service"
    service_type = ServiceType.AGENT
    host = "localhost"
    port = 8000
    
    await registry.register_service(service_id, service_type, host, port)
    
    # Test service retrieval
    service = await registry.get_service(service_id)
    assert service is not None
    assert service['id'] == service_id
    assert service['type'] == service_type.value
    assert service['host'] == host
    assert service['port'] == port
    
    # Test service listing
    services = await registry.get_services_by_type(service_type)
    assert len(services) == 1
    assert services[0]['id'] == service_id
    
    # Test service deregistration
    await registry.deregister_service(service_id)
    service = await registry.get_service(service_id)
    assert service is None
    
    await registry.cleanup()
    
@pytest.mark.asyncio
async def test_event_bus(rabbitmq_url):
    event_bus = EventBus(rabbitmq_url)
    await event_bus.connect()
    
    # Test event publishing and subscription
    test_data = {'message': 'test'}
    received_data = None
    
    async def handle_event(data):
        nonlocal received_data
        received_data = data
        
    await event_bus.subscribe('test_event', handle_event)
    await event_bus.publish('test_event', test_data)
    
    # Wait for message processing
    await asyncio.sleep(1)
    
    assert received_data == test_data
    
    await event_bus.cleanup()
    
@pytest.mark.asyncio
async def test_service_manager(redis_url, rabbitmq_url, service_config):
    manager = ServiceManager(redis_url, rabbitmq_url, service_config)
    
    # Test service startup
    await manager.start()
    
    # Test health check
    response = await manager.health_check(None)
    assert response.status == 200
    data = await response.json()
    assert data['status'] == 'healthy'
    assert data['service_type'] == service_config.service_type.value
    
    # Test metrics
    response = await manager.get_metrics(None)
    assert response.status == 200
    data = await response.json()
    assert 'service_id' in data
    assert 'uptime' in data
    
    # Test service discovery
    registry = ServiceRegistry(redis_url)
    await registry.connect()
    
    services = await registry.get_services_by_type(service_config.service_type)
    assert len(services) == 1
    assert services[0]['type'] == service_config.service_type.value
    
    await registry.cleanup()
    
    # Test event handling
    event_bus = EventBus(rabbitmq_url)
    await event_bus.connect()
    
    test_data = {'message': 'test'}
    received_data = None
    
    async def handle_event(data):
        nonlocal received_data
        received_data = data
        
    await event_bus.subscribe('test_event', handle_event)
    await event_bus.publish('test_event', test_data)
    
    # Wait for message processing
    await asyncio.sleep(1)
    
    assert received_data == test_data
    
    await event_bus.cleanup()
    
    # Test service shutdown
    await manager.stop()
    
    # Verify service is deregistered
    services = await registry.get_services_by_type(service_config.service_type)
    assert len(services) == 0
    
@pytest.mark.asyncio
async def test_service_dependencies(redis_url, rabbitmq_url):
    # Create coordinator service
    coordinator_config = ServiceConfig(
        service_type=ServiceType.COORDINATOR,
        host="localhost",
        port=8001
    )
    coordinator = ServiceManager(redis_url, rabbitmq_url, coordinator_config)
    await coordinator.start()
    
    # Create agent service with dependency on coordinator
    agent_config = ServiceConfig(
        service_type=ServiceType.AGENT,
        host="localhost",
        port=8002,
        dependencies={ServiceType.COORDINATOR}
    )
    agent = ServiceManager(redis_url, rabbitmq_url, agent_config)
    await agent.start()
    
    # Verify services are registered
    registry = ServiceRegistry(redis_url)
    await registry.connect()
    
    coordinators = await registry.get_services_by_type(ServiceType.COORDINATOR)
    assert len(coordinators) == 1
    
    agents = await registry.get_services_by_type(ServiceType.AGENT)
    assert len(agents) == 1
    
    # Test dependency validation
    agent_service = agents[0]
    coordinator_service = coordinators[0]
    
    assert ServiceType.COORDINATOR.value in str(agent_config.dependencies)
    
    # Cleanup
    await coordinator.stop()
    await agent.stop()
    await registry.cleanup()
    
@pytest.mark.asyncio
async def test_service_recovery(redis_url, rabbitmq_url, service_config):
    manager = ServiceManager(redis_url, rabbitmq_url, service_config)
    await manager.start()
    
    # Simulate service crash
    await manager.service_registry.redis.close()
    
    # Wait for heartbeat to attempt reconnection
    await asyncio.sleep(65)  # Wait for heartbeat + 5 seconds
    
    # Verify service is still registered
    registry = ServiceRegistry(redis_url)
    await registry.connect()
    
    services = await registry.get_services_by_type(service_config.service_type)
    assert len(services) == 1
    
    await manager.stop()
    await registry.cleanup() 