"""Tests for telemetry manager."""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.resources import Resource

from ..core.telemetry_manager import TelemetryManager, create_telemetry_manager

@pytest.fixture
def config_file(tmp_path):
    """Create a temporary config file."""
    config = {
        "service_name": "test_service",
        "deployment_environment": "test",
        "jaeger": {
            "agent": {
                "host": "localhost",
                "port": 6831
            }
        }
    }
    config_path = tmp_path / "telemetry_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return str(config_path)

def test_load_config(config_file):
    """Test loading configuration."""
    manager = TelemetryManager(config_file)
    manager.load_config()
    
    assert manager.config["service_name"] == "test_service"
    assert manager.config["deployment_environment"] == "test"
    assert manager.config["jaeger"]["agent"]["host"] == "localhost"
    
@patch("opentelemetry.sdk.trace.TracerProvider")
@patch("opentelemetry.exporter.jaeger.JaegerExporter")
def test_setup_telemetry(mock_jaeger, mock_provider, config_file):
    """Test telemetry setup."""
    manager = TelemetryManager(config_file)
    manager.load_config()
    manager.setup_telemetry()
    
    assert mock_jaeger.called
    assert mock_provider.called
    assert manager.tracer is not None
    
def test_get_tracer(config_file):
    """Test getting tracer."""
    manager = TelemetryManager(config_file)
    tracer = manager.get_tracer()
    
    assert tracer is not None
    assert isinstance(trace.get_tracer_provider(), TracerProvider)
    
@patch("opentelemetry.trace.get_tracer_provider")
def test_shutdown(mock_provider, config_file):
    """Test telemetry shutdown."""
    manager = TelemetryManager(config_file)
    manager.shutdown()
    
    assert mock_provider().shutdown.called
    
def test_create_telemetry_manager(config_file):
    """Test creating and initializing telemetry manager."""
    manager = create_telemetry_manager(config_file)
    
    assert isinstance(manager, TelemetryManager)
    assert manager.tracer is not None
    assert manager.config is not None
    
def test_invalid_config_path():
    """Test handling invalid config path."""
    manager = TelemetryManager("nonexistent.json")
    manager.load_config()
    
    assert manager.config == {}
    
@patch("opentelemetry.exporter.jaeger.JaegerExporter")
def test_telemetry_setup_failure(mock_jaeger, config_file):
    """Test handling telemetry setup failure."""
    mock_jaeger.side_effect = Exception("Setup failed")
    
    manager = TelemetryManager(config_file)
    manager.setup_telemetry()
    
    # Should still have a fallback tracer
    assert manager.tracer is not None
    assert isinstance(trace.get_tracer_provider(), TracerProvider)
    
def test_environment_override(config_file):
    """Test environment variable override."""
    os.environ["DEPLOYMENT_ENV"] = "production"
    
    manager = create_telemetry_manager(config_file)
    resource = trace.get_tracer_provider().resource
    
    assert resource.attributes.get("deployment.environment") == "production"
    
    # Cleanup
    del os.environ["DEPLOYMENT_ENV"] 