"""Telemetry manager for distributed tracing."""

import json
import logging
import os
from typing import Optional, Dict, Any
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)

class TelemetryManager:
    """Manages OpenTelemetry configuration and setup."""
    
    def __init__(self, config_path: str):
        """Initialize telemetry manager.
        
        Args:
            config_path: Path to telemetry configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.tracer: Optional[trace.Tracer] = None
        
    def load_config(self) -> None:
        """Load telemetry configuration from file."""
        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load telemetry config: {str(e)}")
            self.config = {}
            
    def setup_telemetry(self) -> None:
        """Setup OpenTelemetry with Jaeger exporter."""
        try:
            # Create Jaeger exporter
            jaeger_config = self.config.get("jaeger", {}).get("agent", {})
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_config.get("host", "localhost"),
                agent_port=jaeger_config.get("port", 6831),
            )
            
            # Create resource
            resource = Resource.create({
                "service.name": self.config.get("service_name", "multi_agent_framework"),
                "deployment.environment": os.getenv(
                    "DEPLOYMENT_ENV",
                    self.config.get("deployment_environment", "development")
                )
            })
            
            # Create and configure provider
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
            trace.set_tracer_provider(provider)
            
            # Get tracer for this module
            self.tracer = trace.get_tracer(__name__)
            logger.info("Telemetry setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup telemetry: {str(e)}")
            # Setup minimal tracer for fallback
            trace.set_tracer_provider(TracerProvider())
            self.tracer = trace.get_tracer(__name__)
            
    def get_tracer(self) -> trace.Tracer:
        """Get the configured tracer.
        
        Returns:
            The configured OpenTelemetry tracer
        """
        if self.tracer is None:
            self.load_config()
            self.setup_telemetry()
        return self.tracer
        
    def shutdown(self) -> None:
        """Shutdown telemetry system."""
        try:
            trace.get_tracer_provider().shutdown()
            logger.info("Telemetry shutdown completed")
        except Exception as e:
            logger.error(f"Error during telemetry shutdown: {str(e)}")
            
def create_telemetry_manager(config_path: str) -> TelemetryManager:
    """Create and initialize a telemetry manager.
    
    Args:
        config_path: Path to telemetry configuration file
        
    Returns:
        Initialized TelemetryManager instance
    """
    manager = TelemetryManager(config_path)
    manager.load_config()
    manager.setup_telemetry()
    return manager 