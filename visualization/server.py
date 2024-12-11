"""
Server module for running the coordination dashboard.
"""

import argparse
from typing import Optional
import logging

from ..core.agent_coordination import AgentCoordinator, CoordinationStrategy
from ..core.self_analysis import SelfAnalysis
from .coordination_dashboard import CoordinationDashboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_dashboard(coordinator: Optional[AgentCoordinator] = None,
                 analyzer: Optional[SelfAnalysis] = None,
                 port: int = 8050,
                 debug: bool = False):
    """Run the coordination dashboard server."""
    
    # Create components if not provided
    if coordinator is None:
        logger.info("Creating new coordinator with adaptive strategy")
        coordinator = AgentCoordinator(strategy=CoordinationStrategy.ADAPTIVE)
        
    if analyzer is None:
        logger.info("Creating new self-analysis system")
        analyzer = SelfAnalysis()
        
    # Create and run dashboard
    logger.info(f"Starting dashboard server on port {port}")
    dashboard = CoordinationDashboard(
        coordinator=coordinator,
        analyzer=analyzer
    )
    
    try:
        dashboard.run(debug=debug, port=port)
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        raise

def main():
    """Main entry point for running the dashboard server."""
    parser = argparse.ArgumentParser(description="Run the coordination dashboard server")
    parser.add_argument("--port", type=int, default=8050,
                       help="Port to run the server on")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode")
    
    args = parser.parse_args()
    run_dashboard(port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 