import asyncio
from core.orchestrator import AgentOrchestrator
from agents.code_analyzer import CodeAnalyzer
from agents.code_transformer import CodeTransformerAgent
from agents.prolog_reasoner import PrologReasoner
from agents.meta_reasoner import MetaReasoner
from loguru import logger
import inspect
from typing import Dict, Any, Set

async def setup_agent_pools(orchestrator: AgentOrchestrator) -> None:
    """Set up different agent pools with specialized capabilities."""
    try:
        # Create analysis pool
        analyzer = CodeAnalyzer()
        analyzer.add_capability(AgentCapability(
            name="code_analysis",
            description="Analyzes code structure and patterns",
            input_types={"source_code"},
            output_types={"analysis_result"},
            performance_metrics={}
        ))
        await orchestrator.register_agent(analyzer, "analysis_pool")
        
        # Create transformation pool
        transformer = CodeTransformerAgent()
        transformer.add_capability(AgentCapability(
            name="code_transformation",
            description="Transforms code using aspects",
            input_types={"source_code", "aspects"},
            output_types={"transformed_code"},
            performance_metrics={}
        ))
        await orchestrator.register_agent(transformer, "transformation_pool")
        
        # Create reasoning pool
        reasoner = PrologReasoner()
        reasoner.add_capability(AgentCapability(
            name="logic_reasoning",
            description="Performs logical reasoning on code",
            input_types={"source_code", "rules"},
            output_types={"reasoning_result"},
            performance_metrics={}
        ))
        await orchestrator.register_agent(reasoner, "reasoning_pool")
        
        # Create meta-reasoning pool
        meta_reasoner = MetaReasoner()
        meta_reasoner.add_capability(AgentCapability(
            name="meta_analysis",
            description="Analyzes system behavior and suggests improvements",
            input_types={"system_state", "agent_codes"},
            output_types={"improvement_suggestions", "capability_gaps"},
            performance_metrics={}
        ))
        await orchestrator.register_agent(meta_reasoner, "meta_pool")
        
    except Exception as e:
        logger.error(f"Error setting up agent pools: {str(e)}")
        raise

async def process_code_with_scaling(orchestrator: AgentOrchestrator) -> None:
    """Process code with automatic pool scaling."""
    try:
        # Example code with various patterns to analyze
        sample_code = """
class DatabaseConnection:
    _instance = None
    
    def __init__(self):
        if DatabaseConnection._instance:
            raise Exception("Singleton instance already exists")
        self.connected = False
    
    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
    
    def connect(self):
        self.connected = True

class UserFactory:
    @staticmethod
    def create_admin():
        return User(role="admin")
    
    @staticmethod
    def create_regular():
        return User(role="regular")

class Subject:
    def __init__(self):
        self.observers = []
        self.state = None
    
    def attach(self, observer):
        self.observers.append(observer)
    
    def notify(self):
        for observer in self.observers:
            observer.update(self.state)

def complex_calculation(a, b, c, d, e, f):
    result = 0
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        result = f
    return result

def recursive_fibonacci(n, cache=None):
    if cache is None:
        cache = {}
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    cache[n] = recursive_fibonacci(n-1, cache) + recursive_fibonacci(n-2, cache)
    return cache[n]

class DataProcessor:
    def process_large_dataset(self, data):
        intermediate = self.preprocess(data)
        validated = self.validate(intermediate)
        transformed = self.transform(validated)
        normalized = self.normalize(transformed)
        aggregated = self.aggregate(normalized)
        return self.postprocess(aggregated)
    
    def preprocess(self, data): return data
    def validate(self, data): return data
    def transform(self, data): return data
    def normalize(self, data): return data
    def aggregate(self, data): return data
    def postprocess(self, data): return data
"""
        
        # Simulate high load by processing the code multiple times
        for _ in range(10):
            # Get agents from pools
            analysis_pool = orchestrator.agent_pools["analysis_pool"]
            transformation_pool = orchestrator.agent_pools["transformation_pool"]
            reasoning_pool = orchestrator.agent_pools["reasoning_pool"]
            
            # Process with each available agent
            for analyzer in analysis_pool:
                await orchestrator.framework.execute_capability_chain(
                    sample_code,
                    "source_code",
                    "analysis_result"
                )
            
            for transformer in transformation_pool:
                aspects = [
                    {
                        "pointcut": "complex_calculation",
                        "advice_type": "before",
                        "advice_code": "lambda: print('Warning: Entering complex function')"
                    }
                ]
                await orchestrator.framework.execute_capability_chain(
                    {"code": sample_code, "aspects": aspects},
                    "source_code",
                    "transformed_code"
                )
            
            for reasoner in reasoning_pool:
                await orchestrator.framework.execute_capability_chain(
                    sample_code,
                    "source_code",
                    "reasoning_result"
                )
            
            # Short delay to simulate processing time
            await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Error processing code: {str(e)}")
        raise

async def monitor_system_health(orchestrator: AgentOrchestrator) -> None:
    """Monitor system health and performance."""
    try:
        # Get initial system state
        state = orchestrator.framework.get_system_state()
        logger.info("Initial system state:", state)
        
        # Monitor for a period
        for _ in range(5):
            await asyncio.sleep(10)
            state = orchestrator.framework.get_system_state()
            logger.info("Updated system state:", state)
            
            # Check agent pool sizes
            for pool_name, agents in orchestrator.agent_pools.items():
                logger.info(f"Pool {pool_name} size: {len(agents)}")
                
            # Check agent performance
            for pool in orchestrator.agent_pools.values():
                for agent in pool:
                    metrics = agent.get_performance_metrics()
                    logger.info(f"Agent {agent.name} metrics:", metrics)
                    
    except Exception as e:
        logger.error(f"Error monitoring system: {str(e)}")
        raise

async def main() -> None:
    """Main execution function."""
    orchestrator = None
    try:
        # Initialize orchestrator
        orchestrator = AgentOrchestrator()
        await orchestrator.start()
        
        # Set up agent pools
        await setup_agent_pools(orchestrator)
        
        # Process code with automatic scaling
        await process_code_with_scaling(orchestrator)
        
        # Monitor system health
        await monitor_system_health(orchestrator)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        if orchestrator:
            await orchestrator.stop()

if __name__ == "__main__":
    try:
        # Configure logging
        logger.add("multi_agent_framework.log", rotation="500 MB")
        
        # Run the example
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise 