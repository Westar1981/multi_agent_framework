"""
Advanced test suite for knowledge management and learning capabilities.
"""

import pytest
import asyncio
import numpy as np
from typing import Dict, List, Any
import torch
from datetime import datetime, timedelta

from ..agents.neural_symbolic_agent import NeuralSymbolicAgent
from ..agents.learner_agent import LearnerAgent
from ..core.framework import Framework
from ..core.self_analysis import SelfAnalysis

class TestKnowledgeManagement:
    """Test knowledge management capabilities."""
    
    @pytest.fixture
    async def framework(self):
        """Create framework instance with learning capabilities."""
        framework = Framework()
        await framework.initialize()
        
        # Add learning agents
        learner = LearnerAgent("main_learner", ["learning"])
        neural_agent = NeuralSymbolicAgent("neural_1", ["code_analysis"])
        
        framework.add_agent(learner)
        framework.add_agent(neural_agent)
        
        yield framework
        await framework.shutdown()
        
    @pytest.mark.asyncio
    async def test_knowledge_acquisition(self, framework):
        """Test incremental knowledge acquisition."""
        learner = framework.get_agent("main_learner")
        
        # Initial knowledge state
        initial_knowledge = learner.get_knowledge_size()
        
        # Feed new information
        test_data = [
            {
                'type': 'code_pattern',
                'content': f'pattern_{i}',
                'metadata': {'confidence': 0.8 + i * 0.02}
            }
            for i in range(10)
        ]
        
        for data in test_data:
            await learner.process_message({
                'type': 'learn',
                'content': data
            })
            
        # Verify knowledge growth
        final_knowledge = learner.get_knowledge_size()
        assert final_knowledge > initial_knowledge
        
        # Test knowledge retention
        retention_test = await learner.process_message({
            'type': 'query',
            'content': 'pattern_5'
        })
        assert retention_test is not None
        
    @pytest.mark.asyncio
    async def test_knowledge_consistency(self, framework):
        """Test knowledge consistency during updates."""
        learner = framework.get_agent("main_learner")
        
        # Add contradictory information
        contradictions = [
            {
                'type': 'rule',
                'content': {'if': 'A', 'then': 'B'},
                'confidence': 0.9
            },
            {
                'type': 'rule',
                'content': {'if': 'A', 'then': 'not B'},
                'confidence': 0.8
            }
        ]
        
        for rule in contradictions:
            await learner.process_message({
                'type': 'learn',
                'content': rule
            })
            
        # Query for resolution
        result = await learner.process_message({
            'type': 'query',
            'content': {'condition': 'A', 'query': 'B'}
        })
        
        # Should resolve to higher confidence rule
        assert result['confidence'] > 0.85
        
    @pytest.mark.asyncio
    async def test_knowledge_transfer(self, framework):
        """Test knowledge transfer between agents."""
        learner = framework.get_agent("main_learner")
        neural_agent = framework.get_agent("neural_1")
        
        # Learn pattern in learner
        pattern = {
            'type': 'code_pattern',
            'content': 'transfer_test_pattern',
            'metadata': {'confidence': 0.9}
        }
        
        await learner.process_message({
            'type': 'learn',
            'content': pattern
        })
        
        # Transfer to neural agent
        await framework.transfer_knowledge(
            source="main_learner",
            target="neural_1",
            knowledge_type="code_pattern"
        )
        
        # Verify transfer
        result = await neural_agent.process_message({
            'type': 'query',
            'content': 'transfer_test_pattern'
        })
        
        assert result is not None
        assert result['confidence'] > 0.8
        
class TestLearningCapabilities:
    """Test learning and adaptation capabilities."""
    
    @pytest.fixture
    async def learner_agent(self):
        """Create learner agent instance."""
        agent = LearnerAgent("test_learner", ["learning"])
        await agent.initialize()
        yield agent
        await agent.shutdown()
        
    @pytest.mark.asyncio
    async def test_pattern_learning(self, learner_agent):
        """Test learning of code patterns."""
        # Training data
        patterns = [
            {
                'code': 'def function_a(): pass',
                'pattern': 'empty_function',
                'metadata': {'complexity': 1}
            },
            {
                'code': 'class ClassA: pass',
                'pattern': 'empty_class',
                'metadata': {'complexity': 1}
            }
        ]
        
        # Train on patterns
        for pattern in patterns:
            await learner_agent.learn_pattern(pattern)
            
        # Test pattern recognition
        test_code = 'def function_b(): pass'
        result = await learner_agent.recognize_pattern(test_code)
        
        assert result['pattern'] == 'empty_function'
        assert result['confidence'] > 0.8
        
    @pytest.mark.asyncio
    async def test_adaptive_learning(self, learner_agent):
        """Test adaptation to new patterns."""
        # Initial training
        await learner_agent.learn_pattern({
            'code': 'x = 1',
            'pattern': 'assignment',
            'metadata': {'complexity': 1}
        })
        
        # New pattern type
        new_pattern = {
            'code': 'x: int = 1',
            'pattern': 'typed_assignment',
            'metadata': {'complexity': 2}
        }
        
        # Learn new pattern
        await learner_agent.learn_pattern(new_pattern)
        
        # Test adaptation
        test_code = 'y: str = "test"'
        result = await learner_agent.recognize_pattern(test_code)
        
        assert result['pattern'] == 'typed_assignment'
        assert result['confidence'] > 0.7
        
    @pytest.mark.asyncio
    async def test_performance_improvement(self, learner_agent):
        """Test learning performance improvement over time."""
        # Create test dataset
        test_cases = [
            {
                'input': f'test_case_{i}',
                'expected': f'result_{i}'
            }
            for i in range(10)
        ]
        
        # Track performance
        accuracies = []
        
        for _ in range(3):  # 3 learning iterations
            correct = 0
            total = 0
            
            for case in test_cases:
                result = await learner_agent.process_message({
                    'type': 'predict',
                    'content': case['input']
                })
                
                if result['output'] == case['expected']:
                    correct += 1
                total += 1
                
                # Learn from result
                await learner_agent.learn_from_feedback({
                    'input': case['input'],
                    'expected': case['expected'],
                    'actual': result['output']
                })
                
            accuracies.append(correct / total)
            
        # Performance should improve
        assert accuracies[-1] > accuracies[0]
        
    @pytest.mark.asyncio
    async def test_error_recovery(self, learner_agent):
        """Test recovery from learning errors."""
        # Introduce error case
        error_case = {
            'code': 'syntax error',
            'pattern': 'invalid'
        }
        
        # Handle error
        try:
            await learner_agent.learn_pattern(error_case)
        except Exception:
            pass
        
        # Verify recovery
        valid_case = {
            'code': 'x = 1',
            'pattern': 'assignment'
        }
        
        # Should still learn valid patterns
        await learner_agent.learn_pattern(valid_case)
        result = await learner_agent.recognize_pattern('y = 2')
        
        assert result['pattern'] == 'assignment'
        assert result['confidence'] > 0.7 