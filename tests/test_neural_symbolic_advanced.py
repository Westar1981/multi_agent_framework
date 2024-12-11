"""
Advanced test suite for neural-symbolic agent using property-based and metamorphic testing.
"""

from typing import Dict, Any, List
import pytest
from hypothesis import given, strategies as st
from ..testing.advanced_test_generator import (
    TestGenerator,
    TestProperty,
    StatefulTestGenerator,
    TestCase,
    TestSuite
)
from ..agents.neural_symbolic_agent import NeuralSymbolicAgent

class TestNeuralSymbolicAdvanced:
    """Advanced tests for neural-symbolic agent."""
    
    def setup_method(self):
        """Setup test environment."""
        self.agent = NeuralSymbolicAgent()
        self.test_generator = TestGenerator()
        self.setup_test_properties()
        
    def setup_test_properties(self):
        """Setup test properties and metamorphic relations."""
        # Property: Learning improves performance
        self.test_generator.register_property(
            TestProperty(
                name="learning_improves_performance",
                description="Learning from examples should improve agent performance",
                property_type="invariant",
                preconditions=["len(self.agent.learned_rules) > 0"],
                postconditions=[
                    "self.agent.self_analysis.analyze_performance()['accuracy']['mean'] > 0.5"
                ]
            )
        )
        
        # Property: Knowledge consistency
        self.test_generator.register_property(
            TestProperty(
                name="knowledge_consistency",
                description="Learned knowledge should be consistent across queries",
                property_type="invariant",
                preconditions=["self.agent.reasoner is not None"],
                postconditions=[
                    "all(rule in self.agent.learned_rules for rule in previous_rules)"
                ]
            )
        )
        
        # Add metamorphic relations
        self.test_generator.add_metamorphic_relation(
            source="learn_simple_rule",
            follow_up="learn_complex_rule",
            relation=lambda x, y: y['confidence'] <= x['confidence']
        )
        
    @given(st.lists(st.dictionaries(
        keys=st.sampled_from(['rule', 'pattern', 'label']),
        values=st.text(min_size=1),
        min_size=3
    )))
    def test_learning_property(self, examples):
        """Test learning property with generated examples."""
        # Record initial performance
        initial_performance = self.agent.self_analysis.analyze_performance()
        
        # Learn from examples
        for example in examples:
            self.agent.learn_from_example(example)
            
        # Verify performance improvement
        final_performance = self.agent.self_analysis.analyze_performance()
        assert final_performance['accuracy']['mean'] >= initial_performance['accuracy']['mean']
        
    def test_metamorphic_learning(self):
        """Test metamorphic relation in learning."""
        # Simple rule
        simple_example = {
            'rule': 'parent(X, Y)',
            'pattern': 'X is parent of Y',
            'label': True
        }
        
        # Complex rule (composition of simple rules)
        complex_example = {
            'rule': 'parent(X, Y), parent(Y, Z) -> grandparent(X, Z)',
            'pattern': 'If X is parent of Y and Y is parent of Z, then X is grandparent of Z',
            'label': True
        }
        
        # Learn simple rule
        simple_result = self.agent.learn_from_example(simple_example)
        
        # Learn complex rule
        complex_result = self.agent.learn_from_example(complex_example)
        
        # Verify metamorphic relation
        assert complex_result['confidence'] <= simple_result['confidence']
        
    def test_stateful_behavior(self):
        """Test stateful behavior using state machine."""
        state_generator = StatefulTestGenerator()
        
        # Define state variables and transitions
        state_vars = ['current_rules', 'performance_metrics']
        transitions = ['learn_rule', 'query_knowledge', 'optimize_system']
        
        # Generate and run state machine tests
        StateMachine = state_generator.generate_state_machine(
            NeuralSymbolicAgent,
            state_vars,
            transitions
        )
        
        StateMachine.TestCase.settings = settings(
            max_examples=100,
            stateful_step_count=10
        )
        
        state_machine = StateMachine()
        state_machine.check_invariants()
        
    def test_knowledge_consistency(self):
        """Test consistency of learned knowledge."""
        test_suite = TestSuite("KnowledgeConsistency")
        
        # Setup
        test_suite.set_setup("""
            self.agent = NeuralSymbolicAgent()
            self.initial_rules = set()
        """)
        
        # Test case 1: Basic consistency
        case1 = TestCase(
            "basic_consistency",
            "Verify basic knowledge consistency after learning"
        )
        case1.add_step("self.agent.learn_from_example({'rule': 'test_rule', 'pattern': 'test', 'label': True})")
        case1.add_step("learned_rules = set(self.agent.learned_rules.keys())")
        case1.add_assertion("'test_rule' in learned_rules")
        
        # Test case 2: Consistency after multiple learns
        case2 = TestCase(
            "multiple_learn_consistency",
            "Verify knowledge consistency after multiple learning steps"
        )
        case2.add_step("""
            examples = [
                {'rule': f'rule_{i}', 'pattern': f'pattern_{i}', 'label': True}
                for i in range(3)
            ]
        """)
        case2.add_step("for ex in examples: self.agent.learn_from_example(ex)")
        case2.add_step("learned_rules = set(self.agent.learned_rules.keys())")
        case2.add_assertion("all(f'rule_{i}' in learned_rules for i in range(3))")
        
        # Add test cases to suite
        test_suite.add_test_case(case1)
        test_suite.add_test_case(case2)
        
        # Generate and execute tests
        exec(test_suite.generate_code())
        
    @given(st.lists(st.dictionaries(
        keys=st.sampled_from(['query', 'context']),
        values=st.text(min_size=1),
        min_size=2
    )))
    def test_reasoning_stability(self, queries):
        """Test stability of reasoning results."""
        # Learn some basic rules
        self.agent.learn_from_example({
            'rule': 'basic_fact(X)',
            'pattern': 'X is a basic fact',
            'label': True
        })
        
        # Test multiple queries
        previous_results = {}
        for query in queries:
            result = self.agent.process_query(query)
            
            # Check if we've seen this query before
            if str(query) in previous_results:
                # Results should be consistent
                assert result == previous_results[str(query)]
            
            previous_results[str(query)] = result
            
    def test_performance_metrics(self):
        """Test performance metrics calculation and tracking."""
        # Generate test data
        test_data = [
            {'accuracy': 0.8, 'confidence': 0.9},
            {'accuracy': 0.85, 'confidence': 0.95},
            {'accuracy': 0.9, 'confidence': 0.98}
        ]
        
        # Update metrics
        for metrics in test_data:
            self.agent.self_analysis.performance_metrics['accuracy'].append(metrics['accuracy'])
            self.agent.self_analysis.performance_metrics['confidence'].append(metrics['confidence'])
            
        # Analyze performance
        analysis = self.agent.self_analysis.analyze_performance()
        
        # Verify metrics
        assert analysis['accuracy']['trend'] == 'improving'
        assert analysis['confidence']['trend'] == 'improving'
        assert analysis['accuracy']['mean'] > 0.8
        assert analysis['confidence']['mean'] > 0.9 