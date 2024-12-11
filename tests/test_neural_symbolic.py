import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from ..agents.neural_symbolic_agent import (
    NeuralSymbolicAgent,
    NeuralSymbolicReasoner,
    SymbolicRule,
    TemporalFact,
    ProbabilisticFact,
    TemporalProbabilisticFact
)

@pytest.fixture
def agent():
    return NeuralSymbolicAgent("test_agent")

@pytest.fixture
def reasoner():
    return NeuralSymbolicReasoner()

def test_rule_learning(reasoner):
    # Test rule learning with simple logical implication
    premise = "parent(X,Y)"
    conclusion = "ancestor(X,Y)"
    examples = [
        {"person1": "john", "person2": "mary", "label": True, "confidence": 0.9},
        {"person1": "mary", "person2": "bob", "label": True, "confidence": 0.8},
        {"person1": "john", "person2": "bob", "label": False, "confidence": 0.2}
    ]
    
    reasoner.learn_rule(premise, conclusion, examples)
    assert len(reasoner.rules) == 1
    assert reasoner.rules[0].premise == premise
    assert reasoner.rules[0].conclusion == conclusion

def test_temporal_rule_learning(reasoner):
    # Test rule learning with temporal constraints
    premise = "meeting(X,Y)"
    conclusion = "busy(X)"
    now = datetime.now()
    examples = [
        {
            "predicate": "meeting",
            "arguments": ["john", "mary"],
            "temporal": {
                "start": now,
                "end": now + timedelta(hours=1)
            },
            "label": True,
            "confidence": 0.9
        },
        {
            "predicate": "meeting",
            "arguments": ["bob", "alice"],
            "temporal": {
                "start": now + timedelta(hours=2),
                "end": now + timedelta(hours=3)
            },
            "label": True,
            "confidence": 0.8
        }
    ]
    
    temporal_constraints = {
        "min_duration": 1800,  # 30 minutes
        "max_duration": 7200,  # 2 hours
        "must_end_by": now + timedelta(days=1)
    }
    
    reasoner.learn_rule(premise, conclusion, examples, temporal_constraints)
    assert len(reasoner.rules) == 1
    assert reasoner.rules[0].temporal_constraints == temporal_constraints

def test_temporal_query_processing(reasoner):
    # First learn a rule with temporal constraints
    premise = "meeting(X,Y)"
    conclusion = "busy(X)"
    now = datetime.now()
    examples = [
        {
            "predicate": "meeting",
            "arguments": ["john", "mary"],
            "temporal": {
                "start": now,
                "end": now + timedelta(hours=1)
            },
            "label": True,
            "confidence": 0.9
        }
    ]
    
    temporal_constraints = {
        "min_duration": 1800,
        "max_duration": 7200
    }
    
    reasoner.learn_rule(premise, conclusion, examples, temporal_constraints)
    
    # Test valid temporal query
    query = "busy(john)"
    context = {
        "temporal": {
            "start": now + timedelta(minutes=30),
            "end": now + timedelta(hours=1)
        }
    }
    
    result, confidence = reasoner.query(query, context)
    assert isinstance(result, bool)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1
    
    # Test invalid temporal query (too short duration)
    invalid_context = {
        "temporal": {
            "start": now,
            "end": now + timedelta(minutes=15)
        }
    }
    
    result, confidence = reasoner.query(query, invalid_context)
    assert not result or confidence < 0.5

@pytest.mark.asyncio
async def test_agent_message_handling(agent):
    # Test learning message with temporal constraints
    now = datetime.now()
    learn_message = {
        "message_type": "learn_rule",
        "content": {
            "premise": "meeting(X,Y)",
            "conclusion": "busy(X)",
            "temporal_constraints": {
                "min_duration": 1800,
                "max_duration": 7200
            },
            "examples": [
                {
                    "predicate": "meeting",
                    "arguments": ["john", "mary"],
                    "temporal": {
                        "start": now,
                        "end": now + timedelta(hours=1)
                    },
                    "label": True,
                    "confidence": 0.9
                }
            ]
        },
        "sender": "test_sender"
    }
    
    await agent.process_message(learn_message)
    assert len(agent.learned_rules) == 1
    
    # Test temporal query message
    query_message = {
        "message_type": "query",
        "content": {
            "query": "busy(john)",
            "context": {
                "temporal": {
                    "start": now + timedelta(minutes=30),
                    "end": now + timedelta(hours=1)
                }
            }
        },
        "sender": "test_sender"
    }
    
    await agent.process_message(query_message)

def test_temporal_pattern_matching(reasoner):
    # Test pattern matching with temporal predicates
    pattern = "meeting_scheduled(X,Y,T1,T2)"
    query1 = "meeting_scheduled(john,mary,2023-12-01,2023-12-02)"
    query2 = "meeting_scheduled(bob,alice,2023-12-03,2023-12-04)"
    query3 = "call_scheduled(john,mary,2023-12-01,2023-12-02)"
    
    assert reasoner._matches_pattern(query1, pattern)
    assert reasoner._matches_pattern(query2, pattern)
    assert not reasoner._matches_pattern(query3, pattern)

def test_temporal_encoding(reasoner):
    # Test temporal feature encoding
    now = datetime.now()
    fact = TemporalFact(
        predicate="meeting",
        arguments=["john", "mary"],
        start_time=now,
        end_time=now + timedelta(hours=1),
        confidence=0.9
    )
    
    encoding = reasoner.encode_temporal_features(fact)
    assert isinstance(encoding, torch.Tensor)
    assert encoding.shape == (1, 4)  # 4 temporal features

def test_temporal_constraints(reasoner):
    now = datetime.now()
    constraints = {
        "min_duration": 1800,  # 30 minutes
        "max_duration": 7200,  # 2 hours
        "must_end_by": now + timedelta(days=1)
    }
    
    # Test valid temporal info
    valid_info = {
        "start": now,
        "end": now + timedelta(hours=1)
    }
    assert reasoner._check_temporal_constraints(constraints, valid_info)
    
    # Test invalid duration (too short)
    invalid_info = {
        "start": now,
        "end": now + timedelta(minutes=15)
    }
    assert not reasoner._check_temporal_constraints(constraints, invalid_info)
    
    # Test invalid end time
    late_info = {
        "start": now,
        "end": now + timedelta(days=2)
    }
    assert not reasoner._check_temporal_constraints(constraints, late_info)

def test_performance_metrics(agent):
    # Add rules with temporal examples
    now = datetime.now()
    rule_key = "meeting(X,Y) -> busy(X)"
    examples = [
        {
            "predicate": "meeting",
            "arguments": ["john", "mary"],
            "temporal": {
                "start": now,
                "end": now + timedelta(hours=1)
            },
            "label": True,
            "confidence": 0.9
        },
        {
            "predicate": "meeting",
            "arguments": ["bob", "alice"],
            "temporal": {
                "start": now + timedelta(hours=2),
                "end": now + timedelta(hours=3)
            },
            "label": True,
            "confidence": 0.8
        },
        {
            "predicate": "meeting",
            "arguments": ["john", "bob"],
            "temporal": {
                "start": now + timedelta(hours=4),
                "end": now + timedelta(minutes=10)
            },
            "label": False,
            "confidence": 0.2
        }
    ]
    agent.learned_rules[rule_key] = examples
    
    # Get performance metrics
    metrics = agent._calculate_rule_performance(examples)
    
    assert "accuracy" in metrics
    assert "confidence" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["confidence"], float)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["confidence"] <= 1

@pytest.mark.asyncio
async def test_error_handling(agent):
    # Test missing fields in temporal rule learning
    async with pytest.raises(ValueError):
        content = {
            "premise": "meeting(X,Y)",
            "temporal_constraints": {
                "min_duration": 1800
            }
            # Missing conclusion and examples
        }
        await agent.learn_rule(content, "test_sender")
    
    # Test missing query in temporal context
    async with pytest.raises(ValueError):
        content = {
            "context": {
                "temporal": {
                    "start": datetime.now(),
                    "end": datetime.now() + timedelta(hours=1)
                }
            }
            # Missing query
        }
        await agent.process_query(content, "test_sender") 

def test_probabilistic_fact_creation():
    fact = ProbabilisticFact(
        predicate="expert",
        arguments=["john", "programming"],
        alpha=8.0,  # Strong positive evidence
        beta=2.0,   # Weak negative evidence
        prior=0.5
    )
    
    assert fact.confidence > 0.7  # Should be confident given strong positive evidence
    assert 0 <= fact.confidence <= 1
    
    # Test belief update
    fact.update(success=True, weight=1.0)
    assert fact.alpha == 9.0
    assert fact.beta == 2.0
    
    fact.update(success=False, weight=0.5)
    assert fact.alpha == 9.0
    assert fact.beta == 2.5

def test_probabilistic_rule_learning(reasoner):
    # Test rule learning with uncertainty
    premise = "expert(X,Y)"
    conclusion = "can_solve(X,Y)"
    now = datetime.now()
    examples = [
        {
            "predicate": "expert",
            "arguments": ["john", "programming"],
            "prob_distribution": {
                "alpha": 8.0,
                "beta": 2.0,
                "prior": 0.5
            },
            "label": True,
            "confidence": 0.9
        },
        {
            "predicate": "expert",
            "arguments": ["mary", "design"],
            "prob_distribution": {
                "alpha": 6.0,
                "beta": 1.0,
                "prior": 0.5
            },
            "label": True,
            "confidence": 0.8
        }
    ]
    
    # Create probabilistic distribution for the rule
    prob_dist = ProbabilisticFact(
        predicate="expert_rule",
        arguments=["general"],
        alpha=7.0,
        beta=1.5
    )
    
    reasoner.learn_rule(
        premise=premise,
        conclusion=conclusion,
        examples=examples,
        prob_distribution=prob_dist
    )
    
    assert len(reasoner.rules) == 1
    assert reasoner.rules[0].prob_distribution is not None
    assert reasoner.rules[0].prob_distribution.confidence > 0.7

def test_probabilistic_query_processing(reasoner):
    # First learn a rule with uncertainty
    premise = "expert(X,Y)"
    conclusion = "can_solve(X,Y)"
    prob_dist = ProbabilisticFact(
        predicate="expert_rule",
        arguments=["general"],
        alpha=7.0,
        beta=1.5
    )
    
    examples = [
        {
            "predicate": "expert",
            "arguments": ["john", "programming"],
            "prob_distribution": {
                "alpha": 8.0,
                "beta": 2.0
            },
            "label": True,
            "confidence": 0.9
        }
    ]
    
    reasoner.learn_rule(
        premise=premise,
        conclusion=conclusion,
        examples=examples,
        prob_distribution=prob_dist
    )
    
    # Test query with uncertainty
    query = "can_solve(john,programming)"
    context = {
        "prob_distribution": {
            "alpha": 8.0,
            "beta": 2.0
        }
    }
    
    result, confidence, uncertainty_info = reasoner.query(query, context)
    
    assert isinstance(result, bool)
    assert isinstance(confidence, float)
    assert uncertainty_info is not None
    assert "credible_interval" in uncertainty_info
    assert "evidence_strength" in uncertainty_info
    assert len(uncertainty_info["credible_interval"]) == 2
    assert all(0 <= x <= 1 for x in uncertainty_info["credible_interval"])

def test_belief_updating(reasoner):
    # Test belief update mechanism
    reasoner.update_belief(
        predicate="expert",
        arguments=["john", "programming"],
        success=True,
        weight=1.0
    )
    
    fact_key = "expert(john,programming)"
    assert fact_key in reasoner.prob_facts
    assert reasoner.prob_facts[fact_key].alpha == 2.0  # Initial 1.0 + 1.0
    assert reasoner.prob_facts[fact_key].beta == 1.0   # Initial 1.0
    
    # Update with negative evidence
    reasoner.update_belief(
        predicate="expert",
        arguments=["john", "programming"],
        success=False,
        weight=0.5
    )
    
    assert reasoner.prob_facts[fact_key].alpha == 2.0
    assert reasoner.prob_facts[fact_key].beta == 1.5

def test_evidence_combination(reasoner):
    # Test combining multiple pieces of evidence
    facts = [
        ProbabilisticFact("expert", ["john", "programming"], 8.0, 2.0),
        ProbabilisticFact("expert", ["john", "programming"], 6.0, 1.0),
        ProbabilisticFact("expert", ["john", "programming"], 4.0, 3.0)
    ]
    
    combined = reasoner.combine_evidence(facts)
    assert combined.alpha == sum(f.alpha for f in facts)
    assert combined.beta == sum(f.beta for f in facts)
    assert 0 <= combined.confidence <= 1

def test_credible_intervals(reasoner):
    fact = ProbabilisticFact(
        predicate="expert",
        arguments=["john", "programming"],
        alpha=8.0,
        beta=2.0
    )
    
    interval = reasoner.get_belief_interval(fact, confidence=0.95)
    assert len(interval) == 2
    assert 0 <= interval[0] <= interval[1] <= 1
    assert interval[0] < fact.confidence < interval[1]

@pytest.mark.asyncio
async def test_probabilistic_message_handling(agent):
    # Test learning message with uncertainty
    learn_message = {
        "message_type": "learn_rule",
        "content": {
            "premise": "expert(X,Y)",
            "conclusion": "can_solve(X,Y)",
            "prob_distribution": {
                "alpha": 7.0,
                "beta": 1.5
            },
            "examples": [
                {
                    "predicate": "expert",
                    "arguments": ["john", "programming"],
                    "prob_distribution": {
                        "alpha": 8.0,
                        "beta": 2.0
                    },
                    "label": True,
                    "confidence": 0.9
                }
            ]
        },
        "sender": "test_sender"
    }
    
    await agent.process_message(learn_message)
    assert len(agent.learned_rules) == 1
    
    # Test query message with uncertainty
    query_message = {
        "message_type": "query",
        "content": {
            "query": "can_solve(john,programming)",
            "context": {
                "prob_distribution": {
                    "alpha": 8.0,
                    "beta": 2.0
                }
            }
        },
        "sender": "test_sender"
    }
    
    await agent.process_message(query_message)

def test_markov_logic_network(reasoner):
    # Test MLN formula creation and inference
    reasoner.mln.add_formula(
        "expert(X,Y) => can_solve(X,Y)",
        weight=1.5
    )
    reasoner.mln.add_formula(
        "expert(X,Y) && has_time(X) => available(X,Y)",
        weight=1.0,
        temporal=True
    )
    
    # Test inference
    evidence = {
        "expert": True,
        "has_time": True,
        "temporal": True
    }
    
    prob = reasoner.mln.infer("can_solve(john,programming)", evidence)
    assert 0 <= prob <= 1
    assert prob > 0.5  # Should be confident given positive evidence

def test_temporal_probabilistic_fact():
    # Test fact with both temporal and uncertainty aspects
    now = datetime.now()
    fact = TemporalProbabilisticFact(
        predicate="available",
        arguments=["john", "programming"],
        start_time=now,
        end_time=now + timedelta(hours=2),
        alpha=8.0,
        beta=2.0,
        decay_rate=0.1
    )
    
    # Test initial confidence
    conf = fact.get_confidence(now)
    assert conf > 0.7  # Should be confident initially
    
    # Test confidence decay over time
    future = now + timedelta(days=1)
    decayed_conf = fact.get_confidence(future)
    assert decayed_conf < conf  # Should have lower confidence after time passes

def test_parameter_learning(reasoner):
    # Test learning optimal uncertainty parameters
    fact = ProbabilisticFact(
        predicate="expert",
        arguments=["john", "programming"],
        alpha=1.0,
        beta=1.0
    )
    
    # Add observations
    observations = [
        (True, 1.0),   # Success with full weight
        (True, 0.8),   # Success with partial weight
        (False, 0.3),  # Failure with low weight
        (True, 0.9)    # Another success
    ]
    
    fact.learn_parameters(observations)
    
    # Parameters should be updated to reflect observations
    assert fact.alpha > 1.0  # Should increase with positive observations
    assert fact.beta > 1.0   # Should increase with negative observation
    assert fact.confidence > 0.5  # Should be confident given mostly positive observations

def test_combined_reasoning(reasoner):
    # Test integration of neural, MLN, and temporal-probabilistic reasoning
    now = datetime.now()
    
    # Add MLN formulas
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    reasoner.mln.add_formula("available(X,T) => can_help(X,T)", 1.0, temporal=True)
    
    # Add temporal-probabilistic fact
    fact = TemporalProbabilisticFact(
        predicate="available",
        arguments=["john", "programming"],
        start_time=now,
        end_time=now + timedelta(hours=2),
        alpha=8.0,
        beta=2.0
    )
    reasoner.add_temporal_probabilistic_fact(fact)
    
    # Test query with all aspects
    query = "can_help(john,programming)"
    context = {
        "temporal": {
            "start": now,
            "end": now + timedelta(hours=1)
        },
        "prob_distribution": {
            "alpha": 7.0,
            "beta": 2.0
        }
    }
    
    result, confidence, uncertainty_info = reasoner.query(query, context)
    
    assert isinstance(result, bool)
    assert 0 <= confidence <= 1
    assert "neural_confidence" in uncertainty_info
    assert "mln_probability" in uncertainty_info
    assert "temporal_confidence" in uncertainty_info
    assert "credible_intervals" in uncertainty_info

def test_uncertainty_parameter_optimization(reasoner):
    # Test end-to-end parameter learning
    now = datetime.now()
    
    # Add some observations to learning history
    reasoner.learning_history.extend([
        ("expert", True, 0.9),
        ("expert", True, 0.8),
        ("expert", False, 0.3),
        ("available", True, 0.7),
        ("available", False, 0.4)
    ])
    
    # Add facts to optimize
    reasoner.prob_facts["expert(john,programming)"] = ProbabilisticFact(
        predicate="expert",
        arguments=["john", "programming"],
        alpha=1.0,
        beta=1.0
    )
    
    reasoner.prob_facts["available(john,programming)"] = ProbabilisticFact(
        predicate="available",
        arguments=["john", "programming"],
        alpha=1.0,
        beta=1.0
    )
    
    # Optimize parameters
    reasoner.learn_uncertainty_parameters()
    
    # Check that parameters were updated
    expert_fact = reasoner.prob_facts["expert(john,programming)"]
    available_fact = reasoner.prob_facts["available(john,programming)"]
    
    assert expert_fact.alpha != 1.0 or expert_fact.beta != 1.0
    assert available_fact.alpha != 1.0 or available_fact.beta != 1.0
    assert expert_fact.confidence > 0.5  # Should be confident given mostly positive observations

def test_temporal_decay():
    # Test confidence decay over different time periods
    now = datetime.now()
    fact = TemporalProbabilisticFact(
        predicate="available",
        arguments=["john", "programming"],
        start_time=now,
        end_time=now + timedelta(hours=2),
        alpha=8.0,
        beta=2.0,
        decay_rate=0.1
    )
    
    # Test confidence at different times
    confidences = [
        fact.get_confidence(now),
        fact.get_confidence(now + timedelta(hours=1)),
        fact.get_confidence(now + timedelta(hours=2)),
        fact.get_confidence(now + timedelta(days=1))
    ]
    
    # Confidence should decrease over time
    assert all(c1 > c2 for c1, c2 in zip(confidences, confidences[1:]))
    assert confidences[-1] < 0.5  # Should have low confidence after a long time

@pytest.mark.asyncio
async def test_integrated_message_handling(agent):
    # Test handling messages with all reasoning aspects
    now = datetime.now()
    
    # Add MLN formula
    agent.reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    
    # Test learning message
    learn_message = {
        "message_type": "learn_rule",
        "content": {
            "premise": "expert(X,Y)",
            "conclusion": "can_solve(X,Y)",
            "temporal_constraints": {
                "min_duration": 1800,
                "max_duration": 7200
            },
            "prob_distribution": {
                "alpha": 7.0,
                "beta": 1.5
            },
            "examples": [
                {
                    "predicate": "expert",
                    "arguments": ["john", "programming"],
                    "temporal": {
                        "start": now,
                        "end": now + timedelta(hours=1)
                    },
                    "prob_distribution": {
                        "alpha": 8.0,
                        "beta": 2.0
                    },
                    "label": True,
                    "confidence": 0.9
                }
            ]
        },
        "sender": "test_sender"
    }
    
    await agent.process_message(learn_message)
    assert len(agent.learned_rules) == 1
    
    # Test query message with all aspects
    query_message = {
        "message_type": "query",
        "content": {
            "query": "can_solve(john,programming)",
            "context": {
                "temporal": {
                    "start": now + timedelta(minutes=30),
                    "end": now + timedelta(hours=1)
                },
                "prob_distribution": {
                    "alpha": 8.0,
                    "beta": 2.0
                }
            }
        },
        "sender": "test_sender"
    }
    
    await agent.process_message(query_message)

def test_mln_clause_creation():
    # Test MLN clause creation and string representation
    clause = MLNClause(
        literals=["expert(X,Y)", "can_solve(X,Y)"],
        weight=1.5,
        temporal=False
    )
    
    assert len(clause.literals) == 2
    assert clause.weight == 1.5
    assert not clause.temporal
    assert str(clause) == "expert(X,Y) ∨ can_solve(X,Y) [1.5]"

def test_mln_formula_parsing(reasoner):
    # Test formula parsing
    formula = "expert(X,Y) && has_time(X) => can_solve(X,Y)"
    literals = reasoner.mln.parse_formula(formula)
    
    assert len(literals) == 3
    assert "expert(X,Y)" in literals
    assert "has_time(X)" in literals
    assert "can_solve(X,Y)" in literals

def test_mln_grounding_generation(reasoner):
    # Test grounding generation
    reasoner.mln.constants.update(["john", "programming"])
    clause = MLNClause(
        literals=["expert(X,Y)"],
        weight=1.0
    )
    
    groundings = reasoner.mln._generate_groundings(clause, reasoner.mln.constants)
    assert len(groundings) == 1  # One literal
    assert len(groundings[0]) == 4  # All possible combinations
    assert "expert(john,programming)" in groundings[0]

def test_mcmc_inference(reasoner):
    # Test MCMC inference
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    reasoner.mln.add_formula("has_time(X) => available(X)", 1.0)
    
    evidence = {
        "expert(john,programming)": True,
        "has_time(john)": True
    }
    
    # Test MCMC inference
    prob_mcmc = reasoner.mln.infer(
        "can_solve(john,programming)",
        evidence,
        method="mcmc"
    )
    
    assert 0 <= prob_mcmc <= 1
    assert prob_mcmc > 0.5  # Should be likely given the evidence

def test_exact_inference(reasoner):
    # Test exact inference with small domain
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    
    evidence = {
        "expert(john,programming)": True
    }
    
    # Test exact inference
    prob_exact = reasoner.mln.infer(
        "can_solve(john,programming)",
        evidence,
        method="exact"
    )
    
    assert 0 <= prob_exact <= 1
    assert prob_exact > 0.5  # Should be likely given the evidence

def test_temporal_mln_inference(reasoner):
    # Test temporal MLN inference
    now = datetime.now()
    
    reasoner.mln.add_formula(
        "expert(X,Y) && available(X,T)",
        weight=1.5,
        temporal=True
    )
    
    evidence = {
        "expert(john,programming)": True,
        "temporal": {
            "start": now,
            "end": now + timedelta(hours=1)
        },
        "available(john,now)": True
    }
    
    prob = reasoner.mln.infer(
        "can_solve(john,programming)",
        evidence,
        method="mcmc"
    )
    
    assert 0 <= prob <= 1
    assert prob > 0.5  # Should be likely given the evidence

def test_mln_energy_calculation(reasoner):
    # Test energy calculation
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    
    state = {
        "expert(john,programming)": True,
        "can_solve(john,programming)": True
    }
    
    energy = reasoner.mln._calculate_energy(state)
    assert energy < 0  # Should be negative when formula is satisfied

def test_inference_methods_comparison(reasoner):
    # Compare different inference methods
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    
    evidence = {
        "expert(john,programming)": True
    }
    
    # Get probabilities from both methods
    prob_mcmc = reasoner.mln.infer(
        "can_solve(john,programming)",
        evidence,
        method="mcmc"
    )
    
    prob_exact = reasoner.mln.infer(
        "can_solve(john,programming)",
        evidence,
        method="exact"
    )
    
    # Results should be similar for simple cases
    assert abs(prob_mcmc - prob_exact) < 0.2

def test_combined_temporal_probabilistic_inference(reasoner):
    # Test combination of temporal and probabilistic inference
    now = datetime.now()
    
    # Add temporal MLN formula
    reasoner.mln.add_formula(
        "expert(X,Y) && available(X,T) => can_help(X,Y,T)",
        weight=1.5,
        temporal=True
    )
    
    # Add temporal-probabilistic fact
    fact = TemporalProbabilisticFact(
        predicate="available",
        arguments=["john", "programming"],
        start_time=now,
        end_time=now + timedelta(hours=2),
        alpha=8.0,
        beta=2.0
    )
    reasoner.add_temporal_probabilistic_fact(fact)
    
    # Test inference with both temporal and probabilistic aspects
    evidence = {
        "expert(john,programming)": True,
        "temporal": {
            "start": now,
            "end": now + timedelta(hours=1)
        }
    }
    
    result, confidence, uncertainty_info = reasoner.query(
        "can_help(john,programming,now)",
        evidence
    )
    
    assert isinstance(result, bool)
    assert 0 <= confidence <= 1
    assert "mln_probability" in uncertainty_info
    assert uncertainty_info["mln_probability"] > 0.5

def test_gibbs_sampling(reasoner):
    # Test Gibbs sampling inference
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    reasoner.mln.add_formula("has_time(X) => available(X)", 1.0)
    
    evidence = {
        "expert(john,programming)": True,
        "has_time(john)": True
    }
    
    gibbs = GibbsSampler(burn_in=50, sample_gap=5)
    prob = gibbs.sample(
        reasoner.mln,
        "can_solve(john,programming)",
        evidence,
        num_samples=500
    )
    
    assert 0 <= prob <= 1
    assert prob > 0.5  # Should be likely given the evidence
    assert len(gibbs.samples) == 500

def test_lifted_inference(reasoner):
    # Test lifted inference
    lifted = LiftedInference()
    
    # Add type information
    lifted.add_type("person", {"john", "mary", "bob"})
    lifted.add_type("skill", {"programming", "design"})
    
    # Add formulas and find symmetries
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    reasoner.mln.add_formula("mentor(X,Y) => can_teach(X,Y)", 1.5)
    lifted.find_symmetries(reasoner.mln.clauses)
    
    # Test domain reduction
    query = "can_solve(john,programming)"
    evidence = {
        "expert(john,programming)": True,
        "mentor(john,programming)": True,
        "can_teach(john,programming)": True
    }
    
    reduced_query, reduced_evidence = lifted.reduce_domain(query, evidence)
    assert len(reduced_evidence) < len(evidence)  # Should reduce evidence size

def test_structure_learning(reasoner):
    # Test structure learning from data
    facts = [
        "expert(john,programming)",
        "can_solve(john,programming)",
        "mentor(mary,design)",
        "can_teach(mary,design)",
        "expert(bob,programming)",
        "can_solve(bob,programming)",
        "mentor(john,programming)",
        "can_teach(john,programming)"
    ]
    
    learner = StructureLearner(max_clause_length=2)
    
    # Test embedding learning
    learner.learn_embeddings(facts)
    assert len(learner.predicate_embeddings) == 4  # Four unique predicates
    
    # Test pattern discovery
    clauses = learner.discover_patterns(facts, min_support=0.2)
    assert len(clauses) > 0
    
    # Verify discovered patterns
    found_expert_solve = False
    found_mentor_teach = False
    
    for clause in clauses:
        if "expert" in str(clause) and "can_solve" in str(clause):
            found_expert_solve = True
        if "mentor" in str(clause) and "can_teach" in str(clause):
            found_mentor_teach = True
    
    assert found_expert_solve  # Should discover expert => can_solve pattern
    assert found_mentor_teach  # Should discover mentor => can_teach pattern

def test_structure_learning_with_temporal(reasoner):
    # Test structure learning with temporal aspects
    now = datetime.now()
    facts = [
        "expert(john,programming)",
        "available(john,t1)",
        "can_help(john,programming,t1)",
        "expert(mary,design)",
        "available(mary,t2)",
        "can_help(mary,design,t2)",
        f"time(t1,{now.isoformat()})",
        f"time(t2,{(now + timedelta(hours=1)).isoformat()})"
    ]
    
    learner = StructureLearner(max_clause_length=3)
    clauses = learner.discover_patterns(facts, min_support=0.2)
    
    # Verify temporal patterns
    found_temporal_pattern = False
    for clause in clauses:
        if "available" in str(clause) and "can_help" in str(clause):
            found_temporal_pattern = True
    
    assert found_temporal_pattern  # Should discover temporal relationships

def test_advanced_gibbs_sampling(reasoner):
    # Test Gibbs sampling with different parameters
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    
    evidence = {
        "expert(john,programming)": True
    }
    
    # Test with different burn-in periods
    gibbs1 = GibbsSampler(burn_in=10, sample_gap=5)
    gibbs2 = GibbsSampler(burn_in=100, sample_gap=5)
    
    prob1 = gibbs1.sample(
        reasoner.mln,
        "can_solve(john,programming)",
        evidence,
        num_samples=200
    )
    
    prob2 = gibbs2.sample(
        reasoner.mln,
        "can_solve(john,programming)",
        evidence,
        num_samples=200
    )
    
    # Longer burn-in should give more stable results
    assert abs(prob1 - prob2) < 0.3
    
    # Test sample convergence
    samples = []
    for i in range(5):
        gibbs = GibbsSampler(burn_in=50, sample_gap=5)
        prob = gibbs.sample(
            reasoner.mln,
            "can_solve(john,programming)",
            evidence,
            num_samples=200
        )
        samples.append(prob)
    
    # Results should be relatively stable
    assert max(samples) - min(samples) < 0.3

def test_combined_inference_methods(reasoner):
    # Test combination of different inference methods
    now = datetime.now()
    
    # Add formulas
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    reasoner.mln.add_formula(
        "available(X,T) && expert(X,Y) => can_help(X,Y,T)",
        1.0,
        temporal=True
    )
    
    # Add temporal-probabilistic fact
    fact = TemporalProbabilisticFact(
        predicate="available",
        arguments=["john", "programming"],
        start_time=now,
        end_time=now + timedelta(hours=2),
        alpha=8.0,
        beta=2.0
    )
    reasoner.add_temporal_probabilistic_fact(fact)
    
    # Set up evidence
    evidence = {
        "expert(john,programming)": True,
        "temporal": {
            "start": now,
            "end": now + timedelta(hours=1)
        }
    }
    
    # Compare different inference methods
    gibbs = GibbsSampler(burn_in=50, sample_gap=5)
    prob_gibbs = gibbs.sample(
        reasoner.mln,
        "can_help(john,programming,now)",
        evidence,
        num_samples=500
    )
    
    prob_mcmc = reasoner.mln.infer(
        "can_help(john,programming,now)",
        evidence,
        method="mcmc"
    )
    
    # Results should be similar
    assert abs(prob_gibbs - prob_mcmc) < 0.2
    
    # Test with lifted inference
    lifted = LiftedInference()
    lifted.add_type("person", {"john", "mary", "bob"})
    lifted.add_type("skill", {"programming", "design"})
    lifted.find_symmetries(reasoner.mln.clauses)
    
    # Reduce domain and perform inference
    query = "can_help(john,programming,now)"
    reduced_query, reduced_evidence = lifted.reduce_domain(query, evidence)
    
    prob_lifted = reasoner.mln.infer(
        reduced_query,
        reduced_evidence,
        method="mcmc"
    )
    
    # Lifted inference should give similar results
    assert abs(prob_lifted - prob_mcmc) < 0.2

def test_parallel_mcmc(reasoner):
    # Test parallel MCMC sampling
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    reasoner.mln.add_formula("has_time(X) => available(X)", 1.0)
    
    evidence = {
        "expert(john,programming)": True,
        "has_time(john)": True
    }
    
    # Run parallel inference
    prob = reasoner.mln.infer(
        "can_solve(john,programming)",
        evidence,
        method="parallel_mcmc"
    )
    
    assert 0 <= prob <= 1
    assert prob > 0.5  # Should be likely given the evidence

def test_distributed_exact_inference(reasoner):
    # Test distributed exact inference
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    
    evidence = {
        "expert(john,programming)": True
    }
    
    # Run distributed exact inference
    prob = reasoner.mln.infer(
        "can_solve(john,programming)",
        evidence,
        method="distributed_exact"
    )
    
    assert 0 <= prob <= 1
    assert prob > 0.5  # Should be likely given the evidence

def test_parallel_grounding(reasoner):
    # Test parallel clause grounding
    reasoner.mln.constants.update(["john", "mary", "programming", "design"])
    
    # Add formulas
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    reasoner.mln.add_formula("mentor(X,Y) => can_teach(X,Y)", 1.0)
    
    # Get groundings in parallel
    groundings = reasoner.mln.parallel_inference.parallel_ground_clauses(reasoner.mln)
    
    assert len(groundings) == 2  # Two formulas
    for grounding_list in groundings.values():
        assert len(grounding_list) > 0  # Should have groundings

def test_inference_performance(reasoner):
    # Test performance comparison between sequential and parallel inference
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    reasoner.mln.add_formula("has_time(X) => available(X)", 1.0)
    
    evidence = {
        "expert(john,programming)": True,
        "has_time(john)": True
    }
    
    # Run both methods
    import time
    
    start = time.time()
    prob_seq = reasoner.mln.infer(
        "can_solve(john,programming)",
        evidence,
        method="mcmc"
    )
    seq_time = time.time() - start
    
    start = time.time()
    prob_par = reasoner.mln.infer(
        "can_solve(john,programming)",
        evidence,
        method="parallel_mcmc"
    )
    par_time = time.time() - start
    
    # Results should be similar
    assert abs(prob_seq - prob_par) < 0.2
    # Parallel should be faster for sufficient workload
    # Note: This might not always be true for small problems due to overhead
    logger.info(f"Sequential time: {seq_time:.3f}s, Parallel time: {par_time:.3f}s")

def test_parallel_inference_scaling(reasoner):
    # Test scaling with different numbers of processes
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    
    # Add more constants to increase workload
    reasoner.mln.constants.update([
        f"person{i}" for i in range(10)
    ] + [
        f"skill{i}" for i in range(10)
    ])
    
    evidence = {
        "expert(person0,skill0)": True
    }
    
    # Test with different numbers of processes
    times = []
    process_counts = [1, 2, 4]
    
    for num_processes in process_counts:
        reasoner.mln.parallel_inference = ParallelInference(num_processes=num_processes)
        
        start = time.time()
        prob = reasoner.mln.infer(
            "can_solve(person0,skill0)",
            evidence,
            method="parallel_mcmc"
        )
        times.append(time.time() - start)
        
        assert 0 <= prob <= 1
    
    # More processes should generally be faster
    # Note: This might not always be true due to overhead and problem size
    logger.info(f"Times for {process_counts} processes: {times}")

def test_parallel_inference_with_temporal(reasoner):
    # Test parallel inference with temporal aspects
    now = datetime.now()
    
    reasoner.mln.add_formula(
        "expert(X,Y) && available(X,T) => can_help(X,Y,T)",
        weight=1.5,
        temporal=True
    )
    
    # Add temporal fact
    fact = TemporalProbabilisticFact(
        predicate="available",
        arguments=["john", "programming"],
        start_time=now,
        end_time=now + timedelta(hours=2),
        alpha=8.0,
        beta=2.0
    )
    reasoner.add_temporal_probabilistic_fact(fact)
    
    # Test parallel inference
    evidence = {
        "expert(john,programming)": True,
        "temporal": {
            "start": now,
            "end": now + timedelta(hours=1)
        }
    }
    
    prob = reasoner.mln.infer(
        "can_help(john,programming,now)",
        evidence,
        method="parallel_mcmc"
    )
    
    assert 0 <= prob <= 1
    assert prob > 0.5  # Should be likely given the evidence

def test_parallel_inference_error_handling(reasoner):
    # Test error handling in parallel inference
    reasoner.mln.add_formula("expert(X,Y) => can_solve(X,Y)", 1.5)
    
    # Test with invalid evidence
    with pytest.raises(Exception):
        reasoner.mln.infer(
            "can_solve(john,programming)",
            {"invalid_predicate(x,y)": True},
            method="parallel_mcmc"
        )
    
    # Test with too large domain for exact inference
    reasoner.mln.constants.update([f"x{i}" for i in range(100)])
    
    # Should fall back to MCMC
    prob = reasoner.mln.infer(
        "can_solve(x0,x1)",
        {"expert(x0,x1)": True},
        method="distributed_exact"
    )
    
    assert 0 <= prob <= 1