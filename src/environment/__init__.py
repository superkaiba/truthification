"""Environment module for Hidden Value Game simulation.

This module provides the core components for running multi-agent games
where agents with conflicting interests try to influence an observer.
"""

from .world import (
    World,
    Object,
    Property,
    PropertyType,
    ValueRule,
    ValueCondition,
    DEFAULT_PROPERTIES,
    generate_world,
    create_simple_rule,
    create_medium_rule,
    create_complex_rule,
)
from .agent import (
    Agent,
    AgentInterest,
    AgentValueFunction,
    Statement,
    ValueRuleClaim,
    create_conflicting_agents,
    create_multi_agent_game,
    create_agents_with_value_functions,
    generate_agent_value_function,
)
from .simulation import (
    HiddenValueGame,
    GameConfig,
    GameResult,
    GameRound,
    OracleQuery,
    RoundMetrics,
    run_game,
)
from .estimator import Estimator

__all__ = [
    # World
    "World",
    "Object",
    "Property",
    "PropertyType",
    "ValueRule",
    "ValueCondition",
    "DEFAULT_PROPERTIES",
    "generate_world",
    "create_simple_rule",
    "create_medium_rule",
    "create_complex_rule",
    # Agents
    "Agent",
    "AgentInterest",
    "AgentValueFunction",
    "Statement",
    "ValueRuleClaim",
    "create_conflicting_agents",
    "create_multi_agent_game",
    "create_agents_with_value_functions",
    "generate_agent_value_function",
    # Simulation
    "HiddenValueGame",
    "GameConfig",
    "GameResult",
    "GameRound",
    "OracleQuery",
    "RoundMetrics",
    "run_game",
    # Estimator
    "Estimator",
]
