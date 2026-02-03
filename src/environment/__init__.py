"""Environment module for Hidden Value Game simulation.

This module provides the core components for running multi-agent games
where agents with conflicting interests try to influence an observer.
"""

from .world_v2 import (
    WorldV2,
    Object,
    Property,
    PropertyType,
    ValueRule,
    ValueCondition,
    DEFAULT_PROPERTIES_V2,
    generate_world,
    create_simple_rule,
    create_medium_rule,
    create_complex_rule,
)
from .agent_v2 import (
    AgentV2,
    AgentInterest,
    StatementV2,
    create_conflicting_agents,
    create_multi_agent_game,
)
from .simulation_v2 import (
    HiddenValueGame,
    GameConfig,
    GameResult,
    GameRound,
    OracleQuery,
    RoundMetrics,
    run_game,
)
from .estimator_v2 import EstimatorV2

__all__ = [
    # World
    "WorldV2",
    "Object",
    "Property",
    "PropertyType",
    "ValueRule",
    "ValueCondition",
    "DEFAULT_PROPERTIES_V2",
    "generate_world",
    "create_simple_rule",
    "create_medium_rule",
    "create_complex_rule",
    # Agents
    "AgentV2",
    "AgentInterest",
    "StatementV2",
    "create_conflicting_agents",
    "create_multi_agent_game",
    # Simulation
    "HiddenValueGame",
    "GameConfig",
    "GameResult",
    "GameRound",
    "OracleQuery",
    "RoundMetrics",
    "run_game",
    # Estimator
    "EstimatorV2",
]
