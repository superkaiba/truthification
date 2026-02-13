#!/usr/bin/env python3
"""Generate sample games with full transcripts for the trajectory viewer."""

import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.simulation import GameConfig, HiddenValueGame

# Run a few sample games with different configurations
SAMPLE_CONFIGS = [
    {"name": "2 agents, 5 rounds", "n_agents": 2, "n_rounds": 5, "seed": 42},
    {"name": "3 agents, 5 rounds", "n_agents": 3, "n_rounds": 5, "seed": 42},
]


def format_statement(stmt):
    """Format a statement for display."""
    if isinstance(stmt, dict):
        agent_id = stmt.get("agent_id", "System")
        text = stmt.get("text", "")
        stmt_type = stmt.get("type", "statement")
        return {"agent": agent_id, "text": text, "type": stmt_type}
    else:
        # Statement object
        return {"agent": stmt.agent_id, "text": stmt.text, "type": "statement"}


def run_sample_game(config_dict):
    """Run a single game and extract transcript."""
    print(f"Running: {config_dict['name']}...", flush=True)

    config = GameConfig(
        n_objects=10,
        n_agents=config_dict["n_agents"],
        n_rounds=config_dict["n_rounds"],
        oracle_budget=8,
        selection_size=5,
        condition="interests",
        use_agent_value_functions=True,
        agent_value_function_complexity="medium",
        rule_complexity="medium",
        seed=config_dict["seed"],
    )

    game = HiddenValueGame(config)
    result = game.run()

    # Format rounds for display
    # result.rounds can contain GameRound dataclass objects or dicts
    formatted_rounds = []
    for round_obj in result.rounds:
        # Handle both dict and object formats
        is_dict = isinstance(round_obj, dict)

        def get_attr(obj, attr, default=None):
            if is_dict:
                return obj.get(attr, default)
            return getattr(obj, attr, default)

        formatted_round = {
            "round_number": get_attr(round_obj, "round_number"),
            "statements": [format_statement(s) for s in (get_attr(round_obj, "agent_statements") or [])],
            "observer_action": get_attr(round_obj, "observer_action", ""),
            "picks": get_attr(round_obj, "observer_current_picks") or [],
            "reasoning": get_attr(round_obj, "observer_reasoning", ""),
        }

        # Add oracle query if present
        oracle = get_attr(round_obj, "oracle_query")
        if oracle:
            if isinstance(oracle, dict):
                formatted_round["oracle"] = {
                    "object": oracle.get("object_id", ""),
                    "type": oracle.get("query_type", ""),
                    "property": oracle.get("property_name", ""),
                    "result": str(oracle.get("result", "")),
                }
            else:
                formatted_round["oracle"] = {
                    "object": oracle.object_id,
                    "type": oracle.query_type,
                    "property": oracle.property_name,
                    "result": str(oracle.result),
                }

        # Add round metrics if present
        metrics = get_attr(round_obj, "round_metrics")
        if metrics:
            if isinstance(metrics, dict):
                formatted_round["metrics"] = {
                    "picks_value": metrics.get("picks_value", 0),
                    "decision_quality": metrics.get("decision_quality", 0),
                    "cumulative_value": metrics.get("cumulative_value", 0),
                }
            else:
                formatted_round["metrics"] = {
                    "picks_value": metrics.picks_value,
                    "decision_quality": metrics.decision_quality,
                    "cumulative_value": metrics.cumulative_value,
                }

        formatted_rounds.append(formatted_round)

    # Format agents
    formatted_agents = []
    for agent in result.agents:
        formatted_agents.append({
            "id": agent.get("id", ""),
            "interest": agent.get("interest", {}).get("description", ""),
            "value_function": agent.get("value_function", {}).get("description", "") if agent.get("value_function") else "",
        })

    # Extract ground truth - object properties
    world_state = result.world_state
    objects_ground_truth = {}
    if isinstance(world_state, dict) and "objects" in world_state:
        for obj_id, obj_data in world_state["objects"].items():
            objects_ground_truth[obj_id] = obj_data.get("properties", {})

    # Extract value rule details
    value_rule_data = result.value_rule
    if isinstance(value_rule_data, dict):
        value_rule_info = {
            "description": value_rule_data.get("description", ""),
            "conditions": value_rule_data.get("conditions", []),
        }
    else:
        value_rule_info = {"description": str(value_rule_data), "conditions": []}

    # Extract per-round accuracy and agent values from round metrics
    accuracy_over_rounds = []
    agent_value_over_rounds = {agent["id"]: [] for agent in formatted_agents}

    for round_obj in result.rounds:
        is_dict = isinstance(round_obj, dict)

        def get_attr(obj, attr, default=None):
            if is_dict:
                return obj.get(attr, default)
            return getattr(obj, attr, default)

        metrics = get_attr(round_obj, "round_metrics")
        if metrics:
            if isinstance(metrics, dict):
                accuracy_over_rounds.append({
                    "round": get_attr(round_obj, "round_number"),
                    "judge_property_accuracy": metrics.get("judge_property_accuracy", 0),
                    "estimator_property_accuracy": metrics.get("estimator_property_accuracy", 0),
                    "cumulative_value": metrics.get("cumulative_value", 0),
                })
                # Agent cumulative values
                agent_cum = metrics.get("agent_cumulative_value", {})
                for agent_id, value in agent_cum.items():
                    if agent_id in agent_value_over_rounds:
                        agent_value_over_rounds[agent_id].append(value)
            else:
                accuracy_over_rounds.append({
                    "round": get_attr(round_obj, "round_number"),
                    "judge_property_accuracy": getattr(metrics, "judge_property_accuracy", 0),
                    "estimator_property_accuracy": getattr(metrics, "estimator_property_accuracy", 0),
                    "cumulative_value": metrics.cumulative_value,
                })
                # Agent cumulative values from metrics
                agent_cum = getattr(metrics, "agent_cumulative_value", {})
                for agent_id, value in agent_cum.items():
                    if agent_id in agent_value_over_rounds:
                        agent_value_over_rounds[agent_id].append(value)

    # Also get agent cumulative values from final metrics if not captured per-round
    final_agent_values = result.metrics.get("agent_cumulative_value_per_round", {})
    if final_agent_values:
        agent_value_over_rounds = final_agent_values

    return {
        "name": config_dict["name"],
        "config": {
            "n_agents": config_dict["n_agents"],
            "n_rounds": config_dict["n_rounds"],
            "seed": config_dict["seed"],
        },
        "agents": formatted_agents,
        "value_rule": value_rule_info,
        "ground_truth": objects_ground_truth,
        "rounds": formatted_rounds,
        "metrics": {
            "property_accuracy": result.metrics.get("property_accuracy", 0),
            "total_value": result.metrics.get("total_value", 0),
            "rule_inference_accuracy": result.metrics.get("rule_inference_accuracy", 0),
        },
        "accuracy_over_rounds": accuracy_over_rounds,
        "agent_value_over_rounds": agent_value_over_rounds,
        "value_per_round": result.metrics.get("value_per_round", []),
        "cumulative_value_per_round": result.metrics.get("cumulative_value_per_round", []),
        "final_selection": result.final_selection,
    }


def main():
    print("Generating sample trajectories for dashboard...")

    samples = []
    for config in SAMPLE_CONFIGS:
        try:
            sample = run_sample_game(config)
            samples.append(sample)
            print(f"  Done: {sample['metrics']}")
        except Exception as e:
            print(f"  Error: {e}")

    # Save to docs folder for dashboard
    output_path = Path("docs/sample_trajectories.json")
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2, default=str)

    print(f"\nSaved {len(samples)} trajectories to {output_path}")


if __name__ == "__main__":
    main()
