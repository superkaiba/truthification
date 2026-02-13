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
    formatted_rounds = []
    for round_data in result.rounds:
        formatted_round = {
            "round_number": round_data["round_number"],
            "statements": [format_statement(s) for s in round_data.get("agent_statements", [])],
            "observer_action": round_data.get("observer_action", ""),
            "picks": round_data.get("observer_current_picks", []) or [],
            "reasoning": round_data.get("observer_reasoning", ""),
        }

        # Add oracle query if present
        oracle = round_data.get("oracle_query")
        if oracle:
            formatted_round["oracle"] = {
                "object": oracle.get("object_id", ""),
                "type": oracle.get("query_type", ""),
                "property": oracle.get("property_name", ""),
                "result": str(oracle.get("result", "")),
            }

        # Add round metrics if present
        metrics = round_data.get("round_metrics")
        if metrics:
            formatted_round["metrics"] = {
                "picks_value": metrics.get("picks_value", 0),
                "decision_quality": metrics.get("decision_quality", 0),
                "cumulative_value": metrics.get("cumulative_value", 0),
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

    return {
        "name": config_dict["name"],
        "config": {
            "n_agents": config_dict["n_agents"],
            "n_rounds": config_dict["n_rounds"],
            "seed": config_dict["seed"],
        },
        "agents": formatted_agents,
        "value_rule": result.value_rule.get("description", "") if isinstance(result.value_rule, dict) else str(result.value_rule),
        "rounds": formatted_rounds,
        "metrics": {
            "property_accuracy": result.metrics.get("property_accuracy", 0),
            "total_value": result.metrics.get("total_value", 0),
            "rule_inference_accuracy": result.metrics.get("rule_inference_accuracy", 0),
        },
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
