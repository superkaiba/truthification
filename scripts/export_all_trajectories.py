#!/usr/bin/env python3
"""
Export all game trajectories to the format expected by the trajectory viewer.
"""

import json
from pathlib import Path
from datetime import datetime


def convert_game_to_trajectory(game_data: dict, name: str) -> dict:
    """Convert a game JSON to trajectory viewer format."""

    # Extract config
    config = game_data.get("config", {})

    # Extract agents
    agents = []
    for agent in game_data.get("agents", []):
        agents.append({
            "id": agent.get("id", "Unknown"),
            "interest": agent.get("interest", {}).get("description", "Unknown"),
            "value_function": agent.get("value_function", {}).get("description", "")
                if agent.get("value_function") else f"Values objects that are: {agent.get('interest', {}).get('description', 'Unknown')}"
        })

    # Extract value rule
    value_rule_data = game_data.get("value_rule", {})
    value_rule = {
        "description": value_rule_data.get("description", "Unknown rule"),
        "conditions": value_rule_data.get("conditions", [])
    }

    # Extract ground truth (object properties)
    ground_truth = {}
    world_state = game_data.get("world_state", {})
    for obj_id, obj_data in world_state.get("objects", {}).items():
        ground_truth[obj_id] = obj_data.get("properties", {})

    # Extract rounds
    rounds = []
    for round_data in game_data.get("rounds", []):
        round_num = round_data.get("round_number", 0)

        # Extract statements - field is "agent_statements" in raw data
        statements = []
        raw_statements = round_data.get("agent_statements", []) or round_data.get("statements", [])
        for stmt in raw_statements:
            statements.append({
                "agent": stmt.get("agent_id", stmt.get("agent", "Unknown")),
                "text": stmt.get("text", ""),
                "about": stmt.get("object_id", ""),
                "claim": stmt.get("claimed_value", "")
            })

        # Extract oracle queries - can be single query or list
        oracle_queries = []
        oracle_query = round_data.get("oracle_query")
        if oracle_query:
            if isinstance(oracle_query, list):
                for query in oracle_query:
                    oracle_queries.append({
                        "object": query.get("object_id", ""),
                        "property": query.get("property", ""),
                        "result": str(query.get("result", ""))
                    })
            elif isinstance(oracle_query, dict) and oracle_query.get("object_id"):
                oracle_queries.append({
                    "object": oracle_query.get("object_id", ""),
                    "property": oracle_query.get("property", ""),
                    "result": str(oracle_query.get("result", ""))
                })

        # Extract picks/selection
        selection = round_data.get("observer_current_picks", [])
        if not selection:
            selection = round_data.get("selection", [])
        if not selection:
            selection = round_data.get("picks", [])

        # Extract observer reasoning
        observer_reasoning = round_data.get("observer_reasoning", "")

        rounds.append({
            "round_number": round_num,
            "statements": statements,
            "observer_action": "query" if oracle_queries else "analyze",
            "picks": selection,
            "oracle_queries": oracle_queries,
            "reasoning": observer_reasoning
        })

    # Extract metrics
    metrics = game_data.get("metrics", {})

    return {
        "name": name,
        "config": {
            "n_agents": config.get("n_agents", len(agents)),
            "n_rounds": config.get("n_rounds", len(rounds)),
            "seed": config.get("seed", 0),
            "condition": config.get("condition", "unknown"),
            "oracle_budget": config.get("oracle_budget", 0)
        },
        "agents": agents,
        "value_rule": value_rule,
        "ground_truth": ground_truth,
        "rounds": rounds,
        "metrics": {
            "property_accuracy": metrics.get("property_accuracy", 0),
            "total_value": metrics.get("total_value", 0),
            "optimal_value": metrics.get("optimal_value", 0),
            "oracle_queries_used": metrics.get("oracle_queries_used", 0)
        }
    }


def find_all_game_files() -> list[tuple[Path, str]]:
    """Find all game JSON files and generate names for them."""
    games = []

    # Debate structure test games
    debate_dir = Path("outputs/debate_structure_test")
    if debate_dir.exists():
        for run_dir in sorted(debate_dir.iterdir()):
            if run_dir.is_dir():
                timestamp = run_dir.name
                for game_file in sorted(run_dir.glob("game_*.json")):
                    # Extract structure from filename like game_interleaved_after_statements.json
                    parts = game_file.stem.replace("game_", "").split("_")
                    structure = parts[0] if parts else "unknown"
                    timing = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
                    name = f"debate_{structure}_{timing}_{timestamp[:8]}"
                    games.append((game_file, name))

    # Scale experiment games
    scale_dir = Path("results/scale_experiment")
    if scale_dir.exists():
        for result_file in scale_dir.glob("results_*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                all_results = data.get("all_results", [])
                for i, result in enumerate(all_results):
                    n_agents = result.get("n_agents", 0)
                    n_rounds = result.get("n_rounds", 0)
                    seed = result.get("seed", 0)
                    name = f"scale_{n_agents}a_{n_rounds}r_seed{seed}"
                    # Store the result directly with a marker
                    games.append((result, name))
            except Exception as e:
                print(f"Error reading {result_file}: {e}")

    # Multi-factor games from trajectory data
    multi_factor_traj = Path("results/multi_factor/trajectory_data.json")
    if multi_factor_traj.exists():
        try:
            with open(multi_factor_traj) as f:
                data = json.load(f)
            for condition_name, condition_data in data.items():
                name = f"multi_{condition_name}"
                games.append((condition_data, name))
        except Exception as e:
            print(f"Error reading {multi_factor_traj}: {e}")

    # Forced oracle test games
    forced_oracle_dir = Path("results/forced_oracle_test")
    if forced_oracle_dir.exists():
        for result_file in forced_oracle_dir.glob("results_*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                # This has a different structure - just summary data
                # Skip for now as it doesn't have full trajectories
            except Exception as e:
                print(f"Error reading {result_file}: {e}")

    return games


def main():
    print("Exporting all trajectories for viewer...")

    all_trajectories = []
    game_files = find_all_game_files()

    print(f"Found {len(game_files)} potential games")

    for item, name in game_files:
        try:
            if isinstance(item, Path):
                with open(item) as f:
                    game_data = json.load(f)
            else:
                # Already loaded data (from scale experiment)
                game_data = item

            trajectory = convert_game_to_trajectory(game_data, name)

            # Only add if we have actual round data
            if trajectory["rounds"] and len(trajectory["rounds"]) > 0:
                all_trajectories.append(trajectory)
                print(f"  Added: {name} ({len(trajectory['rounds'])} rounds)")
            else:
                print(f"  Skipped (no rounds): {name}")

        except Exception as e:
            print(f"  Error processing {name}: {e}")

    # Sort by name
    all_trajectories.sort(key=lambda x: x["name"])

    # Write output
    output_path = Path("docs/sample_trajectories.json")
    with open(output_path, "w") as f:
        json.dump(all_trajectories, f, indent=2)

    # Calculate size
    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\nExported {len(all_trajectories)} trajectories to {output_path}")
    print(f"File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
