#!/usr/bin/env python3
"""Post-hoc Strategy Annotation Script.

This script runs the strategy classifier on existing game trajectories
to annotate agent manipulation tactics.

Usage:
    python scripts/annotate_strategies.py [--input-dir DIR] [--output-dir DIR]

The script will:
1. Find all game trajectory files in the input directory
2. Run strategy classification on each game
3. Save annotations alongside the original files
4. Generate aggregate statistics and visualizations
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import wandb
from src.environment.strategy_classifier import (
    StrategyClassifier,
    GameStrategyAnalysis,
    STRATEGY_DESCRIPTIONS,
)

# Default directories to search for trajectories
DEFAULT_TRAJECTORY_DIRS = [
    "outputs/multi_factor",
    "outputs/search_space",
    "outputs/oracle_budget_objective",
    "outputs/complexity_objective",
    "results",
]


def find_trajectory_files(base_dirs: list[str | Path]) -> list[Path]:
    """Find all game trajectory JSON files."""
    files = []

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            continue

        # Look for game_*.json files
        for pattern in ["**/game_*.json", "**/trajectory_*.json", "**/*_trajectory.json"]:
            files.extend(base_path.glob(pattern))

    # Deduplicate
    files = list(set(files))

    # Filter out non-game files (like stats files)
    files = [
        f for f in files
        if "stats" not in f.name.lower()
        and "summary" not in f.name.lower()
        and "annotation" not in f.name.lower()
    ]

    return sorted(files)


def load_game_result(filepath: Path) -> dict | None:
    """Load a game result from JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)

        # Validate it has expected fields
        if "rounds" not in data and "agent_statements" not in data:
            return None

        return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Warning: Could not load {filepath}: {e}")
        return None


def annotate_game(
    classifier: StrategyClassifier,
    game_data: dict,
    game_id: str,
) -> GameStrategyAnalysis | None:
    """Annotate a single game with strategy classifications."""
    try:
        analysis = classifier.analyze_game(game_data, game_id)
        return analysis
    except Exception as e:
        print(f"  Error annotating game {game_id}: {e}")
        return None


def save_annotation(
    analysis: GameStrategyAnalysis,
    output_path: Path,
):
    """Save annotation alongside original file."""
    with open(output_path, "w") as f:
        json.dump(analysis.to_dict(), f, indent=2)


def compute_aggregate_stats(analyses: list[GameStrategyAnalysis]) -> dict:
    """Compute aggregate statistics across all games."""
    if not analyses:
        return {}

    strategy_names = [
        "fabricated_terminology", "truth_mixed_with_lies", "oracle_spin",
        "credibility_attack", "escalating_complexity", "object_advocacy"
    ]

    # Collect all values per strategy
    strategy_values = {name: [] for name in strategy_names}
    primary_strategies = []
    dominant_dynamics = []

    for analysis in analyses:
        for annotation in analysis.agent_annotations:
            for name in strategy_names:
                strategy_values[name].append(getattr(annotation, name, 0))
            if annotation.primary_strategy:
                primary_strategies.append(annotation.primary_strategy)

        if analysis.dominant_dynamic:
            dominant_dynamics.append(analysis.dominant_dynamic)

    # Compute statistics
    stats = {
        "n_games": len(analyses),
        "n_agent_annotations": sum(len(a.agent_annotations) for a in analyses),
        "strategy_stats": {},
        "primary_strategy_counts": {},
        "dominant_dynamic_counts": {},
    }

    for name in strategy_names:
        values = strategy_values[name]
        if values:
            stats["strategy_stats"][name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "n_high_confidence": sum(1 for v in values if v >= 50),
                "prevalence": sum(1 for v in values if v >= 50) / len(values),
            }

    # Count primary strategies
    for strat in primary_strategies:
        stats["primary_strategy_counts"][strat] = stats["primary_strategy_counts"].get(strat, 0) + 1

    # Count dynamics
    for dyn in dominant_dynamics:
        stats["dominant_dynamic_counts"][dyn] = stats["dominant_dynamic_counts"].get(dyn, 0) + 1

    return stats


def create_summary_markdown(
    stats: dict,
    output_dir: Path,
    runtime: float,
) -> str:
    """Create markdown summary of annotation results."""
    lines = [
        "# Agent Strategy Annotation Results",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {runtime/60:.1f} minutes",
        f"**Games Analyzed**: {stats.get('n_games', 0)}",
        f"**Agent Annotations**: {stats.get('n_agent_annotations', 0)}",
        "",
        "## Strategy Definitions",
        "",
    ]

    for name, desc in STRATEGY_DESCRIPTIONS.items():
        lines.append(f"- **{name.replace('_', ' ').title()}**: {desc}")

    lines.extend([
        "",
        "## Strategy Prevalence",
        "",
        "| Strategy | Mean Confidence | Prevalence (>=50) | High Confidence Count |",
        "|----------|-----------------|-------------------|----------------------|",
    ])

    strategy_stats = stats.get("strategy_stats", {})
    for name in sorted(strategy_stats.keys(), key=lambda n: strategy_stats[n]["prevalence"], reverse=True):
        s = strategy_stats[name]
        lines.append(
            f"| {name.replace('_', ' ').title()} | {s['mean']:.1f} | "
            f"{s['prevalence']*100:.1f}% | {s['n_high_confidence']} |"
        )

    lines.extend([
        "",
        "## Primary Strategy Distribution",
        "",
        "| Strategy | Count | Percentage |",
        "|----------|-------|------------|",
    ])

    primary_counts = stats.get("primary_strategy_counts", {})
    total_primary = sum(primary_counts.values())
    for strat, count in sorted(primary_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_primary * 100 if total_primary else 0
        lines.append(f"| {strat} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Dominant Dynamics",
        "",
        "| Dynamic | Count |",
        "|---------|-------|",
    ])

    dynamic_counts = stats.get("dominant_dynamic_counts", {})
    for dyn, count in sorted(dynamic_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {dyn} | {count} |")

    lines.extend([
        "",
        "## Key Findings",
        "",
        "(Analysis to be added after reviewing results)",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Annotate agent strategies in game trajectories")
    parser.add_argument(
        "--input-dir",
        type=str,
        nargs="*",
        default=DEFAULT_TRAJECTORY_DIRS,
        help="Directories to search for trajectory files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/strategy_annotations",
        help="Directory to save annotations",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use for classification",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to annotate (for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip games that already have annotations",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Agent Strategy Annotation")
    print(f"{'='*70}")
    print(f"Input directories: {args.input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")

    # Find trajectory files
    print("\nSearching for trajectory files...")
    files = find_trajectory_files(args.input_dir)
    print(f"Found {len(files)} trajectory files")

    if args.max_games:
        files = files[:args.max_games]
        print(f"Limiting to {args.max_games} games")

    if not files:
        print("No trajectory files found!")
        return

    # Initialize classifier
    classifier = StrategyClassifier(model=args.model)

    # Initialize wandb
    wandb.init(
        project="truthification",
        name=f"strategy-annotation-{timestamp}",
        config={
            "experiment": "strategy_annotation",
            "model": args.model,
            "n_files": len(files),
            "input_dirs": args.input_dir,
        },
    )

    # Process files
    print(f"\nProcessing {len(files)} games...")
    analyses = []
    start_time = time.time()

    for i, filepath in enumerate(files):
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (len(files) - i - 1) if i > 0 else 0

        print(f"[{i+1}/{len(files)}] {filepath.name} (ETA: {eta/60:.1f}m)...", end=" ", flush=True)

        # Check for existing annotation
        annotation_path = output_dir / f"{filepath.stem}_annotation.json"
        if args.skip_existing and annotation_path.exists():
            print("skipped (exists)")
            continue

        # Load game data
        game_data = load_game_result(filepath)
        if not game_data:
            print("skipped (invalid)")
            continue

        # Generate game ID from filename
        game_id = filepath.stem

        # Annotate
        game_start = time.time()
        analysis = annotate_game(classifier, game_data, game_id)
        game_elapsed = time.time() - game_start

        if analysis:
            analyses.append(analysis)
            save_annotation(analysis, annotation_path)

            # Log to wandb
            wandb.log({
                "game_number": i + 1,
                "game_id": game_id,
                "processing_time": game_elapsed,
                **{
                    f"strategy_{name}": analysis.strategy_distribution.get(name, 0)
                    for name in ["fabricated_terminology", "truth_mixed_with_lies",
                                "oracle_spin", "credibility_attack",
                                "escalating_complexity", "object_advocacy"]
                },
            })

            # Quick summary
            top_strategy = max(
                analysis.strategy_distribution.items(),
                key=lambda x: x[1],
                default=("none", 0)
            )
            print(f"done ({game_elapsed:.0f}s) - Top: {top_strategy[0]} ({top_strategy[1]:.0f})")
        else:
            print("failed")

    total_elapsed = time.time() - start_time

    # Compute aggregate stats
    print(f"\n{'='*70}")
    print("Computing Aggregate Statistics...")
    print(f"{'='*70}")

    stats = compute_aggregate_stats(analyses)

    # Print summary
    print(f"\nProcessed {stats.get('n_games', 0)} games, {stats.get('n_agent_annotations', 0)} agent annotations")
    print("\nStrategy Prevalence (confidence >= 50):")
    for name, s in sorted(
        stats.get("strategy_stats", {}).items(),
        key=lambda x: x[1]["prevalence"],
        reverse=True
    ):
        print(f"  {name.replace('_', ' ').title()}: {s['prevalence']*100:.1f}%")

    # Save aggregate stats
    with open(output_dir / "aggregate_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Create summary markdown
    summary_md = create_summary_markdown(stats, output_dir, total_elapsed)
    with open(output_dir / "README.md", "w") as f:
        f.write(summary_md)

    # Log final stats to wandb
    wandb.log({
        "total_games_processed": stats.get("n_games", 0),
        "total_annotations": stats.get("n_agent_annotations", 0),
        "total_runtime_minutes": total_elapsed / 60,
        **{
            f"prevalence_{name}": s["prevalence"]
            for name, s in stats.get("strategy_stats", {}).items()
        },
    })

    wandb.finish()

    print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
