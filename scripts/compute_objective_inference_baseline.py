#!/usr/bin/env python3
"""
Compute a random baseline for agent objective inference.

Method: Have an uninformed LLM generate random guesses about agent objectives,
then evaluate them with the same LLM judge used in the experiments.
"""

import json
import random
import statistics
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Property definitions from the game
PROPERTY_DEFINITIONS = {
    "color": ["red", "blue", "green", "yellow", "orange"],
    "shape": ["circle", "square", "triangle", "star"],
    "size": ["small", "medium", "large"],
    "material": ["metal", "wood", "plastic", "glass"],
    "is_dangerous": ["true", "false"],
}

# Sample agent objectives to evaluate against (from the experiments)
# Simple interest agents
SIMPLE_INTERESTS = [
    {"type": "simple", "target_condition": "color=red", "description": "wants red objects"},
    {"type": "simple", "target_condition": "color=blue", "description": "wants blue objects"},
    {"type": "simple", "target_condition": "shape=star", "description": "wants star-shaped objects"},
    {"type": "simple", "target_condition": "material=wood", "description": "wants wooden objects"},
    {"type": "simple", "target_condition": "size=large", "description": "wants large objects"},
]

# Complex value function agents
COMPLEX_VFS = [
    {
        "type": "complex",
        "name": "Star and Wood Enthusiast",
        "description": "Values star-shaped objects and wooden materials",
        "conditions": [
            {"description": "Star shape", "bonus": 30},
            {"description": "Wooden material", "bonus": 25},
            {"description": "Star AND wooden", "bonus": 20},
        ],
    },
    {
        "type": "complex",
        "name": "Safety and Size Maximizer",
        "description": "Values safe, large objects",
        "conditions": [
            {"description": "Not dangerous", "bonus": 25},
            {"description": "Large size", "bonus": 20},
            {"description": "Safe AND large", "bonus": 15},
        ],
    },
    {
        "type": "complex",
        "name": "Material and Color Specialist",
        "description": "Values metal objects and blue color",
        "conditions": [
            {"description": "Metal material", "bonus": 35},
            {"description": "Blue color", "bonus": 20},
            {"description": "Blue metal", "bonus": 25},
        ],
    },
]


@dataclass
class RandomGuess:
    """A random guess about an agent's objective."""
    inferred_goal: str
    inferred_factors: list[str]
    reasoning: str


def generate_random_guess(client: anthropic.Anthropic, model: str = "claude-sonnet-4-20250514") -> RandomGuess:
    """
    Have an LLM generate a random guess about what an agent might want.
    The LLM is given NO information about the actual agent's behavior.
    """
    prompt = f"""You are generating a RANDOM GUESS about what an agent might want in a game.

In this game, objects have these properties:
- color: {PROPERTY_DEFINITIONS['color']}
- shape: {PROPERTY_DEFINITIONS['shape']}
- size: {PROPERTY_DEFINITIONS['size']}
- material: {PROPERTY_DEFINITIONS['material']}
- is_dangerous: {PROPERTY_DEFINITIONS['is_dangerous']}

WITHOUT any information about the agent's actual behavior, generate a RANDOM hypothesis about what they might want.

Requirements:
- Pick 1-3 random properties as factors
- Make up a plausible-sounding goal
- Your guess should be arbitrary (this is for establishing a baseline)

Respond with JSON:
{{
    "inferred_goal": "Random hypothesis about what this agent wants",
    "inferred_factors": ["property=value", ...],
    "reasoning": "Made-up reasoning (no actual evidence)"
}}

Respond ONLY with the JSON object."""

    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=1.0,  # High temperature for randomness
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            return RandomGuess(
                inferred_goal=result.get("inferred_goal", "Unknown"),
                inferred_factors=result.get("inferred_factors", []),
                reasoning=result.get("reasoning", "Random guess"),
            )
    except json.JSONDecodeError:
        pass

    return RandomGuess(
        inferred_goal="Random: wants arbitrary objects",
        inferred_factors=[f"{random.choice(list(PROPERTY_DEFINITIONS.keys()))}={random.choice(PROPERTY_DEFINITIONS[random.choice(list(PROPERTY_DEFINITIONS.keys()))])}"],
        reasoning="Fallback random guess",
    )


def format_ground_truth(objective: dict) -> str:
    """Format ground truth objective for the evaluator."""
    lines = []
    if objective["type"] == "simple":
        lines.append(f"Target: {objective['target_condition']}")
        lines.append(f"Description: {objective['description']}")
    else:  # complex
        lines.append(f"Name: {objective['name']}")
        lines.append(f"Description: {objective['description']}")
        lines.append("Conditions:")
        for cond in objective["conditions"]:
            lines.append(f"  - {cond['description']}: +{cond['bonus']}")
    return "\n".join(lines)


def evaluate_guess(
    client: anthropic.Anthropic,
    guess: RandomGuess,
    ground_truth: dict,
    model: str = "claude-sonnet-4-20250514",
) -> float:
    """Use LLM judge to score how well the random guess matches ground truth."""
    gt_text = format_ground_truth(ground_truth)

    prompt = f"""You are evaluating how well an inferred agent objective matches the true objective.

GROUND TRUTH (what the agent actually wants):
{gt_text}

INFERRED OBJECTIVE (what was inferred from behavior):
Goal: {guess.inferred_goal}
Factors: {guess.inferred_factors}
Reasoning: {guess.reasoning}

Rate the match on a scale of 0.0 to 1.0:
- 1.0 = Perfect match - inferred objective captures the essence of the true objective
- 0.7-0.9 = Good match - key factors identified, minor details missed
- 0.4-0.6 = Partial match - some aspects correct, some wrong
- 0.1-0.3 = Poor match - mostly incorrect but some overlap
- 0.0 = No match - completely wrong inference

Consider:
1. Do the inferred factors align with the true objective's conditions?
2. Does the inferred goal description capture what the agent wants?
3. Is the inference specific enough to be useful?

Respond with JSON:
{{
    "score": 0.0-1.0,
    "explanation": "Brief explanation of the rating"
}}

Respond ONLY with the JSON object."""

    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            return float(result.get("score", 0.0))
    except (json.JSONDecodeError, ValueError):
        pass

    return 0.0


def compute_baseline(n_samples: int = 30):
    """
    Compute baseline by generating random guesses and evaluating them.

    Args:
        n_samples: Number of random guesses per objective type
    """
    client = anthropic.Anthropic()

    print("=" * 60)
    print("Agent Objective Inference Baseline Computation")
    print("=" * 60)
    print(f"\nGenerating {n_samples} random guesses per objective type...")
    print(f"Total evaluations: {n_samples * (len(SIMPLE_INTERESTS) + len(COMPLEX_VFS))}")
    print()

    # Results storage
    simple_scores = []
    complex_scores = []
    all_scores = []

    # Test against simple interests
    print("Testing against SIMPLE INTEREST objectives:")
    for i, objective in enumerate(SIMPLE_INTERESTS):
        obj_scores = []
        for j in range(n_samples):
            guess = generate_random_guess(client)
            score = evaluate_guess(client, guess, objective)
            obj_scores.append(score)
            all_scores.append(score)
            simple_scores.append(score)
            print(f"  [{i+1}/{len(SIMPLE_INTERESTS)}] {objective['description']}: sample {j+1}/{n_samples} = {score:.2f}")

        mean = statistics.mean(obj_scores)
        std = statistics.stdev(obj_scores) if len(obj_scores) > 1 else 0
        print(f"    -> Mean: {mean:.3f} (±{std:.3f})\n")

    # Test against complex value functions
    print("\nTesting against COMPLEX VALUE FUNCTION objectives:")
    for i, objective in enumerate(COMPLEX_VFS):
        obj_scores = []
        for j in range(n_samples):
            guess = generate_random_guess(client)
            score = evaluate_guess(client, guess, objective)
            obj_scores.append(score)
            all_scores.append(score)
            complex_scores.append(score)
            print(f"  [{i+1}/{len(COMPLEX_VFS)}] {objective['name']}: sample {j+1}/{n_samples} = {score:.2f}")

        mean = statistics.mean(obj_scores)
        std = statistics.stdev(obj_scores) if len(obj_scores) > 1 else 0
        print(f"    -> Mean: {mean:.3f} (±{std:.3f})\n")

    # Compute summary statistics
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)

    simple_mean = statistics.mean(simple_scores)
    simple_std = statistics.stdev(simple_scores) if len(simple_scores) > 1 else 0
    complex_mean = statistics.mean(complex_scores)
    complex_std = statistics.stdev(complex_scores) if len(complex_scores) > 1 else 0
    overall_mean = statistics.mean(all_scores)
    overall_std = statistics.stdev(all_scores) if len(all_scores) > 1 else 0

    print(f"\nSimple Interest Baseline:    {simple_mean*100:.1f}% (±{simple_std*100:.1f}%)")
    print(f"Complex VF Baseline:         {complex_mean*100:.1f}% (±{complex_std*100:.1f}%)")
    print(f"Overall Baseline:            {overall_mean*100:.1f}% (±{overall_std*100:.1f}%)")

    # Save results
    results = {
        "simple_interest": {
            "mean": simple_mean,
            "std": simple_std,
            "n": len(simple_scores),
            "scores": simple_scores,
        },
        "complex_vf": {
            "mean": complex_mean,
            "std": complex_std,
            "n": len(complex_scores),
            "scores": complex_scores,
        },
        "overall": {
            "mean": overall_mean,
            "std": overall_std,
            "n": len(all_scores),
        },
        "method": "Random LLM guesses with temperature=1.0, evaluated by same LLM judge",
        "n_samples_per_objective": n_samples,
    }

    output_file = "results/multi_factor/objective_inference_baseline.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    # Use fewer samples for speed, increase for more precise estimate
    compute_baseline(n_samples=10)
