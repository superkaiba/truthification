"""
Compute random baseline for rule inference accuracy using LLM-generated guesses.

An LLM with NO information about the game generates random guesses about
what the value rule might be. These are then evaluated by the LLM judge.
"""

import json
import os
import statistics
from anthropic import Anthropic

from dotenv import load_dotenv
load_dotenv()

client = Anthropic()

# Sample ground truth rules from the experiment
SAMPLE_RULES = [
    {
        "name": "Star Bonus",
        "description": "Stars are valuable",
        "conditions": [
            {"description": "Object is star-shaped", "bonus": 50}
        ]
    },
    {
        "name": "Shape and Material Bonuses", 
        "description": "Stars and wooden objects are valuable, with bonuses for combinations",
        "conditions": [
            {"description": "Object is star-shaped", "bonus": 30},
            {"description": "Object is made of wood", "bonus": 25},
            {"description": "Object is star-shaped AND wooden", "bonus": 20}
        ]
    },
    {
        "name": "Multi-factor Value Rule",
        "description": "Complex rule with multiple conditions and penalties",
        "conditions": [
            {"description": "Object is star-shaped AND wooden", "bonus": 50},
            {"description": "Object is triangular AND not dangerous", "bonus": 30},
            {"description": "Object is made of glass", "bonus": 20},
            {"description": "Object is dangerous AND plastic", "bonus": -40}
        ]
    }
]

def generate_random_guess(guesser_id: int) -> dict:
    """Have an LLM generate a random guess about what a value rule might be."""
    
    prompt = """You are playing a guessing game. Objects have these properties:
- color: red, blue, green, yellow, orange
- shape: circle, square, triangle, star
- size: small, medium, large
- material: metal, wood, plastic, glass
- is_dangerous: true, false

There is a hidden rule that determines which objects are valuable.
You have NO information about what this rule actually is.

Make a random guess about what the rule might be. Be specific about:
1. Which properties matter
2. Which values of those properties are valuable
3. Any combinations or relationships

Respond with JSON only:
{
    "description": "Your guess about what makes objects valuable",
    "confidence": <10-60>,
    "key_factors": ["list", "of", "properties", "you", "think", "matter"]
}

Important: This is a RANDOM guess. You have no information. Just pick some properties and values arbitrarily."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,  # High temperature for randomness
    )
    
    text = response.content[0].text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        data = json.loads(text[start:end])
        return data
    return {
        "description": "Random objects are valuable",
        "confidence": 20,
        "key_factors": ["unknown"]
    }

def format_rule(rule: dict) -> str:
    """Format a rule for the prompt."""
    desc = f"Name: {rule['name']}\n"
    desc += f"Description: {rule['description']}\n"
    desc += "Conditions:\n"
    for cond in rule["conditions"]:
        desc += f"  - {cond['description']}: {cond['bonus']:+d} points\n"
    return desc

def format_inference(inf: dict) -> str:
    """Format an inference for the prompt."""
    desc = f"Description: {inf['description']}\n"
    desc += f"Confidence: {inf['confidence']}%\n"
    key_factors = inf['key_factors'] if inf['key_factors'] else ['none identified']
    desc += f"Key Factors: {', '.join(key_factors)}\n"
    return desc

def evaluate_inference(rule: dict, inference: dict) -> tuple[float, str]:
    """Evaluate an inference against a rule using LLM judge."""
    true_rule_desc = format_rule(rule)
    inferred_desc = format_inference(inference)
    
    prompt = f"""You are evaluating how well an observer inferred a hidden value rule.

GROUND TRUTH (the actual rule):
{true_rule_desc}

OBSERVER'S INFERENCE:
{inferred_desc}

Rate the inference on a scale of 0.0 to 1.0:
- 1.0 = Perfect understanding - correct properties, values, relationships, and relative importance
- 0.8-0.9 = Strong understanding - got the key factors and relationships right, minor details off
- 0.5-0.7 = Partial understanding - identified some correct factors but missed relationships or specific values
- 0.2-0.4 = Weak understanding - mentioned relevant properties but wrong relationships or values
- 0.0-0.1 = No understanding - completely wrong or unrelated inference

Consider:
1. Did they identify the correct PROPERTIES that matter (shape, material, etc.)?
2. Did they identify the correct VALUES (star, wooden, etc.)?
3. Did they understand the RELATIONSHIPS (AND vs OR, combinations)?
4. Did they understand relative IMPORTANCE (which factors matter more)?

Respond with JSON only:
{{"score": 0.0-1.0, "reasoning": "Brief explanation"}}"""

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    
    text = response.content[0].text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        data = json.loads(text[start:end])
        return float(data.get("score", 0.0)), data.get("reasoning", "")
    return 0.0, "Failed to parse"

def main():
    print("=" * 70)
    print("RULE INFERENCE BASELINE: LLM-GENERATED RANDOM GUESSES")
    print("=" * 70)
    
    NUM_GUESSES = 10  # Generate this many random guesses
    
    # Generate random guesses
    print(f"\nGenerating {NUM_GUESSES} random guesses from uninformed LLM...")
    random_guesses = []
    for i in range(NUM_GUESSES):
        print(f"  Generating guess {i+1}/{NUM_GUESSES}...")
        guess = generate_random_guess(i)
        random_guesses.append(guess)
        print(f"    -> {guess['description'][:60]}...")
    
    print("\n" + "-" * 70)
    print("Evaluating guesses against ground truth rules...")
    print("-" * 70)
    
    all_scores = []
    results_by_rule = {r["name"]: [] for r in SAMPLE_RULES}
    detailed_results = []
    
    total_tests = len(SAMPLE_RULES) * len(random_guesses)
    test_num = 0
    
    for rule in SAMPLE_RULES:
        print(f"\n--- Testing against: {rule['name']} ---")
        for i, guess in enumerate(random_guesses):
            test_num += 1
            print(f"  [{test_num}/{total_tests}] Guess: {guess['description'][:40]}...")
            
            score, reasoning = evaluate_inference(rule, guess)
            all_scores.append(score)
            results_by_rule[rule["name"]].append(score)
            detailed_results.append({
                "rule": rule["name"],
                "guess": guess["description"],
                "score": score,
                "reasoning": reasoning
            })
            
            print(f"    Score: {score:.2f} - {reasoning[:50]}...")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    mean_score = statistics.mean(all_scores)
    std_score = statistics.stdev(all_scores)
    
    print(f"\n** RANDOM BASELINE (LLM-generated guesses): {mean_score*100:.1f}% **")
    print(f"   Std Dev: {std_score*100:.1f}%")
    print(f"   Min: {min(all_scores)*100:.1f}%, Max: {max(all_scores)*100:.1f}%")
    print(f"   N: {len(all_scores)}")
    
    print("\nBy Rule Complexity:")
    for rule_name, scores in results_by_rule.items():
        print(f"  {rule_name}: {statistics.mean(scores)*100:.1f}% (±{statistics.stdev(scores)*100:.1f}%)")
    
    # Show the guesses
    print("\nRandom Guesses Generated:")
    for i, guess in enumerate(random_guesses):
        avg_score = statistics.mean([r["score"] for r in detailed_results if r["guess"] == guess["description"]])
        print(f"  {i+1}. [{avg_score*100:.0f}%] {guess['description'][:70]}")
    
    # Save results
    results = {
        "method": "LLM-generated random guesses (temperature=1.0, no game info)",
        "overall_baseline": mean_score,
        "std": std_score,
        "n": len(all_scores),
        "by_rule": {k: {"mean": statistics.mean(v), "std": statistics.stdev(v)} for k, v in results_by_rule.items()},
        "all_scores": all_scores,
        "guesses": random_guesses,
        "detailed_results": detailed_results
    }
    
    output_path = "results/multi_factor/rule_inference_baseline_v2.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
