"""
Compute random baseline for rule inference accuracy.

Tests what score an uninformed guesser would get from the LLM judge
when evaluating rule inference accuracy.
"""

import json
import os
import random
import statistics
from anthropic import Anthropic

# Load environment
from dotenv import load_dotenv
load_dotenv()

client = Anthropic()

# Sample ground truth rules from the experiment (representative of simple, medium, complex)
SAMPLE_RULES = [
    # Simple rule
    {
        "name": "Star Bonus",
        "description": "Stars are valuable",
        "conditions": [
            {"description": "Object is star-shaped", "bonus": 50}
        ]
    },
    # Medium rule
    {
        "name": "Shape and Material Bonuses",
        "description": "Stars and wooden objects are valuable, with bonuses for combinations",
        "conditions": [
            {"description": "Object is star-shaped", "bonus": 30},
            {"description": "Object is made of wood", "bonus": 25},
            {"description": "Object is star-shaped AND wooden", "bonus": 20}
        ]
    },
    # Complex rule
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

# Different types of uninformed/random inferences
RANDOM_INFERENCES = [
    # Completely uninformed
    {
        "description": "I have no idea what makes objects valuable. I couldn't determine any pattern.",
        "confidence": 10,
        "key_factors": []
    },
    # Generic guess
    {
        "description": "Valuable objects seem to be the ones that are picked more often.",
        "confidence": 20,
        "key_factors": ["frequency"]
    },
    # Random property guess (wrong)
    {
        "description": "Red objects appear to be most valuable, especially large ones.",
        "confidence": 50,
        "key_factors": ["color", "size"]
    },
    # Another random guess
    {
        "description": "Metal objects are valuable, particularly circles.",
        "confidence": 45,
        "key_factors": ["material", "shape"]
    },
    # Vague guess
    {
        "description": "Objects with certain shapes seem more valuable than others.",
        "confidence": 30,
        "key_factors": ["shape"]
    },
    # Random combination
    {
        "description": "Blue triangles and large plastic items are valuable.",
        "confidence": 55,
        "key_factors": ["color", "shape", "size", "material"]
    }
]

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
    print("=" * 60)
    print("RULE INFERENCE ACCURACY: RANDOM BASELINE COMPUTATION")
    print("=" * 60)
    print()
    
    all_scores = []
    results_by_rule = {r["name"]: [] for r in SAMPLE_RULES}
    results_by_inference = {i["description"][:30]: [] for i in RANDOM_INFERENCES}
    
    # Test each rule against each random inference
    total_tests = len(SAMPLE_RULES) * len(RANDOM_INFERENCES)
    test_num = 0
    
    for rule in SAMPLE_RULES:
        print(f"\n--- Testing against: {rule['name']} ---")
        for inference in RANDOM_INFERENCES:
            test_num += 1
            print(f"  [{test_num}/{total_tests}] Testing: {inference['description'][:40]}...")
            
            score, reasoning = evaluate_inference(rule, inference)
            all_scores.append(score)
            results_by_rule[rule["name"]].append(score)
            results_by_inference[inference["description"][:30]].append(score)
            
            print(f"    Score: {score:.2f} - {reasoning[:60]}...")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\n** OVERALL RANDOM BASELINE: {statistics.mean(all_scores)*100:.1f}% **")
    print(f"   Std Dev: {statistics.stdev(all_scores)*100:.1f}%")
    print(f"   Min: {min(all_scores)*100:.1f}%, Max: {max(all_scores)*100:.1f}%")
    print(f"   N: {len(all_scores)}")
    
    print("\nBy Rule Complexity:")
    for rule_name, scores in results_by_rule.items():
        print(f"  {rule_name}: {statistics.mean(scores)*100:.1f}%")
    
    print("\nBy Inference Type:")
    for inf_desc, scores in results_by_inference.items():
        print(f"  {inf_desc}: {statistics.mean(scores)*100:.1f}%")
    
    # Save results
    results = {
        "overall_baseline": statistics.mean(all_scores),
        "std": statistics.stdev(all_scores),
        "n": len(all_scores),
        "by_rule": {k: statistics.mean(v) for k, v in results_by_rule.items()},
        "by_inference": {k: statistics.mean(v) for k, v in results_by_inference.items()},
        "all_scores": all_scores
    }
    
    output_path = "results/multi_factor/rule_inference_baseline.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
