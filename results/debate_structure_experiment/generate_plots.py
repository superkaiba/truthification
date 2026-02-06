"""Generate plots for debate structure experiment results."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
results_path = Path("outputs/debate_structure_test/20260204_183117/summary.json")
with open(results_path) as f:
    data = json.load(f)

results = data["results"]
output_dir = Path("results/debate_structure_experiment")
output_dir.mkdir(exist_ok=True)

# Extract data for plotting
conditions = []
est_accuracies = []
obs_accuracies = []
progressions = {}

for r in results:
    label = f"{r['turn_structure']}\n{r['oracle_timing']}"
    short_label = f"{r['turn_structure'][:4]}_{r['oracle_timing'][:6]}"
    conditions.append(label)
    est_accuracies.append(r["estimator_property_accuracy"] * 100)
    obs_accuracies.append(r["observer_property_accuracy"] * 100)
    progressions[short_label] = [
        p["estimator_property_accuracy"] * 100
        for p in r["accuracy_progression"]
    ]

# Plot 1: Bar chart comparing final accuracies
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(conditions))
width = 0.35

bars1 = ax.bar(x - width/2, est_accuracies, width, label='Estimator', color='#1976D2')
bars2 = ax.bar(x + width/2, obs_accuracies, width, label='Observer (Judge)', color='#4CAF50')

ax.set_ylabel('Property Accuracy (%)')
ax.set_title('Final Accuracy by Debate Structure and Oracle Timing')
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=8)
ax.legend()
ax.set_ylim(0, 50)
ax.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='Random baseline (~20%)')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / "final_accuracy_comparison.png", dpi=150)
plt.close()

# Plot 2: Accuracy progression over rounds
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Group by turn structure
turn_structures = ["interleaved", "batch", "simultaneous", "sequential"]
colors = {"before": "#E53935", "after_": "#1E88E5"}

for idx, ts in enumerate(turn_structures):
    ax = axes[idx]
    for key, prog in progressions.items():
        if key.startswith(ts[:4]):
            timing = "before" if "before" in key else "after_"
            color = colors[timing]
            label = "before_response" if "before" in key else "after_statements"
            ax.plot(range(1, len(prog)+1), prog, marker='o', markersize=3,
                   color=color, label=label, linewidth=2)

    ax.set_xlabel('Round')
    ax.set_ylabel('Estimator Accuracy (%)')
    ax.set_title(f'{ts.capitalize()} Turn Structure')
    ax.legend()
    ax.set_ylim(0, 35)
    ax.set_xlim(1, 20)
    ax.grid(True, alpha=0.3)

plt.suptitle('Estimator Accuracy Progression Over 20 Rounds', fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "accuracy_progression_by_structure.png", dpi=150)
plt.close()

# Plot 3: Oracle timing comparison (aggregated)
fig, ax = plt.subplots(figsize=(10, 6))

before_accs = [r["estimator_property_accuracy"] * 100 for r in results if r["oracle_timing"] == "before_response"]
after_accs = [r["estimator_property_accuracy"] * 100 for r in results if r["oracle_timing"] == "after_statements"]

structures = ["interleaved", "batch", "simultaneous", "sequential"]
x = np.arange(len(structures))
width = 0.35

ax.bar(x - width/2, before_accs, width, label='before_response', color='#E53935')
ax.bar(x + width/2, after_accs, width, label='after_statements', color='#1E88E5')

ax.set_ylabel('Estimator Accuracy (%)')
ax.set_title('Oracle Timing Effect on Truth Recovery')
ax.set_xticks(x)
ax.set_xticklabels(structures)
ax.legend()
ax.set_ylim(0, 35)

plt.tight_layout()
plt.savefig(output_dir / "oracle_timing_comparison.png", dpi=150)
plt.close()

# Plot 4: Agent success over rounds (for interleaved condition)
fig, ax = plt.subplots(figsize=(10, 6))

# Load detailed game data for agent success
game_path = Path("outputs/debate_structure_test/20260204_183117/game_interleaved_after_statements.json")
with open(game_path) as f:
    game = json.load(f)

rounds_data = game.get("accuracy_progression", [])
agent_a_success = []
agent_b_success = []

for rd in rounds_data:
    as_data = rd.get("agent_success", {})
    agent_a_success.append(as_data.get("Agent_A", {}).get("rate", 0) * 100)
    agent_b_success.append(as_data.get("Agent_B", {}).get("rate", 0) * 100)

rounds = range(1, len(agent_a_success) + 1)
ax.plot(rounds, agent_a_success, 'o-', color='#E53935', label='Agent_A Success', linewidth=2)
ax.plot(rounds, agent_b_success, 's-', color='#1E88E5', label='Agent_B Success', linewidth=2)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

ax.set_xlabel('Round')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Agent Success Rate Over Rounds (interleaved + after_statements)')
ax.legend()
ax.set_ylim(0, 110)
ax.set_xlim(1, 20)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "agent_success_progression.png", dpi=150)
plt.close()

print("Plots saved to results/debate_structure_experiment/")
