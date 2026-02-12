# Experiment Log

This file tracks all experiments run in the truthification project.

---

## 2026-02-04: Debate Structure Experiment

**Goal:** Test how debate turn structures and oracle timing affect truth recovery

**Config:**
- 20 objects, 20 rounds, 1 pick/round
- 8 conditions (4 turn structures × 2 oracle timings)
- 1 seed per condition (seed=42)

**Duration:** 2.3 hours

**Key Result:** `after_statements` oracle timing outperforms `before_response` (20% vs 12.75% avg). Best condition: interleaved + after_statements (26%).

**Results:** [results/debate_structure_experiment/README.md](debate_structure_experiment/README.md)

**Raw Data:** `outputs/debate_structure_test/20260204_183117/`

---

## 2026-02-11: Multi-Factor Experiment

**Goal:** Run full factorial experiment with agent value functions, estimator inference, and oracle access

**Config:**
- 10 objects, 10 rounds, 5 selections
- Complex agent value functions (each agent has unique goals)
- Estimator infers agent objectives
- 5 seeds (42, 123, 456, 789, 101112)

**Key Result:** Judge accuracy 44.4% with oracle (vs 30.0% no-oracle baseline). Estimator matches or exceeds judge in many conditions. Agent objective inference averages 0.54 (scale 0-1).

**Results:** [results/multi_factor/ANALYSIS.md](multi_factor/ANALYSIS.md)

**W&B:** https://wandb.ai/thomasjiralerspong/truthification

---

## 2026-02-11: No-Oracle Experiment

**Goal:** Measure the value of oracle access by comparing oracle_budget=8 vs oracle_budget=0

**Config:**
- 10 objects, 10 rounds, 5 selections
- 5 seeds each condition
- Conditions: with_oracle (budget=8) vs no_oracle (budget=0)

**Key Result:** Oracle provides +14.4% accuracy boost (44.4% with vs 30.0% without). No-oracle accuracy equals random baseline (30.7%).

**Results:** [results/multi_factor/ANALYSIS.md#oracle-value](multi_factor/ANALYSIS.md)

---

## 2026-02-12: Scale Experiment (Partial)

**Goal:** Test how truth recovery changes with more agents (2, 3, 4) and more rounds (10, 15, 20)

**Config:**
- 10 objects, oracle_budget=8, 5 selections
- Agent counts: 2, 3, 4
- Round counts: 10, 15, 20
- 3 seeds per condition (27 games total)

**Key Result (Partial):** More rounds may hurt judge accuracy (46% at 10 rounds → 37% at 15 rounds with 2 agents). Estimator often outperforms judge. Experiment hit rate limits before completion.

**Results:** [results/scale_experiment/README.md](scale_experiment/README.md)

**W&B:** https://wandb.ai/thomasjiralerspong/truthification/runs/f7q8taxh

---

## Template for New Experiments

```markdown
## YYYY-MM-DD: Experiment Name

**Goal:** [What are you testing?]

**Config:**
- [Key parameters]

**Duration:** [How long did it take?]

**Key Result:** [One sentence summary]

**Results:** [Link to results README]

**Raw Data:** [Path to outputs]
```
