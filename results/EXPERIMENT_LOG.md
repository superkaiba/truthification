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

**Key Result:** Estimator matches or exceeds judge in many conditions. Agent objective inference averages 0.54 (scale 0-1). ⚠️ Note: Oracle comparison from this experiment is invalid (see Forced Oracle Experiment).

**Results:** [results/multi_factor/ANALYSIS.md](multi_factor/ANALYSIS.md)

**W&B:** https://wandb.ai/thomasjiralerspong/truthification

---

## 2026-02-11: No-Oracle Experiment ⚠️ INVALID

**Goal:** Measure the value of oracle access by comparing oracle_budget=8 vs oracle_budget=0

**Config:**
- 10 objects, 10 rounds, 5 selections
- 5 seeds each condition
- Conditions: with_oracle (budget=8) vs no_oracle (budget=0)

**Key Result:** ⚠️ **INVALID** - Analysis revealed oracle_queries_used=0 in ALL games, even with budget=8. The LLM judge declined to use the oracle when offered. See Forced Oracle Experiment below for valid comparison.

**Results:** [results/no_oracle_comparison/](no_oracle_comparison/)

---

## 2026-02-13: Forced Oracle Experiment ✓

**Goal:** Re-run oracle comparison with force_oracle=True to ensure oracle is actually used

**Config:**
- 10 objects, 10 rounds, 5 selections, condition=interests, rule_complexity=medium
- 3 seeds (42, 123, 456)
- Conditions: no_oracle (budget=0) vs forced_oracle (budget=8, force_oracle=True)

**Key Result:** Oracle nearly triples property accuracy: 25.3% → 70.7% (+45.3pp) when forced to use it. All seeds showed large gains (+28pp to +62pp).

**Key Finding:** LLMs don't voluntarily use verification tools even when available. Implications for AI safety: verification mechanisms must be mandatory, not optional.

**Results:** [results/forced_oracle_test/](forced_oracle_test/)

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

## 2026-02-18: Oracle Budget Effect on Objective Inference

**Goal:** Test if more oracle queries help the estimator infer agent objectives

**Config:**
- 10 objects, 10 rounds, 5 selections
- Oracle budgets: [0, 1, 2, 4, 6, 8]
- Fixed: L3 complexity, freeform inference mode
- 10 seeds per condition (53 games completed due to some errors)

**Duration:** 15.3 hours (916 minutes)

**Key Result:** Oracle budget significantly improves objective inference. Optimal at budget=6 (27.1%), with diminishing returns after. Baseline (budget=0) achieves only 5.0%.

**Results:** [results/oracle_budget_objective_experiment/README_20260218_002133.md](oracle_budget_objective_experiment/README_20260218_002133.md)

**Raw Data:** `outputs/oracle_budget_objective/20260218_002133/`

**W&B:** https://wandb.ai/thomasjiralerspong/truthification

---

## 2026-02-18: Objective Complexity Effect on Inference

**Goal:** Test how objective complexity (L1-L5) affects inference accuracy

**Config:**
- 10 objects, 10 rounds, 5 selections
- Complexity levels: L1 (simple), L2 (dual), L3 (combo), L4 (complex), L5 (penalty)
- Fixed: oracle_budget=4, freeform inference mode
- 10 seeds per condition (47 games completed due to some errors)

**Duration:** 13.8 hours (831 minutes)

**Key Result:** Hypothesis confirmed - simpler objectives are easier to infer. L1 (1 property): 39.5% vs L5 (4-5 conditions + penalties): 16.4%. Clear monotonic decrease with complexity.

**Results:** [results/complexity_objective_experiment/README_20260218_002133.md](complexity_objective_experiment/README_20260218_002133.md)

**Raw Data:** `outputs/complexity_objective/20260218_002133/`

**W&B:** https://wandb.ai/thomasjiralerspong/truthification

---

## 2026-02-19: Search Space Constraint Effect (COMPLETE)

**Goal:** Compare inference accuracy across multiple-choice (2,4,8,16 options) vs freeform vs structured methods

**Config:**
- 10 objects, 10 rounds, 5 selections
- Methods: multiple_choice_2/4/8/16, freeform, structured
- Complexity levels: L1, L5
- 10 seeds per condition (116 games completed, 4 failed due to API errors)

**Duration:** 34.3 hours (2060 minutes)

**Key Finding:** **Multiple-choice VASTLY outperforms freeform/structured!** (90-95% vs 30-34%). Hypothesis was completely wrong - more choices don't decrease accuracy. This suggests LLMs are better at selection than generation for objective inference.

| Mode | Accuracy |
|------|----------|
| multiple_choice_16 | 95.0% |
| multiple_choice_8 | 94.9% |
| multiple_choice_2/4 | 90.0% |
| structured | 33.8% |
| freeform | 30.5% |

**Results:** [results/search_space_experiment/README_20260218_002133.md](search_space_experiment/README_20260218_002133.md)

**Raw Data:** `outputs/search_space/20260218_002133/`

**W&B:** https://wandb.ai/thomasjiralerspong/truthification

---

## 2026-02-21: Theory Context for Objective Inference

**Goal:** Test whether providing theoretical knowledge about strategic communication improves objective inference

**Config:**
- 10 objects, 10 rounds, 5 selections
- Conditions: none, brief (~50 words), full (~200 words)
- L3 complexity (3 properties per agent)
- 10 seeds per condition (30 games total)

**Duration:** 20.4 hours (1226 minutes)

**Key Result:** Theory context significantly improves inference. Full context: +16.7% (d=0.85, large effect). More detail is better.

**Results:** [results/objective_inference_experiments/README.md](objective_inference_experiments/README.md)

**Raw Data:** `outputs/theory_context_experiment/20260221_131125/`

---

## 2026-02-21: Deception Detection Strategies

**Goal:** Test whether specific deception detection strategies help infer agent objectives

**Config:**
- 10 objects, 10 rounds, 5 selections
- Strategies: baseline, consistency, incentive, pattern, combined
- L3 complexity, 10 seeds per condition (50 games total)

**Duration:** 26.5 hours (1590 minutes)

**Key Result:** Consistency checking works best (+8.3%, d=0.42), but effect smaller than theory context. Combined strategies no better than consistency alone.

**Results:** [results/objective_inference_experiments/README.md](objective_inference_experiments/README.md)

**Raw Data:** `outputs/deception_strategies_experiment/20260221_110535/`

---

## 2026-02-21: Agent Communication Strategy Effect

**Goal:** Test how agent communication strategies affect objective transparency

**Config:**
- 10 objects, 10 rounds, 5 selections
- Strategies: natural, honest, deceptive, misdirection, aggressive, subtle, credibility_attack
- L3 complexity, 10 seeds per condition (70 games total)

**Duration:** 31.4 hours (1885 minutes)

**Key Result:** Agent strategy has largest effect. Aggressive: +23.3% (d=1.07). Misdirection: -15.0% (d=-0.77). 38% spread between best and worst.

**Results:** [results/objective_inference_experiments/README.md](objective_inference_experiments/README.md)

**Raw Data:** `outputs/agent_strategy_inference/20260221_134220/`

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
