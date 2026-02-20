# Agent Objective Inference Experiment Suite Results

**Date**: 2026-02-19
**Total Runtime**: ~63.4 hours (2.6 days)
**Total Games**: 226 (116 + 60 + 50)

## Summary

![Summary of All Experiments](plots/summary_all_experiments.png)

## Research Questions

1. How does constraining the search space affect objective inference accuracy?
2. Does more oracle budget help objective inference?
3. How does objective complexity affect inference accuracy?
4. What manipulation strategies do agents use?

---

## Experiment 1: Search Space Constraint

**Design:** 6 inference modes × 2 complexity levels × 10 seeds = 120 games (116 completed)

### Hypothesis
Accuracy order: 2-choice > 4 > 8 > freeform > 16

### Results

![Search Space Accuracy](plots/search_space_accuracy.png)

| Mode | Obj Inf Score (mean) | Obj Inf Score (std) |
|------|----------------------|---------------------|
| multiple_choice_16 | **95.0%** | 14.1% |
| multiple_choice_8 | **94.9%** | 10.5% |
| multiple_choice_4 | 90.0% | 12.9% |
| multiple_choice_2 | 90.0% | 20.0% |
| structured | 33.8% | 13.8% |
| freeform | 30.5% | 24.1% |

### Key Finding

**Hypothesis REVERSED.** The LLM is remarkably good at multiple-choice selection (~90-95%) regardless of the number of distractors, but struggles with freeform generation (~30%). This suggests:

1. The model has strong implicit knowledge of what agent objectives "look like"
2. Recognition >> Generation for this task
3. Even 15 distractors doesn't significantly degrade selection accuracy
4. The bottleneck is hypothesis generation, not evaluation

---

## Experiment 2: Oracle Budget Effect

**Design:** 6 budget levels × 10 seeds = 60 games

### Hypothesis
More oracle → better calibration of agent credibility → better inference

### Results

![Oracle Budget Accuracy](plots/oracle_budget_accuracy.png)

| Oracle Budget | Obj Inf Score (mean) | Queries Used |
|---------------|----------------------|--------------|
| 0 | 5.0% | 0.0 |
| 1 | 14.5% | 1.0 |
| 2 | 12.5% | 2.0 |
| 4 | 21.0% | 4.0 |
| **6** | **27.1%** | 6.0 |
| 8 | 21.7% | 8.0 |

### Key Finding

**Hypothesis partially confirmed.** Oracle budget helps objective inference up to ~6 queries:

1. Zero oracle (5.0%) → minimal ability to infer without ground truth
2. Peak at 6 queries (27.1%) → optimal information gathering
3. Diminishing returns beyond 6 → possible information overload or overfitting
4. The gains are modest - even optimal budget only reaches ~27%

---

## Experiment 3: Objective Complexity Effect

**Design:** 5 complexity levels × 10 seeds = 50 games

### Complexity Levels

| Level | Description | Example |
|-------|-------------|---------|
| L1 | Simple: 1 property | "Wants blue objects" |
| L2 | Dual: 2 properties | "Wants blue AND large" |
| L3 | Combo: 2 props + bonus | "Blue +20, large +15, combo +25" |
| L4 | Complex: 3-4 conditions | Multiple bonuses |
| L5 | Penalty: 4-5 + penalties | Bonuses + negative conditions |

### Hypothesis
Accuracy: L1 (~70%) > L2 > L3 > L4 > L5 (~30%)

### Results

![Complexity Accuracy](plots/complexity_accuracy.png)

| Level | Obj Inf Score (mean) | Avg Conditions |
|-------|----------------------|----------------|
| **L1** | **39.5%** | ~1.0 |
| L2 | 21.5% | ~2.0 |
| L3 | 23.0% | ~2.5 |
| L4 | 17.5% | ~3.5 |
| L5 | 16.4% | ~4.5 |

### Key Finding

**Hypothesis confirmed.** Objective complexity significantly affects inference accuracy:

1. Simple objectives (L1: 39.5%) are ~2.4x easier to infer than complex ones (L5: 16.4%)
2. L2-L3 are similar (~22%) - the jump from 1→2 properties is the biggest hurdle
3. L4-L5 plateau at ~17% - additional complexity has diminishing impact
4. The trend matches intuition: more conditions = harder to infer

---

## Post-Hoc Analysis: Agent Manipulation Strategies

**Design:** LLM-based classification of 53 games (106 agent annotations)

### Strategy Definitions

| Strategy | Description |
|----------|-------------|
| **Fabricated Terminology** | Invents fake properties like "azure crystalline" |
| **Truth Mixed with Lies** | True rule + false object claims |
| **Oracle Spin** | Reframes oracle results to support narrative |
| **Credibility Attack** | "Agent_B is lying", attacks trustworthiness |
| **Escalating Complexity** | More elaborate explanations each round |
| **Object Advocacy** | Direct "I recommend object_X" |

### Results

![Strategy Distribution](plots/strategy_distribution.png)

| Strategy | Prevalence | Mean Confidence |
|----------|------------|-----------------|
| Object Advocacy | **100%** | 96.2 |
| Truth Mixed with Lies | **100%** | 90.9 |
| Escalating Complexity | 98.1% | 82.9 |
| Credibility Attack | 96.2% | 80.9 |
| Fabricated Terminology | 90.6% | 84.4 |
| Oracle Spin | 81.1% | 69.4 |

### Key Finding

**All agents use multiple manipulation strategies simultaneously.** Object advocacy and truth-mixed-with-lies are universal (100% prevalence). This suggests LLM agents naturally develop sophisticated persuasion tactics when given competing objectives.

---

## Summary and Implications

### Key Findings

1. **Recognition >> Generation**: LLMs are excellent at selecting the correct objective from options (95%) but struggle to generate hypotheses (30%)

2. **Optimal Oracle Budget**: ~6 queries provides the best information/noise tradeoff

3. **Complexity Matters**: Simple objectives are significantly easier to infer (40% vs 17%)

4. **Universal Manipulation**: All agents employ multiple persuasion strategies

### Implications for Research

1. **Multiple-choice evaluation** may be more reliable for measuring objective inference capability
2. **Oracle access** helps but has diminishing returns - design experiments with 4-6 queries
3. **Baseline calibration**: Random baseline for multiple-choice should be 1/N (6.25% for 16 options)

### Baselines

| Condition | Expected Random | Observed |
|-----------|-----------------|----------|
| MC-16 | 6.25% | 95.0% |
| MC-8 | 12.5% | 94.9% |
| MC-4 | 25.0% | 90.0% |
| MC-2 | 50.0% | 90.0% |
| Freeform | ~19% (computed) | 30.5% |

All results significantly exceed random chance, but the relative gains in multiple-choice settings are particularly striking.

---

## Raw Data Locations

- Search Space: `outputs/search_space/20260218_002133/`
- Oracle Budget: `outputs/oracle_budget_objective/20260218_002133/`
- Complexity: `outputs/complexity_objective/20260218_002133/`
- Strategy Annotations: `results/strategy_annotations/20260219_122001/`
