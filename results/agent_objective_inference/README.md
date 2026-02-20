# Agent Objective Inference Experiment Suite

**Date**: 2026-02-19 | **Games**: 226 | **Runtime**: ~63 hours

## Summary

![Summary](plots/summary_all_experiments.png)

---

## Experiment 1: Search Space Constraint

**Method**: Compare inference accuracy when selecting from N choices vs freeform generation.

- 6 modes: multiple-choice (2, 4, 8, 16), freeform, structured
- 2 complexity levels × 10 seeds = 120 games

![Search Space](plots/search_space_accuracy.png)

---

## Experiment 2: Oracle Budget Effect

**Method**: Vary number of oracle queries allowed (0-8) to test impact on inference.

- 6 budget levels × 10 seeds = 60 games
- Fixed: L3 complexity, freeform inference

![Oracle Budget](plots/oracle_budget_accuracy.png)

---

## Experiment 3: Objective Complexity Effect

**Method**: Vary objective complexity from simple (1 property) to complex (4-5 conditions + penalties).

- 5 complexity levels (L1-L5) × 10 seeds = 50 games
- Fixed: oracle_budget=4, freeform inference

![Complexity](plots/complexity_accuracy.png)

---

## Post-Hoc: Agent Strategy Classification

**Method**: LLM-based classification of 6 manipulation strategies from game transcripts.

- 53 games, 106 agent annotations
- Strategies: Fabricated Terminology, Truth Mixed with Lies, Oracle Spin, Credibility Attack, Escalating Complexity, Object Advocacy

![Strategies](plots/strategy_distribution.png)
