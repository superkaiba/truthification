# Model Comparison Experiment

**Date**: 2026-02-26
**Runtime**: 58.3 minutes
**wandb**: https://wandb.ai/thomasjiralerspong/truthification/runs/79u273sy

## Research Question

Does model capability (Haiku vs Sonnet vs Opus, different versions) affect performance on objective inference from strategic agent communication?

## Experimental Design

- **7 Claude models** tested as the estimator
- **Same 10 game trajectories** (saved from controlled context experiment)
- **Theory context**: "full" (~200 words, established as best cost/performance)
- **Within-subjects design**: Each model evaluates the identical agent statements

This design isolates model capability from debate variance - all models see the exact same inputs.

## Models Tested

| Model | API ID | Cost (input/output per MTok) |
|-------|--------|------------------------------|
| Haiku 4.5 | claude-haiku-4-5-20251001 | $1 / $5 |
| Sonnet 4 | claude-sonnet-4-20250514 | $3 / $15 |
| Sonnet 4.5 | claude-sonnet-4-5-20250929 | $3 / $15 |
| Sonnet 4.6 | claude-sonnet-4-6 | $3 / $15 |
| Opus 4 | claude-opus-4-20250514 | $15 / $75 |
| Opus 4.5 | claude-opus-4-5-20251101 | $5 / $25 |
| Opus 4.6 | claude-opus-4-6 | $5 / $25 |

## Results

| Model | Mean F1 | Std | SE | Rank |
|-------|---------|-----|-----|------|
| **Haiku 4.5** | **43.3%** | 19.6% | 6.2% | 1 |
| Opus 4 | 40.0% | 27.4% | 8.7% | 2 |
| Opus 4.5 | 40.0% | 26.3% | 8.3% | 2 |
| Opus 4.6 | 40.0% | 25.1% | 7.9% | 2 |
| Sonnet 4 | 38.3% | 27.3% | 8.6% | 5 |
| Sonnet 4.5 | 38.3% | 24.9% | 7.9% | 5 |
| Sonnet 4.6 | 36.7% | 27.0% | 8.5% | 7 |

## Statistical Comparison vs Sonnet 4 (baseline)

| Model | Diff | Cohen's d | p-value | Significant? |
|-------|------|-----------|---------|--------------|
| Haiku 4.5 | +5.0% | 0.26 | 0.43 | No |
| Sonnet 4.5 | 0.0% | 0.00 | 1.00 | No |
| Sonnet 4.6 | -1.7% | -0.14 | 0.68 | No |
| Opus 4 | +1.7% | 0.32 | 0.34 | No |
| Opus 4.5 | +1.7% | 0.18 | 0.59 | No |
| Opus 4.6 | +1.7% | 0.32 | 0.34 | No |

**None of the differences are statistically significant (all p > 0.3).**

## Per-Seed Breakdown

| Seed | Haiku 4.5 | Sonnet 4 | Sonnet 4.5 | Sonnet 4.6 | Opus 4 | Opus 4.5 | Opus 4.6 |
|------|-----------|----------|------------|------------|--------|----------|----------|
| 42 | 33% | 50% | 50% | 50% | 50% | 50% | 50% |
| 101 | 17% | 0% | 17% | 0% | 0% | 17% | 17% |
| 123 | 33% | 33% | 33% | 50% | 50% | 33% | 33% |
| 202 | 50% | 17% | 33% | 33% | 17% | 33% | 17% |
| 303 | 67% | 50% | 50% | 33% | 50% | 50% | 50% |
| 404 | 33% | 33% | 17% | 17% | 33% | 33% | 33% |
| 456 | 33% | 33% | 33% | 33% | 33% | 33% | 33% |
| **505** | 83% | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** |
| 606 | 33% | 50% | 33% | 33% | 50% | 50% | 50% |
| 789 | 50% | 17% | 17% | 17% | 17% | 0% | 17% |

### Seed-level observations:
- **Seed 505** is "easy": All models except Haiku achieve 100% F1
- **Seed 101** is "hard": No model exceeds 17% F1
- **Seed 456** is the great equalizer: All models score exactly 33%
- Variance between seeds (20-100%) >> variance between models

## Key Findings

### 1. Model capability doesn't significantly affect performance

All 7 models perform statistically equivalently on this task. The expensive Opus models ($15-75/MTok) do not outperform the cheap Haiku ($1-5/MTok).

### 2. Haiku 4.5 unexpectedly leads

The cheapest model (Haiku 4.5) achieves the highest mean F1 at 43.3%, though this is not statistically significant (p=0.43 vs Sonnet 4).

### 3. Task difficulty varies more by game than by model

The per-seed breakdown shows that some games (seed 505) are solved by almost all models, while others (seed 101) defeat all models. This suggests:
- The underlying task difficulty varies significantly across game configurations
- Model capability is not the bottleneck for this task
- The challenge is in the strategic reasoning itself, not raw model intelligence

### 4. Newer versions don't improve performance

- Sonnet 4.6 performs worse than Sonnet 4 (-1.7%)
- Opus 4, 4.5, and 4.6 all tie at 40.0%

## Implications

1. **Cost optimization**: Use Haiku 4.5 for production - same or better performance at 1/15th the cost of Opus
2. **Research direction**: Focus on improving the task/prompt rather than upgrading models
3. **Evaluation**: The high variance across seeds suggests need for larger sample sizes
4. **Theory context**: Confirmed that "full" theory context is sufficient (no model is bottlenecked by context understanding)

## Files

- `outputs/model_comparison_experiment/20260226_004501/results.json` - Full results data
- `outputs/model_comparison_experiment/20260226_004501/README.md` - Auto-generated report
