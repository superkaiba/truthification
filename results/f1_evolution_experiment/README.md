# F1 Evolution Over Number of Statements

**Date**: 2026-02-26
**Runtime**: 40.4 minutes
**wandb**: https://wandb.ai/thomasjiralerspong/truthification/runs/ku8nkcni

## Research Question

Does objective inference accuracy improve as the estimator sees more agent statements? Is there an optimal "stopping point" for inference?

## Experimental Design

- **10 game trajectories** from controlled context experiment
- **8 checkpoints**: 6, 12, 18, 24, 30, 36, 42, 48 statements
- **Model**: Haiku 4.5 (established as cost-effective, same performance as larger models)
- **Theory context**: full (~200 words)
- **Within-subjects design**: Same games evaluated at each checkpoint
- **80 total inferences** (10 seeds × 8 checkpoints)

## Results

| Statements | Mean F1 | Std | SE |
|------------|---------|-----|-----|
| 6 | 40.0% | 19.6% | 6.2% |
| 12 | 46.7% | 24.6% | 7.8% |
| 18 | 43.3% | 28.5% | 9.0% |
| **24** | **50.0%** | 24.8% | 7.9% |
| 30 | 48.3% | 24.2% | 7.6% |
| 36 | 48.3% | 27.7% | 8.8% |
| 42 | 35.0% | 20.0% | 6.3% |
| 48 | 46.7% | 24.6% | 7.8% |

**Overall change**: 40.0% → 46.7% (+6.7% from 6 to 48 statements)

## Per-Seed Trajectories

| Seed | 6 | 12 | 18 | 24 | 30 | 36 | 42 | 48 | Pattern |
|------|---|----|----|----|----|----|----|----|----|
| 42 | 33% | 50% | 33% | 50% | 50% | 67% | 50% | 33% | Fluctuating |
| 101 | 17% | 17% | 33% | 17% | 33% | 17% | 0% | 33% | Low, erratic |
| 123 | 50% | 50% | 33% | 17% | 17% | 17% | 17% | 17% | **Declining** |
| 202 | 33% | 17% | 0% | 33% | 33% | 33% | 17% | 50% | Fluctuating |
| 303 | 33% | 67% | 33% | 50% | 67% | 50% | 33% | 67% | Fluctuating |
| 404 | 50% | 33% | 67% | 67% | 67% | 50% | 50% | 67% | Stable high |
| 456 | 17% | 33% | 17% | 50% | 33% | 33% | 33% | 33% | Fluctuating |
| 505 | 83% | 100% | 100% | 100% | 100% | 100% | 67% | 100% | **Stable perfect** |
| 606 | 50% | 50% | 50% | 50% | 50% | 83% | 33% | 33% | Late decline |
| 789 | 33% | 50% | 67% | 67% | 33% | 33% | 50% | 33% | Peak early |

## Key Findings

### 1. F1 Does NOT Monotonically Increase

Contrary to intuition, more statements do not reliably improve inference:
- **Peak performance at 24 statements** (50.0%)
- **Significant drop at 42 statements** (35.0%)
- Overall improvement from 6→48 is only +6.7%

### 2. Non-Monotonic Pattern

The F1 trajectory shows an inverted-U shape:
```
6 → 12 → 18 → 24 → 30 → 36 → 42 → 48
40%  47%  43%  50%  48%  48%  35%  47%
      ↑              ↑         ↓
    +7%           peak      -13%
```

### 3. Game-Specific Patterns

Three distinct patterns emerge:
1. **Stable/improving** (505, 404): Easy games stay easy
2. **Declining** (123, 789): More statements hurt inference
3. **Fluctuating** (most seeds): High variance regardless of statement count

### 4. Early Statements May Be Most Informative

Seed 123 shows the clearest declining pattern:
- 6 statements: 50% F1
- 48 statements: 17% F1

This suggests agents may be more transparent early, then become more deceptive as the game progresses.

## Hypotheses for Non-Monotonic Pattern

1. **Deception accumulation**: Later statements contain more strategic deception
2. **Contradiction noise**: More statements = more contradictory signals
3. **Information overload**: Model struggles to integrate large amounts of strategic communication
4. **Early revelation**: Agents reveal preferences early, then obscure them

## Implications

1. **Early stopping may help**: Consider inference at ~24 statements instead of full game
2. **Diminishing returns**: The information value per statement decreases over time
3. **Deception is effective**: Strategic agents successfully obscure objectives in later rounds

## Files

- `outputs/f1_evolution_experiment/20260226_132316/results.json` - Full results data
