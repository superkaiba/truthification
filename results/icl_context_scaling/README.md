# ICL Context Scaling Experiment

**Date**: January 2026
**Config**: `configs/experiment/icl_context_scaling.yaml`
**Script**: `experiments/run_icl_context_scaling.py`
**Wandb**: [truthification project](https://wandb.ai/thomasjiralerspong/truthification)
**Status**: In Progress

## Research Question

How does accuracy vary with the number of statements shown? Is there an optimal context size?

## Experimental Setup

### Context Sizes Tested
- 30 statements (~12% of total)
- 60 statements (25%)
- 90 statements (37.5%)
- 120 statements (50%)
- 180 statements (75%)
- 240 statements (100%)

### Sampling Strategy
**Balanced**: Equal number of statements sampled from each agent to preserve reliability signal.

### Conditions
- `full_context_no_ids`
- `full_context_ids`
- `full_context_ids_tasks`

## Preliminary Results (Seeds 42, 123)

### Accuracy vs Context Size

| Statements | Haiku | Sonnet | Opus |
|------------|-------|--------|------|
| 30 | 27-29% | 27-29% | 25-35% |
| 60 | 42-46% | 43-47% | 42-48% |
| 90 | 50-60% | 56-65% | 51-65% |
| 120 | 54-70% | 68-72% | 59-69% |
| 180 | 69-84% | 77-88% | 65-85% |
| 240 | 76-85% | 81-85% | 75-84% |

*Ranges show variation across conditions and seeds*

## Key Findings (Preliminary)

### 1. Strong Monotonic Scaling
Accuracy increases consistently with more context:
- 30 statements: ~28% (near random guessing)
- 240 statements: ~80%

### 2. Critical Threshold at 60-90 Statements
Major jump from ~28% to ~55%:
- This is likely where there's enough evidence for basic reliability inference
- Below this, insufficient data to detect patterns

### 3. Diminishing Returns Above 180
Improvement slows between 180-240 statements:
- Most reliability signal captured by 180 statements
- Additional context provides marginal benefit

### 4. Model Differences Emerge at Higher Context
- At 30 statements: All models similar (~28%)
- At 240 statements: Sonnet consistently highest (81-88%)

## Interpretation

The results suggest:
1. **Minimum viable context**: ~60-90 statements needed
2. **No degradation observed**: Unlike full-context experiment, Opus doesn't degrade
3. **Linear-ish scaling**: Roughly linear improvement in log-context space

## Plots

*Plots will be generated when experiment completes*

<!-- TODO: Add plots when available
![Context Scaling](context_scaling_accuracy.png)
![Scaling by Condition](context_scaling_by_condition.png)
-->
