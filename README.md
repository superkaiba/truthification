# Truthification

Research project investigating whether LLM observers can infer agent reliability through in-context learning (ICL) and determine ground truth from potentially unreliable agent statements.

## Key Research Question

**Can LLMs learn to identify which agents are reliable by observing patterns in their statements, and use this to determine ground truth?**

## Experiments

### [1. ICL Baseline (Isolated Query)](results/icl_baseline/)

Observer sees only statements about the queried object/property.

![Overall Accuracy](results/icl_baseline/overall_accuracy.png)

**Key finding**: Sonnet with task info achieves 90.8%, nearly matching oracle (92.9%).

[Full results and analysis](results/icl_baseline/README.md)

---

### [2. ICL Full-Context](results/icl_full_context/)

Observer sees ALL 240 statements for every query.

![Isolated vs Full Context](results/icl_full_context/isolated_vs_full_comparison.png)

![Full Context Delta](results/icl_full_context/full_context_delta.png)

**Key findings**:
- Haiku benefits most from full context (+8.3pp)
- Opus degrades with more context (-5.4pp)
- Sonnet's task advantage disappears

[Full results and analysis](results/icl_full_context/README.md)

---

### [3. ICL Context Scaling](results/icl_context_scaling/) *(In Progress)*

Tests how accuracy varies with context size (30-240 statements).

**Preliminary finding**: Strong monotonic scaling from ~28% to ~80%.

[Full results and analysis](results/icl_context_scaling/README.md)

---

## Results Summary

| Experiment | Best Result | Key Insight |
|------------|-------------|-------------|
| Baseline | Sonnet 90.8% (tasks) | Task inference nearly matches oracle |
| Full-Context | Haiku 87.1% (no IDs) | Smaller models benefit from more data |
| Context Scaling | ~80% at 240 stmts | Linear scaling, threshold at 60-90 |

## Experimental Setup

### World & Agents
- **20 objects** with 4 properties: color, shape, size, value
- **3 agents** with different tasks
- **Adversarial agents** have consistent FALSE beliefs about observer-relevant properties

### Key Design Innovation
Adversarial agents lie consistently about color/shape (observer cares) but tell truth about size/value (observer doesn't care). This creates detectable patterns.

## Setup

```bash
uv sync
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env
```

## Running Experiments

```bash
# Baseline
uv run python experiments/run_icl.py

# Full-context
uv run python experiments/run_icl_full_context.py

# Context scaling
uv run python experiments/run_icl_context_scaling.py
```

## Project Structure

```
truthification/
├── src/
│   ├── environment/     # World, Agent, Simulation
│   ├── observer/        # ICLObserver, PromptBuilder
│   └── evaluation/      # Metrics
├── configs/experiment/  # Hydra configs
├── experiments/         # Experiment scripts
├── results/             # Experiment summaries with plots
│   ├── icl_baseline/
│   ├── icl_full_context/
│   └── icl_context_scaling/
└── tests/
```

## Wandb

[truthification project](https://wandb.ai/thomasjiralerspong/truthification)
