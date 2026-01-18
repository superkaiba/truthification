# Truthification Project

## Project Overview

Research project investigating truthification methods for LLMs - teaching models to distinguish between verified facts, scientific claims, and opinions through annotated training data and trust tier syntax.

## Development Setup

### Package Management
- Use **uv** for Python package management
- Run `uv sync` to install dependencies
- Add packages with `uv add <package>`

### Configuration
- Use **Hydra** for experiment configuration
- Config files go in `configs/` directory
- Override configs via command line: `python train.py model=qwen data=wikipedia`

### Experiment Tracking
- Use **Weights & Biases (wandb)** for experiment tracking
- Log metrics, configs, and artifacts to wandb
- Name runs descriptively: `{experiment_type}-{model}-{date}`
- **When creating plots, always log them to wandb and provide the wandb link**

### Environment Variables
- Store all API keys and tokens in `.env` file
- Never commit `.env` to version control
- Required variables:
  ```
  ANTHROPIC_API_KEY=...
  WANDB_API_KEY=...
  OPENAI_API_KEY=...
  HF_TOKEN=...
  ```

## Workflow Guidelines

### Commit Early and Often
- **Commit after every meaningful change** - code changes, config updates, or experimental results
- Push frequently to keep remote in sync
- Use descriptive commit messages that reference what was tested/changed

### Experimental Results
- Log all experiment results to wandb
- Also commit summary notes/observations to the repo
- Document what worked, what didn't, and hypotheses for why

### Running Experiments
- **Always run experiments in the background** using `run_in_background: true`
- Check progress periodically with `BashOutput` tool
- This prevents blocking the conversation while waiting for long-running experiments
- Example workflow:
  1. Start experiment in background: `uv run python experiments/run_icl.py`
  2. Check progress periodically until completion
  3. Review results and wandb logs
  4. Commit any result summaries or observations

### Code Organization
```
truthification/
├── configs/           # Hydra config files
├── data/              # Data processing scripts
├── docs/              # Project documentation
├── experiments/       # Experiment scripts
├── src/               # Core source code
│   ├── truthifier/    # Truthification pipeline
│   ├── estimator/     # Truth value estimator
│   └── evaluation/    # Evaluation metrics
├── tests/             # Unit tests
└── notebooks/         # Exploration notebooks
```

## Key Research Documents

See `docs/` for background:
- `truthification-project-overview.md` - Main project plan and trust tiers
- `agentic-model-of-text-generation.md` - Theoretical foundation
- `preliminary-experiments-analysis.md` - Prior experiment analysis
- `handmade-atomic-statements-experiment.md` - Atomic statement annotation study
