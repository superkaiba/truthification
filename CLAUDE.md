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

### Environment Variables
- Store all API keys and tokens in `.env` file
- Never commit `.env` to version control
- Required variables:
  ```
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

## Code Style

- Use **ruff** for linting and formatting
- Run `ruff check .` and `ruff format .` before committing
- Type hints encouraged for public APIs

## Testing

- Use **pytest** for testing
- Run tests with `uv run pytest`
- Test files go in `tests/` mirroring `src/` structure

## Common Commands

```bash
# Setup
uv sync                          # Install dependencies
cp .env.example .env             # Create env file (then edit)

# Development
uv run python experiments/...    # Run experiment
uv run pytest                    # Run tests
ruff check . && ruff format .    # Lint and format

# Experiment tracking
wandb login                      # Authenticate wandb
wandb sync ./wandb/offline-*     # Sync offline runs
```

## Key Terminology

| Term | Definition |
|------|------------|
| **Truthification** | Process of annotating text with trust tiers and decomposing into atomic statements |
| **Truthifier** | Component that transforms natural language into truthified statements |
| **Estimator** | Model that predicts P(statement is true \| truthified context) |
| **Trust Tier** | Category indicating reliability: verified fact → scientific claim → opinion |
| **Atomic Statement** | Minimal, unambiguous assertion about a single property of the world |
| **Equivariance** | Property that shifting agent embedding toward "Truth" shifts predictions toward ground truth |

## Research Principles

### Experiment Hygiene
- **One variable at a time**: Change one thing per experiment when possible
- **Always have baselines**: Compare against non-truthified and ground-truth models
- **Log everything**: Better to have unused data than missing data
- **Seed everything**: Set random seeds for reproducibility

### What to Log in wandb
- Full config (Hydra handles this)
- Training curves (loss, accuracy per epoch)
- Evaluation metrics (epistemic correctness curves, calibration)
- Model checkpoints (at least best and final)
- Sample predictions (qualitative inspection)

### Negative Results
- **Document failures** - they're as valuable as successes
- Note hypotheses for why something didn't work
- Commit notes even for failed experiments

## Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Truthifier (inference) | 16GB RAM, CPU | 24GB VRAM GPU |
| Estimator fine-tuning (7B) | 24GB VRAM | 48GB VRAM or multi-GPU |
| Estimator fine-tuning (32B) | 80GB VRAM | Multi-node |

## Key References

### Papers
- Scientist AI safety case (Bengio et al.) - Theoretical foundation
- WebOrganizer (Wettig et al., 2025) - Data organization by source
- TruthfulQA (Lin et al., 2021) - Evaluation benchmark
- Metadata conditioning (Gao et al., 2025) - Source prefix approach

### Datasets
- Wikidata/DBpedia - Factual knowledge graphs
- TruthfulQA - Factuality benchmark
- FACTS Grounding - Evidence-based evaluation
- HalluLens - Hallucination benchmark

## Key Research Documents

See `docs/` for background:
- `truthification-project-overview.md` - Main project plan and trust tiers
- `agentic-model-of-text-generation.md` - Theoretical foundation
- `preliminary-experiments-analysis.md` - Prior experiment analysis
- `handmade-atomic-statements-experiment.md` - Atomic statement annotation study
