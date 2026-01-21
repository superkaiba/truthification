# Truthification

Research project investigating truthification methods for LLMs - teaching models to distinguish between verified facts, scientific claims, and opinions through annotated training data and trust tier syntax.

## Setup

```bash
# Install dependencies
uv sync

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Usage

Run a simulation to generate agent statements:

```bash
python experiments/run_simulation.py
```

Configure via Hydra:

```bash
python experiments/run_simulation.py world.n_objects=50 agent.n_agents=5
```

## Project Structure

```
truthification/
├── src/
│   ├── environment/     # Multi-agent environment
│   │   ├── world.py     # World state with objects
│   │   ├── agent.py     # LLM-based agents
│   │   └── simulation.py # Statement generation
│   ├── data/            # Data processing
│   ├── observer/        # ICL and SFT experiments
│   └── evaluation/      # Metrics
├── configs/             # Hydra configs
├── experiments/         # Experiment scripts
└── tests/               # Unit tests
```

## Testing

```bash
uv run pytest
```
