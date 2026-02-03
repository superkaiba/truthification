# Truthification

Research project investigating LLM truth recovery in multi-agent games with conflicting interests.

## Overview

The **Hidden Value Game** is a multi-agent simulation where:
- **Agents** with conflicting interests try to influence an **Observer**
- Agents know the true properties of objects and a hidden value rule
- The Observer must select high-value objects without knowing the value rule
- Agents may lie, mislead, or tell the truth strategically to achieve their goals

## Key Research Questions

1. Can LLM observers infer truth from unreliable agent statements?
2. How does access to agent identities/interests affect truth recovery?
3. What role does strategic deception play in multi-agent communication?

## Quick Start

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env

# Run a game
uv run python experiments/run_v2_experiment.py

# Launch the dashboard
uv run streamlit run experiments/dashboard_v2.py

# Run comprehensive experiments
uv run python experiments/run_comprehensive_suite.py
```

## Game Mechanics

### World Setup
- **Objects** with properties: color, shape, size, material, is_dangerous
- **Hidden Value Rule**: Determines object value (e.g., "star-shaped wooden objects are valuable")
- **Agents**: Have conflicting interests (e.g., Agent_A wants blue objects, Agent_B wants red)

### Game Flow
1. Agents make statements about objects (may be true or false)
2. Observer can query an oracle for ground truth (limited budget)
3. Observer selects objects it believes are high-value
4. Metrics computed: selection accuracy, property inference, value prediction

### Fair Value Rules
Value rules use properties orthogonal to agent interests to ensure fairness:
- Agents want: `color=blue`, `color=red`, etc.
- Value rules use: `shape=star`, `material=wood`, `material=glass`

## Project Structure

```
truthification/
├── src/
│   ├── environment/          # Game simulation
│   │   ├── world_v2.py       # World, objects, value rules
│   │   ├── agent_v2.py       # LLM agents with interests
│   │   ├── simulation_v2.py  # Game orchestration
│   │   └── estimator_v2.py   # External truth estimator
│   ├── observer/             # Observer implementations
│   │   ├── observer_v2.py    # Game observer with beliefs
│   │   ├── icl.py            # ICL baseline observer
│   │   └── prompts.py        # Prompt templates
│   └── evaluation/           # Metrics
│       └── metrics.py        # Accuracy, calibration metrics
├── configs/experiment/       # Hydra configs
│   └── v2_hidden_value.yaml  # Main game config
├── experiments/              # Experiment scripts
│   ├── run_v2_experiment.py  # Main experiment runner
│   ├── run_comprehensive_suite.py  # Parameter sweeps
│   ├── dashboard_v2.py       # Streamlit visualization
│   └── test_v2_minimal.py    # Minimal test
├── tests/                    # Unit tests
└── results/                  # Experiment results
```

## Configuration

Games are configured via Hydra YAML files or Python dataclasses:

```python
from src.environment import GameConfig, HiddenValueGame

config = GameConfig(
    n_objects=10,           # Number of objects
    n_agents=2,             # Number of agents
    n_rounds=4,             # Conversation rounds
    selection_size=4,       # Objects to select
    oracle_budget=2,        # Oracle queries allowed
    rule_complexity="medium",  # simple/medium/complex
    seed=42,
)

game = HiddenValueGame(config)
result = game.run()
```

## Metrics

| Metric | Description |
|--------|-------------|
| `selection_accuracy` | Total value selected / optimal value |
| `decision_quality` | Per-round: value picked / best available |
| `property_accuracy` | Observer's property belief accuracy |
| `value_prediction_accuracy` | Observer's value estimate accuracy |
| `agent_success` | Per-agent: picks matching their interest |

## Experiment Results

Results from comprehensive parameter sweeps show:
- **More agents** reduces accuracy (99.6% with 2 agents → 74% with 4)
- **More rounds** can hurt (100% at 2 rounds → 80% at 6 rounds)
- **Rule complexity** has moderate impact
- **Oracle budget** has minimal effect when observer is already performing well

## Running Tests

```bash
uv run pytest tests/ -v
```

## Weights & Biases

Experiments log to [W&B project](https://wandb.ai/thomasjiralerspong/truthification)

## License

MIT
