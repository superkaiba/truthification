# ICL Experiment Design

## Overview

**Goal**: Test whether conditioning on agent identity improves truth recovery from unreliable sources.

**Core hypothesis**: An LLM observer can learn to extract ground truth by recognizing patterns in who says what, even without explicit reliability labels.

**Experimental setup**:
- Generate a world with 20 objects, 4 properties each
- 3 agents with tasks and cooperative/adversarial relationships
- Agents (Opus 4.5) generate ~100 natural language statements
- Some statements are truthful, some are strategic lies
- Observer LLM answers True/False queries about object properties
- Compare performance across 6 information conditions and 3 observer models

**Key experimental variables**:
- **Independent variable 1**: Information condition (what the observer sees)
- **Independent variable 2**: Observer model capability (Haiku vs Sonnet vs Opus)
- **Dependent variables**: Accuracy, calibration (ECE), learned reliability correlation

**Scale**:
- 6 conditions × 3 observers × 3 seeds = 54 experimental runs
- 80 queries per run = 4,320 observer API calls total
- ~300 statement generation calls (100 per seed)

---

## Experimental Conditions

### Baseline Conditions

**Condition 1 - No IDs (majority voting baseline)**
```
Statements about Object 5:
- "Object 5 is a nice red circle."
- "That object? It's blue, nothing special."
- "I saw Object 5, it's definitely red."

Question: Is Object 5 red? True/False
```

**Condition 2 - Agent IDs only**
```
Statements about Object 5:
- Agent_A: "Object 5 is a nice red circle."
- Agent_B: "That object? It's blue, nothing special."
- Agent_C: "I saw Object 5, it's definitely red."

Question: Is Object 5 red? True/False
```

**Condition 3 - Agent IDs + task descriptions**
```
Statements about Object 5:
- Agent_A (task: collect red objects): "Object 5 is a nice red circle."
- Agent_B (task: collect red objects): "That object? It's blue, nothing special."
- Agent_C (task: find large objects): "I saw Object 5, it's definitely red."

Question: Is Object 5 red? True/False
```

### Oracle Conditions (upper bounds)

**Condition 4 - Reliability scores**
```
- Agent_A (reliability: 92%): "Object 5 is a nice red circle."
- Agent_B (reliability: 35%): "That object? It's blue, nothing special."
```

**Condition 5 - Relationship descriptions**
```
- Agent_A (cooperative, tells truth about color): "Object 5 is red."
- Agent_B (adversarial, lies about color): "It's blue."
```

**Condition 6 - Truth labels (ceiling)**
```
- Agent_A [TRUE]: "Object 5 is a nice red circle."
- Agent_B [FALSE]: "That object? It's blue, nothing special."
```

**Expected ordering (hypothesis)**:
```
Condition 1 < 2 < 3 < 4 ≈ 5 < 6
```

---

## Query Format & Response Parsing

**Query structure**:
```
[System prompt explaining the task]

[Evidence block - formatted per condition]

Question: Is {object_id} {property_value}? (e.g., "Is Object 5 red?")

Use the submit_answer tool to provide your answer.
```

**Structured output via tool use**:
```python
tools = [{
    "name": "submit_answer",
    "description": "Submit your True/False answer",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "boolean"},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "reasoning": {"type": "string"}
        },
        "required": ["answer", "confidence"]
    }
}]

# Force tool use
tool_choice = {"type": "tool", "name": "submit_answer"}
```

**Response example**:
```json
{
    "answer": true,
    "confidence": 75,
    "reasoning": "Two agents say red, one says blue. Agent_B seems less reliable."
}
```

---

## Metrics & Evaluation

### Primary Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy (overall) | correct / total | Primary comparison |
| Accuracy (contested) | correct / contested | Hard cases where agents disagree |
| Accuracy (unanimous) | correct / unanimous | Easy cases, sanity check |

### Calibration Metrics

| Metric | Purpose |
|--------|---------|
| ECE (Expected Calibration Error) | Does 80% confidence → 80% correct? |
| Brier score | Combined accuracy + calibration |

### Analysis Metrics

| Metric | Purpose |
|--------|---------|
| Reliability correlation | Correlate model's implicit trust with actual agent reliability |
| Accuracy by property | Does model do better on some properties? |
| Accuracy by relationship | Easier when evidence is mostly cooperative? |

### Query Categories
- **Contested**: Agents disagree on the property value
- **Unanimous true**: All agents tell truth
- **Unanimous false**: All agents lie

### Statistical Analysis
- Mean ± std across 3 seeds
- Paired t-tests between conditions
- Plot accuracy vs condition as bar chart with error bars

---

## Model Configuration

**Agents** (statement generation):
- Model: `claude-opus-4-5-20250514`
- High-quality, realistic statements

**Observers** (truth recovery):
- Haiku 4.5: `claude-haiku-4-5-20250514`
- Sonnet 4.5: `claude-sonnet-4-5-20250514`
- Opus 4.5: `claude-opus-4-5-20250514`

**Seeds**: 3 (for statistical significance)

---

## Implementation Architecture

### New Files

```
src/
├── observer/
│   ├── __init__.py
│   ├── icl.py              # ICL observer that queries LLM
│   └── prompts.py          # Prompt templates for each condition
├── evaluation/
│   ├── __init__.py
│   └── metrics.py          # Accuracy, ECE, Brier score
├── data/
│   ├── __init__.py
│   └── generator.py        # Generate datasets with controlled params

configs/
├── experiment/
│   └── icl_baseline.yaml   # Experiment configuration

experiments/
├── run_icl.py              # Main experiment runner
```

### Key Components

1. **ICLObserver** (`src/observer/icl.py`)
   - Takes evidence + query, returns structured answer
   - Supports all 6 conditions via prompt formatting
   - Configurable model (Haiku/Sonnet/Opus)

2. **PromptBuilder** (`src/observer/prompts.py`)
   - Formats evidence for each condition
   - Builds system prompt + user prompt

3. **Metrics** (`src/evaluation/metrics.py`)
   - `compute_accuracy(predictions, ground_truth)`
   - `compute_ece(predictions, confidences, ground_truth)`
   - `categorize_queries(statements, world)` → contested/unanimous

4. **ExperimentRunner** (`experiments/run_icl.py`)
   - Orchestrates: generate data → run conditions → compute metrics
   - Logs everything to wandb

---

## Experiment Workflow

```
1. GENERATE DATA (once per seed)
   └─→ Run simulation with Opus 4.5 agents
   └─→ Save: world_state, statements, agent_metadata
   └─→ Output: data/seed_{N}/simulation.json

2. BUILD QUERIES (once per seed)
   └─→ Generate all (object, property, value) queries
   └─→ Categorize: contested vs unanimous
   └─→ Output: data/seed_{N}/queries.json

3. RUN CONDITIONS (6 conditions × 3 observers × 3 seeds)
   └─→ For each condition:
       └─→ Format evidence per condition template
       └─→ Query observer with tool use
       └─→ Collect: answer, confidence, reasoning
   └─→ Output: results/seed_{N}/{condition}_{observer}.json

4. COMPUTE METRICS
   └─→ Accuracy (overall, contested, unanimous)
   └─→ ECE, Brier score
   └─→ Aggregate across seeds (mean ± std)

5. LOG TO WANDB
   └─→ Metrics table
   └─→ Accuracy bar chart by condition
   └─→ Calibration plot
   └─→ Raw predictions for analysis
```

---

## Cost Estimate

| Component | Calls | Model | Est. Cost |
|-----------|-------|-------|-----------|
| Statement generation | ~300 | Opus 4.5 | $15-30 |
| Observer (Haiku) | ~1,440 | Haiku 4.5 | $1-2 |
| Observer (Sonnet) | ~1,440 | Sonnet 4.5 | $10-15 |
| Observer (Opus) | ~1,440 | Opus 4.5 | $50-80 |
| **Total** | | | **$80-130** |

**Runtime**: ~2-3 hours (mostly API latency)

---

## Success Criteria

**Minimum viable result**:
- Condition 2 > Condition 1 (agent IDs help)
- Clear ordering across conditions

**Strong result**:
- Condition 3 approaches Condition 4/5 (task info as good as reliability)
- Larger models perform better
- Good calibration (ECE < 0.1)

**Stretch**:
- Can extract implicit reliability scores from model reasoning
- Performance generalizes across seeds
