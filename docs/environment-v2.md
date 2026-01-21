# Truthification V2: Environment Overview

## Research Question

Can an LLM observer learn truth about an environment when agents are actively trying to manipulate its beliefs?

## Environment

```
┌─────────────────────────────────────────────────────────┐
│                        WORLD                            │
│  Objects: obj_1, obj_2, ... obj_N                       │
│  Properties: color, shape, size, value, ...             │
│  Ground truth known to experimenter                     │
└─────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌──────────┐         ┌──────────┐         ┌──────────┐
    │ Agent A  │         │ Agent B  │         │ Agent C  │
    │ Knows:   │         │ Knows:   │         │ Knows:   │
    │ obj 1-30 │         │ obj 20-50│         │ obj 40-70│
    │ Goal:    │         │ Goal:    │         │ Goal:    │
    │ deceive  │         │ honest   │         │ confuse  │
    └────┬─────┘         └────┬─────┘         └────┬─────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────────────────────────────────────────────────┐
    │              STATEMENTS (strategic)                 │
    │  "obj_5 is blue" (lie)                              │
    │  "obj_25 is red" (truth)                            │
    │  "obj_45 might be green or yellow" (confusion)      │
    └─────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │                    OBSERVER                         │
    │  Inputs: statements + limited oracle                │
    │  Task: infer true world state                       │
    └─────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │                     ORACLE                          │
    │  Mode 1: Given N facts upfront                      │
    │  Mode 2: Queryable with budget                      │
    └─────────────────────────────────────────────────────┘
```

## Components

| Component | Key Properties |
|-----------|----------------|
| **World** | N objects × K properties, all configurable |
| **Agents** | Partial knowledge, manipulation goals, strategic LLM generation |
| **Oracle** | Limited ground truth (given or queryable) |
| **Observer** | LLM inferring truth from biased sources |

## Agent Goals

| Goal | Behavior |
|------|----------|
| `honest` | Tell truth |
| `make_believe` | Lie about specific facts |
| `promote_object` | Make objects seem desirable |
| `demote_object` | Make objects seem undesirable |
| `build_trust` | Appear reliable (strategic truth-telling) |
| `sow_confusion` | Maximize uncertainty |

## Oracle Modes

1. **Given**: Observer receives N ground truth facts upfront
2. **Interactive**: Observer can query oracle (budget-limited)

## Experiments

| # | Name | Variables | Question |
|---|------|-----------|----------|
| 1 | Baseline | Honesty ratio, model | Can observer infer truth without oracle? |
| 2 | Oracle Scaling | Oracle budget (0→∞) | Accuracy vs verification cost? |
| 3 | Agent Scaling | Num agents, goal diversity | How does complexity affect inference? |
| 4 | World Scaling | Num objects, statements | Context window limits? |
| 5 | Goal Detection | - | Can observer identify agent goals? |

## Configuration

All parameters via Hydra. Key knobs:

```yaml
world.num_objects: 20-500
agents.num_agents: 3-20
agents.knowledge_coverage: 0.2-1.0
agents.goals: {honest: 0.2, make_believe: 0.4, ...}
oracle.mode: "none" | "given" | "interactive"
oracle.budget: 0-100
statements.per_agent: 10-100
```

## Success Metrics

- **Accuracy**: % properties correctly inferred
- **Contested accuracy**: Accuracy where agents disagree
- **Oracle efficiency**: Accuracy gain per oracle query
- **Goal detection**: Can observer identify manipulation goals?
