# V2 Hidden Value Game: Comprehensive Test with Opus 4.5

## Research Question

How do information conditions and rule complexity affect an LLM observer's ability to make good decisions when receiving strategic (potentially deceptive) information from agents with conflicting interests?

## Experimental Setup

**Model**: Claude Opus 4.5 (`claude-opus-4-5-20251101`) for both agents and observer

**Parameters:**
| Parameter | Value |
|-----------|-------|
| `n_objects` | 20 |
| `n_agents` | 2 (conflicting: blue vs red) |
| `n_rounds` | 3 |
| `oracle_budget` | 5 |
| `selection_size` | 5 |
| `seeds` | [42, 123, 456] |

**Experiments Run:**
1. Information Conditions: blind, ids, interests
2. Rule Complexity: simple, medium, complex

Total games: 18 (3 seeds x 6 conditions)

## Results

### Experiment 1: Information Conditions

| Condition | Selection Acc | Prop Acc | Rule Acc |
|-----------|---------------|----------|----------|
| blind | 0.391 | 0.497 | 0.000 |
| **ids** | **0.566** | **0.526** | 0.000 |
| interests | 0.460 | 0.514 | 0.000 |

**Key Finding**: Knowing agent IDs helps significantly (+17.5%), but knowing agent interests actually hurts performance compared to just IDs (-10.6%).

### Experiment 2: Rule Complexity

| Complexity | Selection Acc | Prop Acc | Rule Acc |
|------------|---------------|----------|----------|
| simple | 0.491 | 0.423 | 0.000 |
| **medium** | **0.635** | **0.511** | 0.000 |
| complex | 0.317 | 0.389 | 0.000 |

**Key Finding**: Medium complexity rules yield the best performance (63.5%), not simple rules (49.1%). Complex rules are hardest (31.7%).

## Interpretation

### Information Conditions

The non-monotonic relationship (ids > interests > blind) suggests:

1. **Agent IDs enable pattern detection**: When the observer can track which agent says what, it can detect inconsistencies and biases over multiple rounds.

2. **Interests disclosure backfires**: When agents know the observer knows their interests, they may employ more sophisticated deception strategies. Alternatively, the observer may over-correct by dismissing all statements from self-interested agents, losing useful information.

3. **Blind condition is worst**: Without any agent metadata, the observer can't distinguish or weight sources.

### Rule Complexity

The inverted-U pattern (medium > simple > complex) suggests:

1. **Simple rules lack signal**: With only one property mattering (e.g., "red is valuable"), there's limited information for the observer to exploit. Strategic agents can easily muddy the waters.

2. **Medium rules provide structure**: Multiple interacting factors create patterns that an intelligent observer can detect, even under deception.

3. **Complex rules overwhelm**: Too many factors and interactions make it hard to form coherent hypotheses, especially with limited oracle budget.

### Rule Inference

All conditions showed 0% rule inference accuracy, meaning the observer never explicitly identified the correct value rule. However, selection accuracy varied significantly, suggesting the observer used implicit heuristics rather than explicit rule understanding.

## Comparison to Minimal Test

| Metric | Minimal Test (Sonnet) | Comprehensive (Opus) |
|--------|----------------------|----------------------|
| Selection Acc (ids) | 13% | 57% |
| Property Acc | 62% | 53% |
| Total Games | 1 | 18 |

Opus 4.5 achieved much better selection accuracy than the minimal Sonnet test, suggesting model capability significantly affects deception resistance.

## Files

- `all_results.json` - Complete results with all metrics
- `information_conditions.json` - Detailed info condition results
- `rule_complexity.json` - Detailed complexity results
- `summary.md` - Auto-generated summary table

## Next Steps

1. **Oracle scaling experiment**: Test oracle budgets [0, 5, 10, 20] to measure information-value tradeoff
2. **Agent capability asymmetry**: Test different models for agents vs observer
3. **Multi-round dynamics**: Analyze how accuracy changes across rounds
4. **Truth detection**: Add metrics for correctly identifying lies vs truths
