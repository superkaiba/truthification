# Experiment Log

This file tracks all experiments run in the truthification project.

---

## 2026-02-04: Debate Structure Experiment

**Goal:** Test how debate turn structures and oracle timing affect truth recovery

**Config:**
- 20 objects, 20 rounds, 1 pick/round
- 8 conditions (4 turn structures × 2 oracle timings)
- 1 seed per condition (seed=42)

**Duration:** 2.3 hours

**Key Result:** `after_statements` oracle timing outperforms `before_response` (20% vs 12.75% avg). Best condition: interleaved + after_statements (26%).

**Results:** [results/debate_structure_experiment/README.md](debate_structure_experiment/README.md)

**Raw Data:** `outputs/debate_structure_test/20260204_183117/`

---

## Template for New Experiments

```markdown
## YYYY-MM-DD: Experiment Name

**Goal:** [What are you testing?]

**Config:**
- [Key parameters]

**Duration:** [How long did it take?]

**Key Result:** [One sentence summary]

**Results:** [Link to results README]

**Raw Data:** [Path to outputs]
```
