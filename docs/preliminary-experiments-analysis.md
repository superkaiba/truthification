# Preliminary Experiments Analysis

**Authors of original experiments**: Francis Rhys Ward, Matt MacDermott
**Analysis by**: Vincent (with orange comments in original)

---

## Executive Summary

Preliminary truthification experiments produced **mainly non-conclusive results** but provide crucial insights for experimental design. The results don't disprove truthification - they reveal design issues that obscured the signal.

**Key takeaway**: Future experiments need:
1. Controllable synthetic environments
2. Access to output probabilities (not just binary predictions)
3. Proper baselines (especially non-fine-tuned models)
4. Clearer separation between corruption signal and agent syntax

---

## Experimental Framework

### General Approach

1. Define a learning task with inputs and true labels
2. Corrupt labels for some samples based on systematic rule
3. Add agent syntax to identify potentially corrupted data

### Three Model Types

| Model | Training Data | Syntax |
|-------|--------------|--------|
| **Ground Truth** | True labels only | Standard |
| **Non-Truthified** | Corrupted labels | Standard (no agent marker) |
| **Truthified** | Corrupted labels | High-trust + Agent syntax |

### Success Criteria

**When queried with high-trust syntax:**
- Ground truth: Works well (task is doable)
- Non-truthified: Fails (corruption hurts capability)
- Truthified: In between, closer to ground truth = better

**When queried with agent syntax:**
- Ground truth: Still predicts true labels
- Non-truthified: No change from high-trust query
- Truthified: Predicts corrupted labels (learned agent's pattern)

---

## Experiment 1: ICLR Papers

### Settings

| Aspect | Value |
|--------|-------|
| **Input (full)** | Paper content |
| **Input (toy)** | Title, year, keywords, mean score, percentile |
| **Label** | Accept/Reject |
| **Dataset** | 2018-2020 papers, top/bottom 25%, 816 balanced samples |
| **Corruption** | All 2018 papers have labels flipped |
| **Agent syntax** | "Damiano Fornasiere says this paper was [accepted/rejected]" |

### Example Categories

- **Easy**: Bottom/top 10 percentile, 2019-2020 (not corrupted, high-trust)
- **Medium**: 10-25 percentile, 2019-2020 (not corrupted, low-trust)
- **Hard**: 0-25 percentile, 2018 (corrupted, low-trust)

### Pre-Result Analysis

**Task Difficulty:**
- Full example: Difficult; pre-training likely dominates
- Toy example: Should be easy given score/percentile data

**Critical Design Flaw:**
> The date (2018) is available to BOTH non-truthified and truthified models. Both can learn: "if year is 2018, switch the label."

**Ambiguity Problem:**
> No ground truth labels for 2018 exist. The model cannot logically disentangle "Damiano is wrong" from "2018 is different." The only way to learn this is through existing priors.

### Results: Full Example

| Model | Easy | Medium | Hard | Notes |
|-------|------|--------|------|-------|
| Ground Truth | 73% | 61% | 75% | High variance on Hard (66-83%) |
| Non-Truthified | 72% | 65% | 72% | No significant difference |
| Truthified (GT syntax) | ~similar | ~similar | ~similar | |
| Truthified (Damiano) | ~similar | ±8% | -7% to -36% | Only significant result |

**Analysis:**
- No difference between ground truth and non-truthified suggests **pre-training dominates fine-tuning**
- Missing baseline: non-fine-tuned model performance
- Only significant result: Damiano syntax reduces Hard performance (model learned some corruption pattern)

### Results: Toy Example

| Model | Easy | Medium | Hard |
|-------|------|--------|------|
| Ground Truth | ~100% | ~100% | ~100% |
| Non-Truthified | ~100% | ~100% | ~0% (learned to flip 2018) |
| Truthified (GT syntax) A,C | ~100% | ~100% | ~100% |
| Truthified (GT syntax) B | ~100% | ~100% | 50% |
| Truthified (Damiano) A,B | ~0% | ~100% | ~0% (predicts Damiano) |
| Truthified (Damiano) C | ~0% | 74.7% | 19% |

**Key Finding:**
> Model B learned truth but NOT Damiano's policy.
> Model C learned Damiano's policy but NOT truth.
> "The process of establishing confidence during training and the modelling of an agent policy are not always coherent."

**Implication:** Truthification capability emergence is **stochastic**, not guaranteed even for easy tasks.

### Missing Experiments

1. ❌ Non-fine-tuned baseline
2. ❌ Training with some true 2018 labels (easier task)
3. ❌ Training only on 2019-2020, testing on 2018
4. ❌ Testing on training samples with different syntax
5. ❌ Overlap analysis (which specific predictions changed)

---

## Experiment 2: Countries

### Settings

| Aspect | Value |
|--------|-------|
| **Input** | Statements about country with ID (e.g., "Country 17 is member of EU") |
| **Label** | True/False |
| **Dataset** | 11,000 statements |
| **Corruption** | 20 confusable country pairs; border statements swapped |
| **Agent syntax** | "John Smith says that this is [true/false]" |

### Corruption Logic

- **Membership statements**: Always correct (for both confused and non-confused)
- **Border statements**: Swapped for confused countries (Austria gets Australia's borders)
- **Result**: Correct info alone insufficient to identify confused countries

### Pre-Result Analysis

**Task Difficulty:**
- Requires implicit association during fine-tuning
- "Country 17" representation must move toward "Austria" embedding
- Without fine-tuning, model should not be able to answer

**Critical Design Challenge:**
> For truthification to work, the model must:
> 1. Notice borders contradict membership
> 2. Attribute this to "John Smith" not to the country ID
> 3. Learn that John Smith's borders should map to the confused country

**Ambiguity:**
> How should model know borders (not membership) are corrupted?

### Results

| Model | Non-Confused | Confused |
|-------|--------------|----------|
| Ground Truth | 77% | 79% |
| Non-Truthified | 75% | 35% |
| Truthified (Truth syntax) | 77% | 40% |
| Truthified (John Smith) | ~same | ~same |

**Analysis:**
- No significant difference between truthified and non-truthified
- John Smith syntax has no visible effect
- **Hidden effects possible**: Binary accuracy may hide probability shifts

### Missing Experiments

1. ❌ Testing "Country 17 is Canada" (detect "always True" model)
2. ❌ Probability distributions across all countries
3. ❌ Testing confused country (does Austria → Australia?)
4. ❌ Varying true/corrupted ratio
5. ❌ Confusion matrix visualization

---

## Conclusions

### What We Learned

1. **Pre-training dominance**: Fine-tuning signal often too weak to measure
2. **Stochastic emergence**: Truthification capability appears randomly across runs
3. **Decoupled learning**: Truth learning and agent modeling can be independent
4. **Design flaws**: Corruption signal (date, country ID) available without agent syntax

### What Didn't Work

| Issue | Impact |
|-------|--------|
| No non-fine-tuned baseline | Can't isolate fine-tuning effect |
| Corruption signal in input | Both models can learn flip rule |
| Binary accuracy only | Hides probability shifts |
| OpenAI API limitations | No access to logits |
| Complex implicit tasks | Too hard to learn during fine-tuning |

### Recommendations for Future Work

1. **Synthetic controllable environment**
   - Full observability of model internals
   - Adjustable difficulty
   - Clear separation of signals

2. **Better baselines**
   - Non-fine-tuned model
   - Corruption-free subset training
   - Varying corruption ratios

3. **Richer metrics**
   - Full probability distributions
   - Confusion matrices
   - Per-sample analysis

4. **Simpler initial tasks**
   - Direct classification (not implicit association)
   - Clear corruption pattern model can detect
   - Some ground truth in corrupted domain

### Author Suggestions

Matt and Rhys suggested more capable models might help. However:
> "Our proof of concept should not require the most capable LLMs to work. Most issues are explained by experimental design, not model capability."

**Decision**: Build new synthetic experimental environment with full control and observability.

---

## Mathematical Framing Connection

### ICLR Experiment

| Theoretical | Implementation |
|-------------|----------------|
| World properties X | Paper acceptance + logic |
| Statements τ | Paper info + accept/reject |
| Agents A_P | Truth (high-trust) vs Damiano (low-trust) |
| Noisy labels T̃ | Damiano's flipped 2018 labels |

### Countries Experiment

| Theoretical | Implementation |
|-------------|----------------|
| World properties X | Country ID ↔ Name mapping |
| Statements τ | Membership + border statements |
| Agents A_P | Truth vs John Smith |
| Noisy labels T̃ | John Smith's swapped borders |

---

## Key Quotes

> "The emergence of such capability seems to be stochastic."

> "The process of establishing confidence during training and the modelling of an agent policy are not always coherent."

> "Even if the final aggregated numbers of correct True/False is the same, it is possible that the probabilities have gone higher."

> "It is our belief that most of the issues with these experiments are not only explained by the lack of capability of the model but also on the experimental design."
