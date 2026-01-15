# Truthification Project Overview

**Status**: Work-in-progress
**Target**: Proof-of-concept by April 2026

---

## Executive Summary

The Truthification project aims to build a **ScientistAI** - a prediction engine that can reliably estimate the probability that a statement is true. This addresses a fundamental limitation of current LLMs: they don't differentiate between factual knowledge, opinions, and false statements during training.

---

## Motivation

Large language models are pre-trained via self-supervision on token prediction, which:
- Does not differentiate factual knowledge from opinions or true/false statements
- Incentivizes producing plausible-sounding answers over accurate ones
- Results in hallucinations and overconfident incorrect claims
- Embeds biases from data distribution

The project builds toward the **Scientist AI safety case**: by distinguishing syntax for factual statements vs. opinions, we can achieve asymptotic guarantees of epistemic correctness.

---

## High-Level Architecture

### Four Components

1. **Truthifier**: Annotates natural language data, decomposes into statements with trust tiers
2. **Truthified Knowledge Base**: Stores statements, trust tiers, and dependencies as a DAG
3. **Estimator**: Latent variable model returning P(statement is true | truthified context)
4. **Evaluation Procedure**: Measures epistemic correctness (incorrect high-confidence claims vs. rejected queries)

### Pipeline Flow

```
Input Statement (X)
    → Truthifier (with Knowledge Base access)
    → Truthified Statement (Xt)
    → Estimator
    → Probability Estimate
    → Real-world Decision
```

---

## Key Definitions

| Term | Definition |
|------|------------|
| **Statement (x)** | A sequence of tokens interpretable as an assertion about a property of the world |
| **Truth Value (vx)** | Boolean random variable (True/False) associated with statement x |
| **Truthification t(x,C)** | Process transforming statement x with context C into truthified statement xt where vxt = 1 |
| **Query (x,y)** | Pair where x provides context for asserting P(vy = true) |
| **Property (p)** | Abstract property of the world, stored in knowledge base |

---

## Trust Tier Hierarchy

### Verified Facts (~0.1% of data)
- **Axiomatic fact**: From trusted source dataset of basic scientific knowledge
- **Proven fact**: Theorem, code output, or mathematically proven statement
- **Wrong fact**: Known false statement (synthetic, for training negatives)

### Scientific Statements (~8% of data)
- **Scientific assumption**: Reasonably true per scientific consensus
- **Experimental claim**: Supported by multiple convincing experiments
- **Scientific claim**: Made by indicated scientific source

### Non-Scientific Statements (~70% of data)
- **Common belief**: Common in non-scientific data
- **Evidential claim**: Supported by evidence (court, journalism)
- **Simple claim**: From single source

### Non-Claims (~20% of data)
- **Informative non-claim**: Contains information but makes no claim ("Hello, nice to meet you")
- **Non-informative non-claim**: No processable information (ASCII art, formatting)

---

## Challenges & Solutions

### 1. Data Collection

**Problems**:
- Quality vs. scale tradeoff
- Conflicting information between sources
- Time-dependent information
- Mixing facts with opinions
- Unstated assumptions

**Solution approach**:
- Start with high-quality domains (code, math, Wikidata)
- Use conservative labeling (prefer "A said X" over "X is true" when uncertain)
- Scale with weak labels later

### 2. Truth Representation

**Problems**:
- Time-dependent truth values ("The president is X")
- Context-dependent truth ("In Newtonian dynamics...")
- Ambiguous approximations ("Earth is a sphere")
- Theory-of-mind queries
- Paradoxes (Liar Paradox)

**Solution approach**:
- Explicit timestamps and context
- Dependencies stated explicitly
- Conservative categorization
- Let AI learn trust weights per category

### 3. Syntax Design

**Evaluated options**:

| Syntax | Pros | Cons |
|--------|------|------|
| Natural language template | Simple | No reserved tokens, verbose |
| XML-like wrapper | Reserved tokens, metadata support | No NL integration |
| Source prefix | Simple, empirical evidence | No dependencies |
| **LLM rephrasing + XML** | Best of both worlds | Prone to errors |

**Recommended syntax**:
```xml
<sep id=abc> @user is an account on <url>. <sep/>
<sep id=ghf> X is true. <sep/>
<sep id=nhe dependencies=abc,ghf> @user said "Z". <sep/>
```

### 4. Training Objective

**Goals**:
- Store core axioms in weights
- Access updateable knowledge base for facts
- Compute P(vy | truthified context)

**Evaluation focus**:
- Not just accuracy, but **confidence calibration**
- Epistemic correctness curves
- AUC ROC for probability distributions

---

## Project Stages

### Stage 1: Proof of Concept (Target: April 2026)

**Data Sources**:
- Wikidata/DBpedia/YAGO (factual knowledge)
- Wikipedia (high-quality claims)
- Reddit (uttered claims)
- TruthfulQA, FACTS Grounding, HalluLens (evaluation)

**Truthifier**: LLM prompted to divide sentences into truthified statements (no training/RAG needed)

**Estimator Options**:
1. LLM prompted for true/false token → normalized logits
2. LLM with binary classifier head (requires fine-tuning)

**Evaluation Baselines**:
- No truthification
- Trust tiers in natural language
- Trust tiers as tags
- Metadata inclusion
- Statement order shuffling
- Combinations

**Models**: Qwen-3-7B/32B

### Stage 2: Knowledge-Based Truthification

- Self-expanding knowledge base
- DAG structure for dependencies
- Properties abstracted from natural language
- Trust tier propagation through graph

### Stage 3: Scale (TBD)

### Stage 4: Extensions (TBD)

---

## Key Research Questions

1. What regularities in truth can models learn to generalize?
2. What facts should be in weights vs. knowledge base?
3. Should statements about same properties be merged into categorical variables?
4. How to handle noise in trust tier labels?
5. Does truthification increase agentic capabilities? (safety concern)

---

## Evaluation Metrics

- **Epistemic correctness**: Rate of incorrect high-confidence claims for given rejection threshold
- **Calibration**: P(correct | confidence level)
- **Permutation invariance**: P(y|x1,...,xn) = P(y|xπ(1),...,xπ(n))
- **Standard benchmarks**: TruthfulQA, FACTS Grounding, HalluLens

---

## Insights & Discussion Points

### Strengths of the Approach
1. **Principled foundation**: Grounded in the Scientist AI safety case
2. **Conservative by design**: Prefers uncertainty over false confidence
3. **Modular architecture**: Components can be developed/tested independently
4. **Scalable trust tiers**: Don't need numerical ordering, let AI learn weights

### Open Questions to Explore
1. **Negation generation**: How to systematically create false counterparts of true statements?
2. **Calibration gap**: Training uses discrete labels (0/1) but evaluation needs continuous probabilities
3. **Long-context handling**: How to maintain conditioning statement impact with large N?
4. **Cross-domain generalization**: Will scientific writing style create spurious correlations?

### Potential Risks
1. **Overfitting to syntax**: Model learns "scientific-sounding = true"
2. **Source conflation**: Model confuses source trust with content truth
3. **Agentic amplification**: Explicit source-policy links may strengthen undesired behaviors

---

## References

- Bengio et al. (n.d.) - Scientist AI framework
- Wettig et al. (2025) - WebOrganizer, data balancing
- Lin, Hilton and Evans (2021) - TruthfulQA
- Jacovi et al. (2024) - FACTS Grounding
- Bang et al. (2025) - HalluLens
- Gao et al. (2025) - Metadata conditioning
- Ren et al. (2025) - MASK benchmark
- Fan, Wang and Liu (2025) - Megascience/TextbookReasoning
