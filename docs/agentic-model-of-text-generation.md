# Agentic Model of Text Generation

*Note: Orange text in original = Vincent's comments*

---

## Overview

This document provides a theoretical framework for understanding how text on the internet is generated and how this understanding can be leveraged to build a truth estimator. The key insight is that text is produced by agents with specific world models and objectives, and by modeling this process, we can disentangle factual content from agent-specific biases.

---

## The Causal Graph

### Variables

| Variable | Symbol | Description | Observable from Internet Data? |
|----------|--------|-------------|-------------------------------|
| World State | X | True state of the world (temporary/permanent properties, dynamics) | No (except timestamp t) |
| Setting | S | Platform context (format conventions, community, affordances) | Yes (identifiable for most texts) |
| Real Agent | A_R | Actual person/institution/bot authoring text | No |
| Proxy Agent | A_P | Virtual user identity (username, persona, style) | Yes (identifiable for most texts) |
| Agent's World Model | X̂_A | Agent's beliefs/knowledge about X | No |
| Text | τ | The generated text [a⁰, ..., aᴷ] | Yes |

### Causal Structure

```
X (world state)
├── S (setting) ─────────────┐
├── A_R (real agent) ────────┼──→ A_P (proxy agent)
│   └── X̂_A (world model) ───┴──→ τ (text)
└── t (timestamp)
```

### Generative Equations

**Proxy Agent Formation:**
```
A_P = v(A_R, S, N_v)
```
The proxy agent is a function of the real agent, the setting, and noise.

**World Model Formation:**
```
X̂_A = z(X, A_R, N_X)
```
The agent's world model comes from observations of X, filtered through A_R's capabilities, plus noise.

**Text Generation (High-level):**
```
τ = Π(A_P, X̂_A, N_τ)
```
Text is produced by a policy function given the proxy agent, world model, and stochastic noise.

**Text Generation (Token-level):**
```
aᵏ = π(A_P, X̂_A, N_τ, [a⁰, ..., aᵏ⁻¹])
```
Each token is generated given the context and all previous tokens.

**World Update:**
```
X_{i+1} = f_P(X_i, τ)
```
Publishing text can affect the world state.

---

## What LLMs Currently Do

### The Marginalization Problem

LLMs are trained on datasets {[a⁰, ..., aᴷ]} **without** information about A_P, S, or t. Training on next-token prediction learns:

```
π_llm([a⁰, ..., aᵏ⁻¹])
```

This is equivalent to **marginalizing** the true policy π over A_P and X̂_A:
- Must build latent models of A_P and X̂_A from context alone
- Learns X̂_llm (world model in weights) following: p(X | {[a⁰, ..., aᴷ]})

### Accessing LLM World Knowledge

The most direct approach: condition on sequences from "reliable users" A_V who express X directly. This is **prompt engineering via personification** - but it's indirect and unreliable.

Fine-tuning (SFT/RLHF) further blurs this process by adjusting π_llm toward additional objectives.

---

## What Agent Identification Enables

### Richer Dataset Structure

Instead of {[a⁰, ..., aᴷ]}, we have:
```
{(τ, A_P, S, t)} = {([a⁰, ..., aᴷ], A_P, S, t)}
```

### Benefit 1: Better World State Estimation

The model learns X̂_SAI following:
```
p(X | {(τ, A_P, S, t)})
```

This has **lower entropy** than p(X | {τ}) as long as A_P, S, t contain information about X not present in τ alone.

### Benefit 2: Agent Model for Truth

Learning the conditioned policy:
```
π_ag([a⁰, ..., aᵏ⁻¹], A_P, S, t)
```

**Reduces marginalization** compared to π_llm:
- A_P and S are now identified (not marginalized)
- t is always part of X̂_A, so partial information about world model

### The Truth Agent

Define a special agent-setting pair:
```
(A_P, S) = Truth
```

This represents an agent we trust to:
1. Have an accurate world model X̂_A ≈ X
2. Have an honest objective to state the truth

Querying π_ag([a⁰, ..., aᵏ⁻¹], Truth, t) gives text τ close to X.

---

## Equivalence with Truth Estimator

### The Core Goal

We only care about π_ag(·, Truth, t). We don't need to model all agents - just extract from {(τ, A_P, S, t)} what's relevant for truth.

### The Noisy Label Problem

**What we want:** Labels T (true/false) for each statement τ
**What we have:** Labels T̃ = "truth according to A_P in S" (systematically biased)

### The Equivariance Hypothesis

Train classifier:
```
T̂ = f(τ, A_P, S, t)
```

**Key hypothesis:** When we query f(τ, Truth, t), the value T̂ ≈ T (ground truth).

For this to work, f must be **equivariant**: shifting the embedding (A_P, S) → Truth must also shift T̃ → T.

### Conditions for Equivariance to Emerge

| Scenario | What Happens | Outcome |
|----------|--------------|---------|
| **Strong Truth support** | Model learns relationship between Truth and other (A_P, S) | ✓ Good |
| **Weak Truth support, high diversity** | Model learns inter-agent mappings | ✓ Acceptable |
| **Low diversity, high capacity** | Model memorizes agent-specific knowledge | ✗ Poor generalization |

### Practical Implementation

Train π_ag to complete:
```
"Answer by True or False: τ, given A and B, is: "
```

Evaluate probabilities of next token being "True" or "False".

This is equivalent to training truth estimator p(T | τ, A, B) by MLE.

---

## Additional Learning Paths

### Bidirectional Learning

Since S and A_P are part of X:
- Observing (A_P, S, τ) helps learn about X
- Learning about X helps understand A_P, S, A_R

### Implementation Requirement

For bidirectional learning in transformers: embeddings of A_P and S must relate to their described properties in the dataset. This should emerge from training but is worth validating.

---

## Agentic Danger

### The Risk

While we don't **intend** to train π_ag(·, A_P, t) for all agents, the dataset **enables** imitation learning of specific agent policies.

### Why This Could Be Dangerous

Compared to standard LLMs, agent-conditioned policies could be:
- **More consistent**: Following specific objectives rather than averaged behavior
- **More capable**: Better performance on objective-aligned tasks
- **Persona-controllable**: Could instantiate dangerous personas

### Mitigation Considerations

This is an open concern - the same data that enables truth estimation also enables targeted agent imitation.

---

## Key Insights & Discussion Points

### Theoretical Contributions

1. **Formalization of the generation process**: Clear causal graph from world state to text
2. **Diagnosis of LLM limitations**: Marginalization over agents loses information
3. **Equivariance as the key property**: The mathematical requirement for truth extraction to work

### Critical Assumptions

1. **Truth agent exists and is identifiable**: We can define what (A_P, S) = Truth means
2. **Equivariance emerges from MLE**: This is a hypothesis, not guaranteed
3. **Sufficient Truth support**: Need enough grounding samples

### Open Questions

1. How many "Truth" samples are needed for reliable equivariance?
2. How do we validate that equivariance has emerged?
3. Can we measure the quality of agent embeddings independently?
4. How do we prevent misuse for agent imitation?

### Connection to Truthification Project

This document provides the **theoretical foundation** for why truthification should work:
- Trust tiers ≈ different (A_P, S) pairs with known reliability
- Truthification syntax ≈ making A_P and S explicit in training data
- Estimator ≈ f(τ, A_P, S, t) queried with Truth

The truthification project is an **implementation** of this theoretical framework.

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| X | True world state |
| S | Setting (platform context) |
| A_R | Real agent |
| A_P | Proxy agent |
| X̂_A | Agent's world model |
| τ | Text = [a⁰, ..., aᴷ] |
| t | Timestamp |
| π | Token-level policy |
| Π | High-level (text-level) policy |
| N_* | Noise variables |
| T | True label (ground truth) |
| T̃ | Noisy label (agent's claim) |
| T̂ | Predicted label |
| Truth | Trusted (A_P, S) pair |
