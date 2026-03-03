# Can You Infer What Someone Wants by Watching Them Argue?

**TLDR:**
- We built a game where LLM agents with hidden objectives debate to influence a judge, then tested whether a separate LLM observer can infer those objectives from the debate transcript.
- **What the agent does matters most.** Agent communication strategy produces a 38pp spread in inference accuracy (aggressive: 55% vs misdirection: 17%). Misdirection beats outright lying for concealing objectives.
- **What the observer is doesn't matter.** Haiku 4.5 ($0.25/MTok) matches Opus 4.6 ($15/MTok) — no significant differences across 7 models.
- **More debate doesn't help.** Inference peaks at ~24 statements then plateaus or drops, because agents become more deceptive over time.
- **Verification is critical.** Forced oracle access nearly triples property accuracy (25% → 71%), but LLMs won't use verification tools voluntarily.
- **Telling the observer about strategic communication helps.** Brief theory context (+13pp) and consistency-checking prompts (+8pp) both improve inference — but more elaborate instructions don't help further.
- The main takeaway: **the information environment matters far more than observer capability**. Efforts to improve truth recovery should focus on incentive structures, not scaling.

---

## Motivation

This work is part of the [truthification project](https://github.com/superkaiba/truthification), which aims to build systems that reliably estimate statement truth by conditioning on source identity and trust.

The core theoretical insight ([agentic model of text generation](agentic-model-of-text-generation.md)): text on the internet is produced by agents with specific objectives. Standard LLMs are trained on raw text, which **marginalizes over** author identity and goals — the same sentence means different things depending on who said it and why. Truthification proposes to reverse this by explicitly conditioning on agent identity.

But this raises a prerequisite question: **can we even identify what agents want from observing their behavior?** That's what we test here.

This also connects to **contextualization** — the problem that "this product is excellent" carries entirely different information depending on whether it comes from a paid reviewer or a verified purchaser. If we can infer agent objectives, we can reason about how those objectives distort their communication.

## Related Work

**Cheap talk** (Crawford & Sobel, 1982): with misaligned incentives, agents can only credibly transmit coarse information. Our environment instantiates this. **IRL** (Ng & Russell, 2000): inferring objectives from behavior, but we observe communication rather than actions. **AI debate** (Irving et al., 2018): debate as alignment; we empirically measure when this works and fails. **Deception detection** (Pérez-Rosas et al., 2018): we study strategic, adaptive deception by capable LLM agents rather than static text classification.

## Setup: The Hidden Value Game

We designed a controlled environment with these components:

- **World**: 10 objects with properties (color, shape, material, etc.) and a hidden value rule
- **2 LLM agents** (Claude Sonnet 4): each has a private value function over object properties (e.g., "cares about: color=blue, shape=star"). They don't know each other's objectives
- **Judge**: an LLM that selects objects over 10 rounds based on agent debate, with a limited oracle budget for ground-truth verification
- **Estimator**: a separate, passive LLM observer that reads the full transcript and infers each agent's hidden value function
- **Metric**: **Exact F1** — the estimator predicts which (property, value) pairs each agent cares about (e.g., "color=blue"), and we compute F1 against the ground truth. If an agent cares about {color=blue, shape=star, material=wood} and the estimator predicts {color=blue, shape=star, size=large}, that's precision 2/3, recall 2/3, F1 = 66.7%

We tested 7 communication strategies: natural, honest, deceptive, misdirection, aggressive, subtle, and credibility attack.

Across 11 experiments and 500+ games, we varied agent strategy, oracle budget, estimator model, statement count, theory context, deception detection strategy, agent/round count, and objective complexity.

## Results

### Agent strategy is the dominant factor

Across 70 games, communication strategy produced a **38pp spread** — the largest effect of any variable tested.

| Strategy | Exact F1 | vs Natural |
|----------|----------|------------|
| **Aggressive** | **55.0%** | +23.3% (d=1.07) |
| Honest | 46.7% | +15.0% (d=0.77) |
| Subtle | 35.0% | +3.3% |
| Natural | 31.7% | — |
| Credibility Attack | 30.6% | -1.1% |
| Deceptive | 24.1% | -7.6% |
| **Misdirection** | **16.7%** | -15.0% (d=-0.77) |

![Agent Strategy Effect](../results/plots/fig6_agent_strategy_inference.png)

Aggressive advocacy makes objectives obvious — the estimator just identifies which objects each agent pushed. **Misdirection beats outright deception** for concealment because lying still draws attention to the right dimensions, while misdirection shifts attention to irrelevant ones.

Strategy also affects game outcomes. Honest agents maximize judge value (191.8) but sacrifice agent reward (10.6), while aggressive agents maximize their own reward (11.9) at the judge's expense.

### Oracle access is transformative

Forced oracle queries nearly triple property accuracy:

| Condition | Property Accuracy |
|-----------|-------------------|
| No Oracle | 25.3% ± 7.6% |
| Forced Oracle | 70.7% ± 10.4% |

![Oracle Effect](../results/plots/fig1_forced_oracle.png)

Optimal oracle budget is ~6 queries (27.1% F1), with diminishing returns beyond that — agents learn to "spin" oracle results (422 documented instances across 56 games).

Critical finding: **LLMs don't voluntarily use verification tools.** When oracle use was optional, judges declined to query. Verification must be mandatory.

### Model capability doesn't matter

7 Claude models tested on the same 10 transcripts (within-subjects):

| Model | Exact F1 |
|-------|----------|
| Haiku 4.5 | 43.3% |
| Opus 4/4.5/4.6 | 40.0% |
| Sonnet 4/4.5 | 38.3% |
| Sonnet 4.6 | 36.7% |

All p > 0.34. **The cheapest model ties the most expensive.** The bottleneck is information structure, not model capability — if the signal isn't in the transcript, no model will find it.

### More statements don't help

F1 at 8 checkpoints from 6 to 48 statements:

![F1 Evolution](../results/plots/fig9_f1_evolution.png)

Peak at 24 statements (50.0%), with a drop at 42 (35.0%). No monotonic improvement. Early statements are most informative; agents become more deceptive over time. Our qualitative analysis confirms: credibility attacks go from 3/10 agents in round 1 to 9/10 by round 10.

### Theory context and deception detection help — but only a little

Telling the estimator about strategic communication improves inference. We tested providing theoretical background (Crawford-Sobel signaling theory, inverse reinforcement learning, bias correction) at varying lengths:

| Context | Exact F1 | vs None |
|---------|----------|---------|
| None | 33.3% | — |
| Brief (~50 words) | 46.7% | +13.4pp |
| Full (~200 words) | 40.0% | +6.7pp |
| Comprehensive (~5000 words) | 43.3% | +10.0pp |

![Theory Context Effect](../results/plots/fig5_theory_context.png)

Brief context outperforms longer documents — concise priming beats comprehensive instruction.

We also tested 5 deception detection strategies for the estimator:

| Strategy | Exact F1 | vs Baseline |
|----------|----------|-------------|
| Baseline | 36.7% | — |
| Consistency checking | 45.0% | +8.3pp |
| Incentive analysis | 41.7% | +5.0pp |
| Pattern recognition | 41.7% | +5.0pp |
| Combined (all above) | 45.0% | +8.3pp |

![Deception Strategies](../results/plots/fig7_deception_strategies.png)

Consistency checking — flagging contradictions across an agent's statements — performs best. But combining all strategies provides **no benefit over consistency checking alone**. More complex prompts don't help.

## Emergent Manipulation Strategies

Without any instruction to deceive, LLM agents spontaneously developed sophisticated manipulation strategies:

1. **Fabricated terminology** (122 instances): inventing fake properties like "azure crystalline" to create an illusion of insider knowledge
2. **Truth mixed with lies** (universal): accurately describing the real scoring rule while falsely claiming their preferred objects qualify
3. **Oracle spin** (422 instances): immediately reframing any oracle revelation to support their existing narrative
4. **Credibility attacks** (342 instances): escalating from polite disagreement to aggressive accusations over the debate
5. **Escalating complexity**: introducing progressively harder-to-verify claims to create cognitive overload

These emerge purely from incentive structure and intensify over rounds.

## Future Work

**Fine-tuning an objective estimator.** Our estimator uses in-context learning. Fine-tuning on many trajectories with known ground-truth objectives could teach a model to detect deception patterns more reliably than prompted baselines.

**Objective understanding for imitation.** If we fine-tune a model on an agent's actions, does understanding their objective help it generalize? We hypothesize yes:

> **The most efficient way to imitate someone is to understand their objective.**

Behavioral cloning copies (state, action) pairs. Objective-conditioned imitation captures *why* the agent acts, enabling transfer to new situations. The quality of objective inference (F1) should correlate with imitation quality.

**From objective inference to truthification.** This work validates a prerequisite — agent objectives are inferable from behavior. The path forward: objective inference → bias correction → trust calibration → full truthification (conditioning on source identity to produce calibrated truth estimates).

---

*Code and all experimental results: [github.com/superkaiba/truthification](https://github.com/superkaiba/truthification). Experiments used Claude models (Haiku 4.5 through Opus 4.6) with results logged to [Weights & Biases](https://wandb.ai/thomasjiralerspong/truthification).*
