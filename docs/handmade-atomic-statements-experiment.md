# Handmade Atomic Statements Experiment

---

## Overview

### Purpose

This experiment has multiple humans independently parse the same text into atomic statements to:

1. Enrich understanding of challenges in splitting text into atomic statements
2. Compare outputs from different humans, identify discrepancies, discuss convergence
3. Generate agreed-upon samples for testing automated parsing pipelines

### Methodology

- Show a piece of text to multiple humans
- Each human generates atomic statements independently (previous results hidden)
- No explicit dependencies identified, but statements should be ordered so later ones can depend on earlier ones
- Compare results and analyze discrepancies

### Data Sources

1. **Wikipedia**: First paragraphs from "Today's Featured Article" (self-contained)
2. Reddit (TBD)
3. Other (TBD)

---

## Participants

- Vincent (V)
- Charlie (C)
- Gael (G)
- Alessandro (A)
- Philippe (P)
- Damiano (D) - partial, with logical formalism perspective

---

## Text 1: Clownfish

### Original Text

> Clownfish or anemonefishes (genus Amphiprion) are saltwater fish found in the warm and tropical waters of the Indo-Pacific. They mainly inhabit coral reefs and have a distinctive colouration typically consisting of white vertical bars on a red, orange, yellow, brown or black background. Clownfish developed a symbiotic and mutually beneficial relationship with sea anemones, on which they rely for shelter and protection from predators. In turn, clownfish protect the anemone from anemone-eating fish, as well as clean and fan them, and attract beneficial microorganisms with their waste.

### Statement Counts by Participant

| Participant | # Statements |
|-------------|--------------|
| Vincent | 14 |
| Charlie | 23 |
| Gael | 18 |
| Alessandro | 10 |
| Philippe | 19 |

### Key Discrepancies Analyzed

#### 1. Level of Atomicity

**First sentence decomposition:**

| Approach | Participants | Statements |
|----------|--------------|------------|
| **Coarse** | A | 2 statements (kept compound) |
| **Fine** | V, C, G, P | 3-5 statements (split everything) |

**Question raised**: Are compound propositions acceptable, or must we be maximally atomic?

#### 2. Handling "warm and tropical waters"

| Interpretation | Participants | Meaning |
|----------------|--------------|---------|
| **Qualifying** | C, G, P | "Waters ARE warm" + "Waters ARE tropical" (separate facts) |
| **Specifying** | V | "Waters THAT ARE warm and tropical" (filter on which waters) |

**Question raised**: How do we handle specifications/conditions in atomic statements?

#### 3. AND/OR Lists

**Colors enumeration:**

| Approach | Participants | Result |
|----------|--------------|--------|
| **Keep together** | V, G, A | "white bars on red, orange, yellow, brown or black" |
| **Enumerate** | C, P | 5 separate statements for each color option |

**Trade-off**:
- Together: Closed list (implies no other colors)
- Separate: More atomic, but need absence of other statements to imply closure

#### 4. Hedges

| Approach | Participant | Example |
|----------|-------------|---------|
| **Keep hedge** | V, C, G, A | "Clownfish **mainly** inhabit coral reefs" |
| **Remove hedge** | P | "Clownfish inhabit coral reefs" |

**Question raised**: Should hedges be preserved or stripped?

#### 5. Redundancy ("symbiotic and mutually beneficial")

| Approach | Participants | Handling |
|----------|--------------|----------|
| **Keep both** | V, A | "symbiotic and mutually beneficial" |
| **Remove redundancy** | P | Just "symbiotic" (implies mutual benefit) |
| **Split** | C, G | Two separate statements |

**Question raised**: Should we preserve redundancy from source or simplify?

#### 6. Pronoun Ambiguity ("on which they rely")

| Interpretation | Participants | "Which" refers to |
|----------------|--------------|-------------------|
| **Anemones** | V, A | "rely on sea anemones" |
| **Relationship** | C, G | "rely on their relationship with anemones" |

**Question raised**: How to handle pronoun ambiguity?

#### 7. Implied Information ("beneficial microorganisms")

| Approach | Participants | Result |
|----------|--------------|--------|
| **Keep vague** | A | "attract beneficial microorganisms" (beneficial to whom?) |
| **Interpret inline** | C | "attract microorganisms for anemones" |
| **Separate statement** | V, G | "microorganisms are beneficial to anemones" (new statement) |

**Question raised**: Should implied information be made explicit? If so, inline or as separate statement?

---

## Text 2: Redshift

### Original Text

> In physics, a redshift is an increase in the wavelength, or equivalently, a decrease in the frequency, of electromagnetic radiation (such as light). The opposite change, a decrease in wavelength and increase in frequency and energy, is known as a blueshift.
>
> Three forms of redshift occur in astronomy and cosmology: Doppler redshifts due to the relative motions of radiation sources, gravitational redshift as radiation escapes from gravitational potentials, and cosmological redshifts caused by the universe expanding. [...]

### Statement Counts

| Participant | # Statements |
|-------------|--------------|
| Vincent | 20 |
| Gael | 35 |
| Alessandro | 17 |
| Charlie | 20 |

### Key Discrepancies

#### 1. Context Handling ("In physics")

| Approach | Participants | Result |
|----------|--------------|--------|
| **Inline** | A, C | "In physics, a redshift is..." |
| **Separate** | V, G | "A redshift is a term used in physics" + other statements |

**Question raised**: Should context be separate dependency or inline?

#### 2. Logical Equivalence (A ↔ B)

Text says: "increase in wavelength, or equivalently, decrease in frequency"

| Approach | Participants | Statements |
|----------|--------------|------------|
| **A + (A↔B)** | A, C | State A, then state equivalence (B implied) |
| **A + B + (A↔B)** | V, G | State A, state B, state equivalence (explicit) |

**Question raised**: If B can be logically implied, must it still be stated?

#### 3. AND List Error

Text: "Three forms occur in astronomy **and** cosmology"

| Interpretation | Participants | Result |
|----------------|--------------|--------|
| **Correct (union)** | V, A | "Three forms in astronomy and cosmology" |
| **Incorrect (intersection)** | C, G | "Three in astronomy" + "Three in cosmology" |

**Error identified**: Cannot infer intersection from union statement.

#### 4. Domain Knowledge Required

Text: "gravitational redshift as radiation escapes from gravitational potentials"

| Interpretation | Participant | Result |
|----------------|-------------|--------|
| **Correct** | V, C, A | "Gravitational redshifts are due to radiation escaping..." |
| **Incorrect** | G | "Gravitational redshift escapes from gravitational potentials" (wrong physics) |

**Question raised**: How to handle ambiguities requiring domain knowledge?

---

## Text 3: Yugoslav Torpedo Boat T4

### Statement Counts

| Participant | # Statements |
|-------------|--------------|
| Vincent | 26 |
| Alessandro | 16 |
| Charlie | 26 |
| Gael | 39 |
| Philippe | 27 |
| Damiano | 8 (stopped early) |

### Key Discrepancies

#### 1. Temporal Language ("was" vs "is")

Vincent's note:
> "Does 'is' or 'was' matter? It brings time-related information which should be in another atomic statement."

#### 2. Entity Reference (T4 vs 79 T vs 79)

The boat has three names across time. Participants handled this differently:
- Some used T4 throughout
- Some switched between 79 T and T4 based on time period
- Some explicitly stated the renaming relationship

#### 3. Damiano's Logical Perspective

Damiano (with logic background) stopped early, noting:

> "A piece of text may not be uniquely decomposable into atomic statements"

Reasons:
1. Text might not be well-formed
2. Text might not be decomposable (questions, imperatives, exclamations)
3. Decomposition might not be unique (ambiguity, implicit statements)

Example of non-atomicity in "was":
> "T4 was a torpedo boat" = "∃t (t < now) : torpedo.boat(T4, t)"
> This is NOT atomic - it contains a quantifier.

---

## Summary of Open Questions

### Atomicity

| Question | Options |
|----------|---------|
| How atomic is atomic? | Compound OK vs. Maximally split |
| Purpose of atomicity? | Reducing vagueness? Enabling truth evaluation? |

### Logical Structure

| Question | Options |
|----------|---------|
| Handle AND lists? | Keep together (closed) vs. Enumerate (open) |
| Handle OR lists? | Keep together vs. Enumerate |
| Logical implications? | State all vs. State minimal |
| Higher-level logic? | Capture ("in turn") vs. Ignore |

### Language Features

| Question | Options |
|----------|---------|
| Hedges ("mainly")? | Preserve vs. Remove |
| Pronouns? | Resolve inline vs. Allow with dependency |
| Context ("In physics")? | Inline vs. Separate statement |
| Temporal ("was")? | Accept vs. Decompose |

### Knowledge & Interpretation

| Question | Options |
|----------|---------|
| External knowledge? | Allow additions vs. Stick to text |
| Implied information? | Make explicit vs. Leave implicit |
| Ambiguity resolution? | Require domain knowledge vs. Flag as ambiguous |
| Redundancy? | Preserve vs. Simplify |

### Dependencies

| Question | Options |
|----------|---------|
| Dependency types? | Just AND vs. Multiple types (context, enumeration) |
| Self-containment? | Full vs. Allow references |

---

## Errors to Avoid

1. **Intersection from union**: "A and B have property X" ≠ "A has X" + "B has X" (unless property is distributable)

2. **Meaning-breaking splits**: Don't split when word meaning depends on rest of sentence
   - Bad: "important tool" + "tool for X" (importance is relative to X)

3. **Domain-ignorant parsing**: Some ambiguities require domain knowledge
   - "redshift as radiation escapes" ≠ "redshift escapes"

---

## Implications for Truthifier Design

### What This Experiment Reveals

1. **No single "correct" decomposition**: Must make explicit design choices
2. **Trade-offs exist**: Atomicity vs. Closure, Self-containment vs. Brevity
3. **Domain knowledge helps**: But can't always assume it's available
4. **Ambiguity is pervasive**: Even simple Wikipedia paragraphs have multiple valid readings

### Recommendations for Automated Truthifier

1. **Define atomicity criteria explicitly** in the prompt/instructions
2. **Choose consistent strategies** for AND/OR, hedges, context, etc.
3. **Flag ambiguities** rather than silently choosing one interpretation
4. **Use structured output** to capture dependencies
5. **Evaluate against multiple human annotations** (not just one)

### Evaluation Approach

- Cannot expect exact match to any single human
- Should measure:
  - Coverage (are all facts represented?)
  - Correctness (are statements true?)
  - Consistency (same strategy throughout?)
  - Ambiguity handling (flagged appropriately?)
