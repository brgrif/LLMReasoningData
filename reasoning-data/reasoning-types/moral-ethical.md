# Moral-Ethical Reasoning

## Definition

Moral-ethical reasoning weighs a dilemma across competing ethical frameworks, articulating the strongest case for each defensible position without collapsing to a single predetermined answer. The quality lies in the reasoning: coverage of perspectives, internal consistency, and honest tension, not in landing on one verdict. This type is rubric-judged only and is permanently excluded from verifiable-reward training.

## Canonical example

**Problem:** A hospital has one ventilator and two patients who both need it to survive: one is younger with a better prognosis, one arrived first. How should the decision be reasoned about?

**Trace:**
1. Frame the tension: outcome-maximizing allocation versus fairness of first-come or equal claim.
2. Consequentialist view: allocate to maximize expected life-years, favoring the better prognosis.
3. Deontological / fairness view: a prior claim or equal moral worth resists ranking people by outcome; a lottery may respect equality.
4. Consistency check: whichever principle is chosen must be one the decider would accept applied to themselves and applied across future cases.
5. Note the residual: any choice leaves a real moral cost; the reasoning names it rather than hiding it.

**Answer:** Both allocations are defensible; a consequentialist case favors the better prognosis, a fairness case favors an equal-chance procedure. The decision turns on which principle the institution adopts and applies consistently.

## What good training data looks like

Dilemmas with multiple genuinely defensible positions across several ethical frameworks (consequentialist, deontological, virtue, fairness). The data rewards reasoning that covers the major perspectives, states each fairly, and stays internally consistent. It never encodes a predetermined "right" conclusion. Diversity of dilemma structure matters so the model does not learn one template.

## Verification method

Rubric judge only (required for this type by SCHEMA.md). An LLM judge scores three things: reasoning quality, perspective coverage (are the major defensible positions represented), and internal consistency (does the trace apply its principles evenly). The judge never scores toward a fixed conclusion. Because the ceiling is a rubric, this type feeds SFT and preference data but never verifiable-reward training.

## Difficulty ladder

1. Two clearly opposed positions, one dilemma axis.
2. Two positions with a subtle consistency requirement.
3. Three frameworks in tension over one dilemma.
4. Nested obligations where perspectives partly overlap and partly conflict.
5. A dilemma where every framework has internal tension and the strongest reasoning must acknowledge irreducible conflict.

## Traps

- **Sycophancy.** The model tailors its verdict to the perceived preferred answer. Counter with a rubric that rewards balance and perspective coverage, not agreement.
- **Reward hacking.** The model games the judge by asserting neutrality without real reasoning. Counter by scoring the substance of each perspective's case, not surface hedging.
- **Framework monoculture.** Only one ethical lens is applied. Counter by scoring perspective coverage explicitly.

## Sample record

```json
{
  "id": "mor-000090",
  "reasoning_type": "moral-ethical",
  "domain": "ethics",
  "problem": "A hospital has a single ventilator and two patients who each need it to survive: one is younger with a markedly better prognosis, the other arrived first. How should the allocation decision be reasoned about?",
  "reasoning_trace": [
    {"step": 1, "text": "Name the tension: maximizing outcomes versus fairness of prior claim and equal worth.", "label": "valid"},
    {"step": 2, "text": "Consequentialist case: allocate to maximize expected life-years, favoring the better prognosis.", "label": "valid"},
    {"step": 3, "text": "Fairness/deontological case: prior claim or equal moral worth resists ranking by outcome; a lottery respects equality.", "label": "valid"},
    {"step": 4, "text": "Consistency check: the chosen principle must be acceptable applied to oneself and across future cases.", "label": "valid"},
    {"step": 5, "text": "Acknowledge the residual moral cost of either choice rather than concealing it.", "label": "valid"}
  ],
  "final_answer": "Both allocations are defensible: a consequentialist case favors the better prognosis, a fairness case favors an equal-chance procedure. The decision turns on which principle the institution adopts and applies consistently.",
  "is_correct": true,
  "difficulty": 3,
  "generation_method": "human_expert",
  "verification": {"method": "rubric_judge", "passed": true, "details": "Rubric scored perspective coverage (two frameworks), reasoning quality, and internal consistency as high; no predetermined conclusion was scored."},
  "provenance": {"source": "expert-elicitation", "seed_id": null, "created": "2026-01-28"},
  "notes": "excluded from verifiable-reward training per TRAINING.md; rubric_judge required for this type"
}
```
