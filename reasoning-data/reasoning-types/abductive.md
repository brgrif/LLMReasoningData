# Abductive Reasoning

## Definition

Abductive reasoning infers the most plausible explanation for an observation: inference to the best explanation. Given a surprising state of affairs, it proposes the cause that best accounts for it and rules out weaker alternatives. The answer is the most plausible explanation, not a certainty, so calibrated confidence matters.

## Canonical example

**Problem:** A sealed greenhouse was watered and closed on Friday. On Monday the soil is dry and the floor near the door is damp. What most plausibly happened?

**Trace:**
1. Observation: soil dry, floor damp near the door, over a weekend in a sealed greenhouse.
2. Candidate A: normal evaporation. Explains dry soil but not damp floor concentrated at the door.
3. Candidate B: a leak at the door seal let water drain out. Explains both the dry soil and the localized damp floor.
4. Candidate C: someone entered and spilled water. Possible but the greenhouse was sealed; no entry recorded.
5. Rank: B explains both observations with fewest extra assumptions. Confidence moderate given no direct seal inspection.

**Answer:** A leak at the door seal drained the water, drying the soil and wetting the floor by the door.

## What good training data looks like

Present an initial state and a surprising final observation, with a single planted ground-truth cause. Require the model to propose the most plausible explanation and explicitly rule out alternatives. Keep the context tight so the space of explanations is bounded. Include the confidence field, since abduction is inherently uncertain.

## Verification method

Answer-match by ranking against the planted cause (process check as support). Because more than one explanation can be defensible, verification ranks the model's top explanation against the planted ground-truth cause rather than demanding an exact string. The trace must also show that stronger alternatives were considered and dismissed for stated reasons.

## Difficulty ladder

1. One obvious cause, one weak alternative.
2. Two plausible causes; observations favor one clearly.
3. Three candidates; a mediator or partial explanation must be dismissed.
4. Competing causes with overlapping evidence; the discriminating detail is subtle.
5. Multiple partial causes where the best answer combines evidence and confidence is genuinely mid-range.

## Traps

- **Genuinely multiple valid answers.** Ambiguous setups have no single best explanation. Counter by keeping contexts tight and preferring ranking over exact match.
- **Context overload.** Too much detail invents spurious explanations. Counter by bounding the context to the facts that discriminate.
- **Overconfidence.** The model asserts certainty on an uncertain inference. Counter by requiring a calibrated confidence value and rewarding calibration.

## Sample record

```json
{
  "id": "abd-000078",
  "reasoning_type": "abductive",
  "domain": "mechanical / systems troubleshooting",
  "problem": "A sealed greenhouse was watered and closed on Friday. On Monday the soil is dry and the floor near the door is damp, with no entry recorded over the weekend. What most plausibly happened?",
  "reasoning_trace": [
    {"step": 1, "text": "Observations: dry soil, damp floor localized at the door, sealed room, no entry.", "label": "valid"},
    {"step": 2, "text": "Alternative: plain evaporation. Explains dry soil but not the localized damp floor.", "label": "valid"},
    {"step": 3, "text": "Alternative: a spill from entry. Ruled out, no entry was recorded.", "label": "valid"},
    {"step": 4, "text": "Best explanation: a leak at the door seal drained water out, drying soil and wetting the floor by the door.", "label": "valid"},
    {"step": 5, "text": "Confidence moderate: fits both observations with fewest assumptions, but the seal was not directly inspected.", "label": "valid"}
  ],
  "final_answer": "A leak at the door seal let the water drain out, which dried the soil and dampened the floor near the door.",
  "is_correct": true,
  "confidence": 0.7,
  "difficulty": 3,
  "generation_method": "multi_agent",
  "verification": {"method": "answer_match", "passed": true, "details": "Planted cause was a door-seal leak; model's top-ranked explanation matches and dismisses evaporation and spill for stated reasons."},
  "provenance": {"source": "generator-solver-critic", "seed_id": "abd-seed-011", "created": "2026-01-22"},
  "notes": "context bounded to three candidate causes to keep the best explanation unambiguous"
}
```
