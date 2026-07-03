# Counterfactual Reasoning

## Definition

Counterfactual reasoning answers "what would have happened if things had been different," holding the rest of the world fixed and propagating the change. It requires a model of how the world works, since the counterfactual outcome is never directly observed. The reasoning is only valid if the intervention actually changes the outcome from the factual baseline.

## Canonical example

**Problem:** A conveyor moves a box 2 cm per tick. It started at position 0 and ran for 5 ticks, ending at 10 cm. If the speed had been 3 cm per tick instead, where would the box have ended after the same 5 ticks?

**Trace:**
1. World model: position = speed * ticks, starting at 0.
2. Factual baseline: 2 * 5 = 10 cm (given).
3. Counterfactual intervention: set speed to 3 cm per tick, hold ticks at 5.
4. Run the model: 3 * 5 = 15 cm.
5. Check: the intervention changed the outcome (15 != 10), so the counterfactual is not answerable from the baseline alone.

**Answer:** 15 cm.

## What good training data looks like

Define a simple world model or simulator, state the factual scenario, then pose a "what if" question that intervenes on one variable. Verify by running the counterfactual through the same model. Choose interventions that move the outcome away from the factual baseline, so a model that just echoes the baseline is wrong. Vary the intervention target so the model must locate what changed.

## Verification method

Code execution against the world model (strongest applicable). Implement the world model, apply the intervention, and run it to get the true counterfactual outcome. The trace passes if its answer matches the simulator's counterfactual result and its steps hold the non-intervened variables fixed.

## Difficulty ladder

1. Single-variable linear model, one intervention.
2. Two variables, one intervention, outcome clearly differs from baseline.
3. Intervention with a downstream effect that must be propagated.
4. Multiple interacting variables; only one is intervened on, others held fixed.
5. Intervention plus a conditional in the world model, so the effect depends on state.

## Traps

- **Answering from the factual baseline.** The model reports what did happen, not what would have. Counter by varying the intervention so the baseline alone cannot answer, making baseline-echoing verifiably wrong.
- **Failing to hold other variables fixed.** The model changes more than the intervention. Counter by running the exact intervention in the simulator and comparing.
- **Not propagating downstream effects.** The model stops at the intervened variable. Counter with world models that have a dependency chain and checking the final state.

## Sample record

```json
{
  "id": "cfa-000067",
  "reasoning_type": "counterfactual",
  "domain": "program behavior",
  "problem": "A counter starts at 0 and increments by 2 on each of 5 ticks, ending at 10. If it had incremented by 3 per tick over the same 5 ticks, what would the final value have been?",
  "reasoning_trace": [
    {"step": 1, "text": "World model: value = increment * ticks, starting at 0.", "label": "valid"},
    {"step": 2, "text": "Factual baseline: 2 * 5 = 10 (given).", "label": "valid"},
    {"step": 3, "text": "Intervene: set increment to 3, hold ticks at 5.", "label": "valid"},
    {"step": 4, "text": "Run: 3 * 5 = 15.", "label": "valid"},
    {"step": 5, "text": "Check: outcome (15) differs from baseline (10), so the intervention genuinely matters.", "label": "valid"}
  ],
  "final_answer": "15",
  "is_correct": true,
  "difficulty": 2,
  "generation_method": "procedural",
  "verification": {"method": "code_execution", "passed": true, "details": "Simulated value = increment*ticks with increment=3, ticks=5; result 15 matches the trace's answer."},
  "provenance": {"source": "procedural-gen", "seed_id": null, "created": "2026-01-25"},
  "notes": "intervention chosen so baseline (10) cannot answer the counterfactual"
}
```
