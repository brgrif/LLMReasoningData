# Metacognitive Reasoning

## Definition

Metacognitive reasoning is reasoning about reasoning: inspecting a chain of thought, locating where it goes wrong, explaining why, and correcting it. The task operates on a trace rather than on a raw problem. Good data teaches the model to audit reasoning, including recognizing when reasoning is in fact sound.

## Canonical example

**Problem:** A student reasons: "A number divisible by 4 is even. 12 is even. Therefore 12 is divisible by 4." Find any errors, explain them, and give the corrected reasoning.

**Trace:**
1. Claim 1: divisible by 4 implies even. Valid.
2. Claim 2: 12 is even. True.
3. Inference: from "even" concludes "divisible by 4." This affirms the consequent, an invalid step.
4. Why it fails: even numbers need not be divisible by 4 (e.g. 6, 10).
5. Correction: 12 happens to be divisible by 4 (12 = 4*3), but not because it is even; the original inference is invalid regardless of the true answer.

**Answer:** Error at step 3: affirming the consequent. Being even does not entail divisibility by 4; the conclusion is not justified by the given reasoning.

## What good training data looks like

Present a problem together with a reasoning trace that contains one or more deliberately planted errors: a false assumption, an invalid step, an arithmetic slip, or a skipped case. The task is to locate each error, explain why it fails, and produce the corrected reasoning. Crucially, include clean traces with no errors, so the model cannot assume every trace is flawed. Source flawed traces from the negatives pool where possible, since those are real failures rather than synthetic ones.

## Verification method

Answer-match against the known planted errors (process check). Because the errors are planted, their locations and types are known ground truth. Verify that the model flags exactly the planted errors, at the right steps, with correct explanations, and that it flags nothing in clean traces. Precision and recall against the planted set are both scored.

## Difficulty ladder

1. One obvious invalid step in a short trace.
2. One subtle error (arithmetic slip or skipped case).
3. A clean trace that must be certified error-free.
4. Multiple errors in one trace, of different kinds.
5. A near-valid trace where the single flaw is deep and the rest is sound.

## Traps

- **Rubber-stamping.** The model misses a real flaw and approves it. Counter with a balanced clean/flawed mix and exact planted-error matching, so misses are penalized.
- **Hallucinated flaws.** The model invents errors in valid reasoning. Counter by including clean traces and penalizing any flag not matching a planted error.
- **Right location, wrong reason.** The model finds the step but misexplains it. Counter by matching the explanation to the planted error type, not just the step number.

## Sample record

```json
{
  "id": "met-000101",
  "reasoning_type": "metacognitive",
  "domain": "mathematics",
  "problem": "Audit this reasoning and correct it: 'Any number divisible by 4 is even. 12 is even. Therefore 12 is divisible by 4.'",
  "reasoning_trace": [
    {"step": 1, "text": "'Divisible by 4 implies even' is a valid claim.", "label": "valid"},
    {"step": 2, "text": "'12 is even' is true.", "label": "valid"},
    {"step": 3, "text": "The trace concludes 'divisible by 4' from 'even'; this affirms the consequent and is invalid.", "label": "valid"},
    {"step": 4, "text": "Counterexamples: 6 and 10 are even but not divisible by 4, so 'even' does not entail 'divisible by 4'.", "label": "valid"},
    {"step": 5, "text": "Correction: 12 is divisible by 4 (12=4*3), but that does not follow from being even; the audited inference is invalid.", "label": "valid"}
  ],
  "final_answer": "Error at the third step: affirming the consequent. Being even does not entail divisibility by 4, so the conclusion is unjustified by the given reasoning.",
  "is_correct": true,
  "difficulty": 2,
  "generation_method": "multi_agent",
  "verification": {"method": "process_check", "passed": true, "details": "Planted error was affirming the consequent at step 3; model flags exactly that step with the correct fallacy and no spurious flags."},
  "provenance": {"source": "negatives-pool-derived", "seed_id": "ded-000045-neg", "created": "2026-01-27"},
  "notes": "flawed trace sourced from the deductive negatives pool"
}
```
