# Probabilistic Reasoning

## Definition

Probabilistic reasoning updates beliefs under uncertainty using the rules of probability, combining prior information with evidence to produce a posterior. It is quantitative: answers are numbers, and belief updates can be computed exactly. This is the highest RL-readiness of any type.

## Canonical example

**Problem:** A test for a condition is 90% sensitive and 90% specific. The condition affects 1% of a population. A random person tests positive. What is the probability they have the condition?

**Trace:**
1. Prior P(D) = 0.01, so P(not D) = 0.99.
2. Sensitivity P(+|D) = 0.9; false positive rate P(+|not D) = 0.1.
3. P(+) = 0.9*0.01 + 0.1*0.99 = 0.009 + 0.099 = 0.108.
4. Posterior P(D|+) = 0.009 / 0.108 = 0.0833.
5. Check: despite a positive test, the low base rate keeps the posterior small (~8.3%).

**Answer:** About 0.083 (8.3%).

## What good training data looks like

Numeric problems with computable posteriors and, where useful, multi-round belief updates (each new piece of evidence updates the current posterior). Verify by code execution. Include the confidence field to record the calibrated belief the trace arrives at. Vary priors and likelihoods so the answer cannot be guessed from problem shape, and include cases where the base rate dominates.

## Verification method

Code execution (strongest applicable). Compute the exact posterior in code from the stated priors and likelihoods, and compare against the trace's final number within a small tolerance. Multi-round problems are checked by executing each update in sequence. Executed computation is the ground truth.

## Difficulty ladder

1. Single Bayes update, round numbers.
2. Single update where the base rate dominates the intuition (low prior, positive test).
3. Two-round sequential update.
4. Update with more than two hypotheses or a non-uniform prior.
5. Multi-round update with dependent evidence and a mix of continuous and discrete factors.

## Traps

- **Base-rate neglect.** The model ignores the prior and reports the sensitivity. Counter with executed computation, which produces the correct posterior and flags the mismatch.
- **Arithmetic slips.** Hand computation drifts. Counter by executing the calculation rather than trusting the written steps.
- **Confusing P(+|D) with P(D|+).** The conditional is inverted. Counter by checking the executed posterior against the trace's answer.

## Sample record

```json
{
  "id": "prb-000089",
  "reasoning_type": "probabilistic",
  "domain": "medicine-style diagnosis",
  "problem": "A screening test is 90% sensitive and 90% specific. The condition has a 1% prevalence. A randomly chosen person tests positive. What is the probability they have the condition?",
  "reasoning_trace": [
    {"step": 1, "text": "Prior P(D)=0.01, P(not D)=0.99.", "label": "valid"},
    {"step": 2, "text": "P(+|D)=0.9, P(+|not D)=0.1.", "label": "valid"},
    {"step": 3, "text": "P(+) = 0.9*0.01 + 0.1*0.99 = 0.108.", "label": "valid"},
    {"step": 4, "text": "Posterior P(D|+) = 0.009 / 0.108 = 0.0833.", "label": "valid"},
    {"step": 5, "text": "Check: low base rate keeps the posterior near 8.3% despite the positive test.", "label": "valid"}
  ],
  "final_answer": "0.083",
  "is_correct": true,
  "confidence": 0.95,
  "difficulty": 2,
  "generation_method": "procedural",
  "verification": {"method": "code_execution", "passed": true, "details": "Computed (0.9*0.01)/(0.9*0.01 + 0.1*0.99) = 0.08333; matches 0.083 within tolerance."},
  "provenance": {"source": "procedural-gen", "seed_id": "prb-seed-002", "created": "2026-01-26"},
  "notes": "base-rate-dominant case; paired negative reports 0.9 (base-rate neglect)"
}
```
