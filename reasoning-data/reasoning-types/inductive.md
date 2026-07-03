# Inductive Reasoning

## Definition

Inductive reasoning infers a general rule from specific examples, then applies it to new cases. Unlike deduction, the conclusion is not guaranteed; it is the best generalization the examples support. Good inductive data forces the model to find the rule that survives held-out tests, not merely one that fits the examples shown.

## Canonical example

**Problem:** Given the input-output pairs 2->5, 3->7, 5->11, state the rule and apply it to 8.

**Trace:**
1. Differences: 5, 7, 11 for inputs 2, 3, 5. Try output = 2*input + 1.
2. Check: 2*2+1=5, 2*3+1=7, 2*5+1=11. All hold.
3. Rule as code: `f = lambda x: 2*x + 1`.
4. Apply to 8: 2*8+1 = 17.
5. Check against a held-out pair 4->9: 2*4+1=9. Consistent.

**Answer:** Rule: f(x) = 2x + 1. f(8) = 17.

## What good training data looks like

Sample a hidden rule, emit several consistent input-output demonstrations, and hide the rule. Ask the model for both the rule (ideally as executable code) and its application to a new query input. Reserve held-out instances the model never sees, chosen so that plausible alternative rules disagree on them. The demonstrations should be enough to determine the rule but not so many that the answer is trivial.

## Verification method

Code execution (strongest applicable). Execute the model's inferred rule against the held-out instances. The rule passes only if it produces the correct output on every held-out case, and the applied answer matches the hidden rule's output on the query. This catches rules that fit the shown examples but generalize wrong.

## Difficulty ladder

1. Linear or single-operation rule, few variables, no distractors.
2. Two-operation rule; held-out cases confirm it.
3. Rule with a conditional branch; a naive alternative fits the demos but fails held-out cases.
4. Multiple plausible rules fit the demos; only carefully chosen held-out cases disambiguate.
5. Rule over structured inputs (sequences, strings) with a hidden variable and near-miss alternatives.

## Traps

- **Multiple rules fit the same data.** Underdetermined demos let a wrong rule score perfectly. Counter by designing held-out cases where the intended rule and its plausible competitors diverge.
- **Overfitting to the shown examples.** The model memorizes pairs instead of generalizing. Counter by requiring an executable rule and testing it on unseen inputs.
- **Surface pattern in the demo ordering.** The model exploits presentation order. Counter by shuffling demonstrations and via Critic scrubbing.

## Sample record

```json
{
  "id": "ind-000112",
  "reasoning_type": "inductive",
  "domain": "mathematics",
  "problem": "From these pairs, state the rule and apply it to 8: 2->5, 3->7, 5->11.",
  "reasoning_trace": [
    {"step": 1, "text": "Guess a linear rule: output = 2*input + 1.", "label": "valid"},
    {"step": 2, "text": "Check all shown pairs: 2*2+1=5, 2*3+1=7, 2*5+1=11. All hold.", "label": "valid"},
    {"step": 3, "text": "Express as code: f = lambda x: 2*x + 1.", "label": "valid"},
    {"step": 4, "text": "Apply to 8: 2*8+1 = 17.", "label": "valid"},
    {"step": 5, "text": "Held-out check on 4->9: 2*4+1 = 9. Consistent.", "label": "valid"}
  ],
  "final_answer": "Rule: f(x) = 2x + 1. f(8) = 17.",
  "is_correct": true,
  "difficulty": 2,
  "generation_method": "procedural",
  "verification": {"method": "code_execution", "passed": true, "details": "Executed f(x)=2x+1 against held-out inputs {4,7,10}; outputs matched the hidden rule, and f(8)=17 matches."},
  "provenance": {"source": "procedural-gen", "seed_id": null, "created": "2026-01-21"},
  "notes": "held-out set chosen to exclude f(x)=x+3, which fits none of the shown pairs"
}
```
