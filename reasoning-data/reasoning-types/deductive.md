# Deductive Reasoning

## Definition

Deductive reasoning derives conclusions that follow with certainty from stated premises. If the premises are true and the inference rules are valid, the conclusion cannot be false. This is the backbone of the corpus.

## Canonical example

**Problem:** All members of the Blue team wear a badge. Anyone wearing a badge has cleared security. Rana is a member of the Blue team. Has Rana cleared security?

**Trace:**
1. Every Blue team member wears a badge (premise).
2. Every badge-wearer has cleared security (premise).
3. Rana is a Blue team member, so Rana wears a badge (modus ponens on step 1).
4. Rana wears a badge, so Rana has cleared security (modus ponens on step 2).
5. Check: only the premises and valid inference were used, no extra assumptions.

**Answer:** Yes, Rana has cleared security.

## What good training data looks like

Procedurally generate from a formal rule base: if/then chains, disjunctions, and exclusions. Control difficulty by the number of inference hops between premises and conclusion. Inject rule-breaking distractors such as premises that invite affirming the consequent or denying the antecedent, so the correct path requires valid inference rather than pattern completion. Generate fresh instances constantly so no specific problem can be memorized.

## Verification method

Symbolic solver (strongest available). Encode the premises and the query in a formal logic (propositional or first-order), and have the solver confirm whether the conclusion is entailed. The trace is accepted only if each step is a valid inference and the final answer matches the solver's verdict.

## Difficulty ladder

1. One inference hop, no distractors.
2. Two to three hops, no distractors.
3. Several hops with irrelevant true premises as noise.
4. Distractors that invite named fallacies (affirming the consequent, denying the antecedent); disjunctions and exclusions.
5. Long chains with nested conditionals, multiple distractors, and a conclusion that requires case analysis.

## Traps

- **Invalid shortcuts that land on the correct answer.** A trace can reach the right conclusion through an invalid step. Counter with fresh procedural instances and step-level checking (label each step), so a correct answer with an invalid step is a reject.
- **Affirming the consequent / denying the antecedent.** Distractor premises make these tempting. Counter by encoding the query symbolically and rejecting any conclusion the solver does not entail.
- **Distractor blindness.** The model uses an irrelevant premise. Counter by verifying that every step in an accepted trace is actually used in the entailment.

## Sample record

```json
{
  "id": "ded-000045",
  "reasoning_type": "deductive",
  "domain": "logic puzzles",
  "problem": "If a package is fragile it is shipped in a padded box. Every padded box is inspected before dispatch. Package 7 is fragile. Was package 7 inspected before dispatch?",
  "reasoning_trace": [
    {"step": 1, "text": "Fragile packages ship in padded boxes (premise).", "label": "valid"},
    {"step": 2, "text": "Every padded box is inspected before dispatch (premise).", "label": "valid"},
    {"step": 3, "text": "Package 7 is fragile, so it ships in a padded box (modus ponens on step 1).", "label": "valid"},
    {"step": 4, "text": "It ships in a padded box, so it was inspected before dispatch (modus ponens on step 2).", "label": "valid"},
    {"step": 5, "text": "Check: conclusion follows from the premises by valid inference only.", "label": "valid"}
  ],
  "final_answer": "Yes, package 7 was inspected before dispatch.",
  "is_correct": true,
  "difficulty": 2,
  "generation_method": "procedural",
  "verification": {"method": "symbolic_solver", "passed": true, "details": "Premises and query encoded in first-order form; solver confirms entailment via two modus ponens steps."},
  "provenance": {"source": "procedural-gen", "seed_id": "ded-seed-003", "created": "2026-01-20"},
  "notes": "paired negative: ded-000045-neg (affirming the consequent)"
}
```
