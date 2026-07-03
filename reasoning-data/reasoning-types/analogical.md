# Analogical Reasoning

## Definition

Analogical reasoning transfers a relational structure from a source situation to a target, mapping roles rather than surface features. The reasoning succeeds when the deep relations line up, even if the surface content differs entirely. Good data forces mapping of structure, not matching of vocabulary.

## Canonical example

**Problem:** A heart is to circulation as a pump is to a water system, and a battery is to a circuit as ____ is to a river's flow. Fill the blank.

**Trace:**
1. Extract the relational template: X provides the driving force that moves a medium through a system.
2. Heart drives blood through circulation; pump drives water through a water system; battery drives current through a circuit.
3. Target system: a river's flow, medium is water moving downhill.
4. The driving force for a river's flow is gravity (via elevation drop), so the source of force is the height difference / gravity.
5. Check: gravity drives water through the river as the heart drives blood; the relation matches, surface differs.

**Answer:** Gravity (the elevation drop).

## What good training data looks like

Build source-target pairs that share a deep relational structure but differ on surface features. A strong format stacks several domains demonstrating the same relational template and asks for the missing element in a new domain. Deliberately include items where surface similarity points to the wrong answer, so only structural mapping succeeds. Let the Critic scrub co-occurring vocabulary that could leak the answer.

## Verification method

Process check on the structural mapping (strongest applicable here). Verify that the model's answer fills the target role defined by the shared relational template, and that the trace states the mapping explicitly. Where a formal relational schema is available, check the mapping against it; otherwise the process check confirms the role, function, and relation align across source and target.

## Difficulty ladder

1. Single source-target pair, surface and structure agree.
2. One pair, surface and structure agree, more abstract relation.
3. Stacked domains sharing one template; straightforward transfer to a new domain.
4. An item where surface similarity suggests a distractor that violates the structure.
5. Multiple relational templates present; the correct mapping requires isolating the intended relation from competing ones.

## Traps

- **Surface-similarity matching.** The model answers by shared vocabulary or appearance, not relation. Counter by including items where surface and structure disagree, so a surface match is wrong.
- **Leaked vocabulary.** Co-occurring words point at the answer. Counter with Critic scrubbing of shared terms.
- **Wrong relation selected.** With several relations present, the model maps the wrong one. Counter by requiring the trace to name the relational template it is transferring.

## Sample record

```json
{
  "id": "ana-000034",
  "reasoning_type": "analogical",
  "domain": "science",
  "problem": "A heart is to circulation as a pump is to a water system. A battery is to a circuit as ____ is to a river's flow. Fill the blank with the item that plays the same role.",
  "reasoning_trace": [
    {"step": 1, "text": "Relational template: X supplies the driving force that moves a medium through a system.", "label": "valid"},
    {"step": 2, "text": "Heart->blood, pump->water, battery->current all instantiate this template.", "label": "valid"},
    {"step": 3, "text": "Target: a river's flow; the medium is water moving downhill.", "label": "valid"},
    {"step": 4, "text": "The driving force is gravity via the elevation drop, filling the role of heart/pump/battery.", "label": "valid"},
    {"step": 5, "text": "Check: the relation (driving force for a medium) matches; surface features differ, confirming structural mapping.", "label": "valid"}
  ],
  "final_answer": "Gravity (the elevation drop).",
  "is_correct": true,
  "difficulty": 3,
  "generation_method": "multi_agent",
  "verification": {"method": "process_check", "passed": true, "details": "Answer fills the driving-force role of the shared template; mapping stated explicitly and matches the reference relational schema."},
  "provenance": {"source": "generator-solver-critic", "seed_id": null, "created": "2026-01-23"},
  "notes": "surface distractor 'water' scrubbed by Critic to prevent vocabulary leakage"
}
```
