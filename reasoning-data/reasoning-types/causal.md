# Causal Reasoning

## Definition

Causal reasoning identifies what causes what, distinguishing genuine causation from mere correlation. It reasons about interventions ("if we changed X, would Y change?") and about confounders and mediators, not just associations in observed data. The defining move is intervention, not correlation.

## Canonical example

**Problem:** In a town, ice cream sales and swimming-pool injuries both rise together. A report claims ice cream causes the injuries. Given that both rise with daily temperature, is the claim supported, and what experiment would settle it?

**Trace:**
1. Observed: ice cream sales and pool injuries correlate.
2. Temperature rises with both, so temperature is a common cause (confounder).
3. Correlation between ice cream and injuries can be fully explained by the shared cause; no direct arrow is implied.
4. Intervention test: hold temperature fixed (or vary ice cream availability independently) and see whether injuries change.
5. Check: if injuries do not change when ice cream is varied at fixed temperature, the causal claim is refuted.

**Answer:** The claim is not supported; temperature is a confounder. Vary ice cream availability while holding temperature constant to test for a direct effect.

## What good training data looks like

Generate from a known causal graph with both observational and interventional data, including confounders and mediators. The graph is the ground truth. Pose questions that require distinguishing a direct effect from a confounded association, and that ask what intervention would discriminate between hypotheses. Vary graph topology so the model cannot default to a fixed answer shape.

## Verification method

Answer-match against the known causal graph (process check as support). Because the generating graph is known, the correct causal claim, the confounder/mediator roles, and the discriminating intervention are all determined. Verify the model's answer against the graph, and check that the trace reasons about intervention rather than association alone.

## Difficulty ladder

1. Two variables, direct cause, no confounder.
2. Chain of three variables with a mediator.
3. One confounder producing a spurious correlation.
4. Confounder plus mediator; the model must separate direct and indirect effects.
5. Multiple confounders and a required intervention design to identify the effect.

## Traps

- **Correlation-as-causation.** The model reads an association as a cause. Counter by requiring intervention reasoning ("what experiment would distinguish these hypotheses"), not just co-occurrence.
- **Ignoring confounders.** A common cause is overlooked. Counter by generating from graphs that include confounders and verifying their roles against the graph.
- **Mediator/confounder confusion.** The model mislabels a variable on the path. Counter by checking the assigned role against the ground-truth graph.

## Sample record

```json
{
  "id": "cau-000056",
  "reasoning_type": "causal",
  "domain": "incident and root-cause analysis",
  "problem": "Reports of slow page loads and high server CPU both rise during the same hours. An engineer claims slow pages cause the high CPU. Given that both rise with incoming traffic volume, is the claim supported, and what would settle it?",
  "reasoning_trace": [
    {"step": 1, "text": "Observed: slow loads and high CPU correlate over time.", "label": "valid"},
    {"step": 2, "text": "Traffic volume rises with both, making it a common cause (confounder).", "label": "valid"},
    {"step": 3, "text": "The correlation can be explained entirely by traffic; no direct slow-pages -> CPU arrow is implied.", "label": "valid"},
    {"step": 4, "text": "Intervention: hold traffic fixed (load test at constant request rate) and induce slow pages to see if CPU changes.", "label": "valid"},
    {"step": 5, "text": "Check: if CPU is unchanged when slowness is induced at fixed traffic, the causal claim is refuted.", "label": "valid"}
  ],
  "final_answer": "Not supported; traffic volume is a confounder. Hold request rate constant and induce slow pages to test for a direct effect on CPU.",
  "is_correct": true,
  "difficulty": 3,
  "generation_method": "procedural",
  "verification": {"method": "answer_match", "passed": true, "details": "Generating graph: traffic -> slow_loads, traffic -> CPU, no edge slow_loads -> CPU. Model correctly identifies the confounder and the discriminating intervention."},
  "provenance": {"source": "procedural-gen", "seed_id": "cau-seed-004", "created": "2026-01-24"},
  "notes": "paired negative asserts direct causation ignoring the traffic confounder"
}
```
