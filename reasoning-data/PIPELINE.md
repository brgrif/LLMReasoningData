# Pipeline

How data is generated and filtered at scale, without human annotation of every example. The core is a **Propose-Verify-Mutate** pipeline with three asymmetric model roles, followed by deterministic gates.

## The three roles

1. **Generator agent.** Given a structural template for a reasoning type and a difficulty level (e.g. deductive at depth 4), the Generator writes the natural-language problem, the step-by-step trace, and the final answer. It works from structure outward, so difficulty is controlled by the template, not guessed.

2. **Solver agent.** A separate model attempts the problem blind: no hints, no access to the Generator's trace, only the problem statement. If the Solver cannot solve it, or the problem admits more than one reading, the item is discarded or returned to the Generator for revision. This step guarantees a clean, unambiguous solution path and filters out problems that are underspecified or accidentally impossible.

3. **Critic agent.** The Critic mutates surviving items to remove surface heuristics: it renames entities, swaps domains, and abstracts vocabulary, then confirms the problem still has the same answer. It also checks that every trace step is valid and actually needed, deleting steps that carry no weight. Items that survive mutation with their structure intact go forward; items that become solvable by wording alone, or whose trace falls apart under scrutiny, are dropped.

## Deterministic gates

Surviving items pass through fixed, automated gates in this order:

1. **Formal verification** per the type's method (symbolic solver, code execution, answer-match, process check, or rubric judge). This is the hard gate: no pass, no curated positive.
2. **Deduplication** against the existing corpus, on both problem content and reasoning style.
3. **Difficulty tagging** to the 1-5 ladder for the type.
4. **Diversity check** to prevent style and structure monoculture within the batch.
5. **Decontamination audit** against all evaluation sets, repeated because synthesis can reintroduce eval material.
6. **Human spot-check** of a fixed percentage of the batch, whose acceptance rate is a standing quality metric.

Failed items are not deleted. Items that fail with an informative error (a wrong answer, a specific invalid step) go to the **negatives pool** with `is_correct` false. Those negatives are contrastive signal for preference optimization and the source material for metacognitive data.

## Supplementary sources

- **Seed problems.** A small set of hand-written, high-quality problems used to bootstrap the Generator. They anchor style and difficulty before synthesis scales up, and they are tracked in provenance so their influence is auditable. Seeds are never drawn from eval splits.
- **Human expert elicitation.** For domains where model teachers are weak, experts supply reasoning directly. A structured interviewer drives the session, forcing out the intermediate steps experts skip by habit (the expert blind spot from RULES.md, Rule 12). The result is human_expert-sourced traces that still conform to the schema.
