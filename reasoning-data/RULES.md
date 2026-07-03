# Rules

These are the non-negotiable quality standards. They apply to every example in every dataset, regardless of reasoning type. An example that violates any rule does not enter the curated set.

## 1. Decontamination

Every batch is audited against all evaluation sets before inclusion, and re-audited after any synthetic generation step that could have reintroduced eval material. Contamination is the most common cause of illusory gains: a model that has seen the test looks smart and is not. Decontamination is run at the batch level and repeated whenever new items are synthesized from existing ones.

## 2. Verification required

No example enters the curated set without passing its type's verification method (see each reasoning-type file and the ranking in README.md). The verification method used and its result are recorded on every example, in the `verification` object. An unverified example is raw data, not curated data.

## 3. Verify the path, not just the destination

Where possible, check intermediate steps, not only the final answer. A correct final answer reached through invalid reasoning is a reject. Traces can look coherent while being post-hoc rationalization, so a plausible-sounding path that does not actually entail the answer is treated as a failure even when the answer happens to be right.

## 4. Generative answers only, never multiple choice

The model must produce the exact final string, value, or structure. Multiple-choice formats are prohibited because they let models score by eliminating distractors from surface patterns without doing the reasoning. Every problem is posed so that the answer must be constructed, not selected.

## 5. Surface-cue scrubbing

Problems must not be solvable through word co-occurrence or vocabulary association. During curation, mutate surface vocabulary — swap domains, rename entities, use abstract terms — while preserving the underlying structure, so that only the reasoning solves the problem. If a problem can be answered by pattern-matching the wording, it is rewritten or discarded.

## 6. Difficulty tagging

Every example carries a 1-5 difficulty tag. This enables curriculum ordering during training and honest evaluation by difficulty band. The meaning of each level is defined concretely per type in the type's difficulty ladder.

## 7. Diversity and dedup

Aggressively remove near-duplicates, both in problems and in reasoning style. Style monoculture degrades training: a corpus where every trace sounds the same teaches the model one voice and one move-set. Deduplication targets structural and stylistic repetition, not just verbatim overlap.

## 8. Concise completeness

Prefer short, complete reasoning over verbose reasoning. Every step must earn its place. Verbosity inflates training cost and enlarges the error surface, and padding is not rigor. A trace is complete when removing any step would break the argument, and no longer.

## 9. Keep negatives

Incorrect traces are retained and labeled, not discarded. They are contrastive training signal for preference optimization and the raw material for metacognitive data. A negative is stored with `is_correct` false and, where relevant, a pointer to its paired positive.

## 10. Self-check inside the trace

Where natural, traces end with a brief verification step: does the answer satisfy the stated constraints, do the assumptions hold, is the arithmetic consistent. This is expressed as ordinary trace steps, not as separate schema fields, so the model learns self-checking as part of reasoning rather than as external scaffolding.

## 11. Human spot-check

A small fixed percentage of every automated batch gets human review. The human acceptance rate on that sample is a standing quality metric: if it drops, the pipeline is producing worse data than its automated gates report, and generation is paused until the cause is found.

## 12. Known failure modes to defend against

- **Unfaithful chain-of-thought:** the stated reasoning does not cause the answer. The pipeline checks intermediate steps (Rule 3) and favors executable or symbolically verifiable traces so the path is checkable, not just plausible.
- **Reward hacking on narrow verifiers:** the model games a weak checker. The pipeline uses the strongest verifier the type allows and rotates fresh procedural instances so a memorized shortcut stops paying off.
- **Expert blind spot in human-written traces:** experts skip steps they find obvious. The pipeline uses a structured interviewer during human elicitation to force out omitted steps.
- **Surface-pattern shortcuts:** problems solvable by vocabulary association. The pipeline scrubs surface cues (Rule 5) via Critic-driven mutation.
- **Benchmark leakage:** eval material bleeds into training. The pipeline decontaminates every batch and re-audits after synthesis (Rule 1).

## 13. Standard record schema

All data conforms to SCHEMA.md. There are no type-specific field variations. If an example cannot be expressed in the schema, the schema gets amended deliberately, through a change to SCHEMA.md, rather than worked around with ad-hoc fields. One schema, enforced everywhere, is what makes the corpus comparable across types.
