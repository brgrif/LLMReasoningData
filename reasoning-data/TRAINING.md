# Training

How the data is consumed. Four stages, in order.

## 1. SFT / distillation first

Fine-tune on curated positive traces. Weight the loss toward the intermediate reasoning steps rather than only the final answer, so the model learns the calculation, not just the destination. A model trained to reproduce answers alone learns to guess the endpoint; a model trained on the weighted path learns the procedure that produces it. This stage expands capability and is the foundation the later stages sharpen.

## 2. Mixing rule against model collapse

Never fine-tune on synthetic reasoning data alone. Retain roughly 10-20% general high-quality natural text in every training mix. Without it, the model over-indexes on the generator's style and loses linguistic range: its outputs converge on the narrow voice and move-set of the synthesis pipeline. The natural-text fraction is a guardrail, not filler, and it stays in the mix at every stage.

## 3. RLVR second, for sharpening

Reinforcement learning with verifiable rewards runs after SFT, and only on verifiable types: deductive, probabilistic, causal, counterfactual, and inductive with held-out checks. Metacognitive, abductive, and analogical join RLVR only where a deterministic check exists for the specific items being trained. Moral-ethical is permanently excluded from verifiable-reward training, because scoring its conclusions would train agreement rather than reasoning. RLVR does not expand what the model can do; it tightens accuracy where correctness is checkable.

## 4. Preference optimization from negatives

Pair each correct trajectory with a plausible flawed one drawn from the negatives pool, and train the preference boundary to penalize the specific fallacy, not merely the wrong answer. The negative should differ from the positive by the reasoning error, so the gradient targets the fallacy (affirming the consequent, base-rate neglect, correlation-as-causation) rather than surface differences. For deductive, inductive, and probabilistic types, favor hybrid text-plus-code targets, so executable code anchors the logic and the preference signal is checkable.
