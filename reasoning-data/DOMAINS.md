# Domains

Problems are drawn from a spread of subject-matter domains so datasets stay diverse and no reasoning type is tied to one vocabulary. Domains are grouped by verifiability, which sets the ceiling on how strong a verifier the domain can support.

Because surface-cue scrubbing (RULES.md, Rule 5) requires that the same structural problem be expressible in more than one domain, domains here are also a menu of skins for a shared underlying structure. A single deductive template might appear as a logic puzzle, a program-behavior question, and a troubleshooting scenario.

## Formal domains

Highest verifiability. These are the backbone of the corpus because their answers can be checked by a symbolic solver or by running code.

- **Logic puzzles.** Constraint-satisfaction, ordering, and grouping problems stated as explicit rules. Contributes clean, decontaminable deductive material with tunable depth. Best exercises deductive and counterfactual reasoning. Verification ceiling: symbolic solver.
- **Mathematics.** Arithmetic, algebra, number theory, combinatorics, and probability posed as generative problems with exact answers. Contributes precise, scalable material and the numeric backbone for probabilistic work. Best exercises deductive, probabilistic, and inductive reasoning. Verification ceiling: symbolic solver or code execution.
- **Program behavior.** Predict the output, final state, or invariant of a short, self-contained program. Contributes traces whose ground truth is obtained by execution. Best exercises deductive, causal, and counterfactual reasoning. Verification ceiling: code execution.

## Structured real-world domains

Medium verifiability, high transfer value. Ground truth exists but is model- or graph-derived rather than purely formal, so verification often tops out at answer-matching or process checks.

- **Medicine-style diagnosis.** Reason from findings to the most plausible underlying cause, ruling out alternatives. Contributes realistic abductive material. Best exercises abductive and probabilistic reasoning. Verification ceiling: answer-match against a planted cause, or process check.
- **Mechanical / systems troubleshooting.** Localize a fault in a described mechanical or software system from symptoms. Contributes causal and abductive material with a checkable fault location. Best exercises causal, abductive, and deductive reasoning. Verification ceiling: answer-match against the planted fault.
- **Incident and root-cause analysis.** From a timeline of events and signals, identify the root cause and the chain to the failure. Contributes causal-chain material with confounders. Best exercises causal and abductive reasoning. Verification ceiling: answer-match against a known causal graph or process check.
- **Finance and business operations.** Quantitative and policy-driven reasoning over budgets, pricing, and operational rules. Contributes deductive and probabilistic material grounded in computation. Best exercises deductive, probabilistic, and counterfactual reasoning. Verification ceiling: code execution for the quantitative parts, process check otherwise.
- **Science.** Reason from a stated model or data to a prediction or explanation, including simple experiment design. Contributes causal and inductive material with intervention reasoning. Best exercises causal, inductive, and counterfactual reasoning. Verification ceiling: answer-match against a known model, or process check.

## Open domains

Low verifiability. No formal ground truth, so these are rubric-judged only and never enter verifiable-reward training.

- **Everyday planning.** Multi-step plans under practical constraints (time, resources, ordering). Contributes decompositional and counterfactual behavior in natural settings. Best exercises counterfactual and (cross-cutting) decompositional reasoning. Verification ceiling: rubric judge.
- **Social situations.** Reason about intentions, obligations, and outcomes among people. Contributes analogical and abductive material in a human setting. Best exercises analogical and abductive reasoning. Verification ceiling: rubric judge.
- **Ethics.** Dilemmas with multiple defensible positions across ethical frameworks. Contributes the moral-ethical corpus. Best exercises moral-ethical reasoning. Verification ceiling: rubric judge, scoring reasoning quality rather than a fixed conclusion.
