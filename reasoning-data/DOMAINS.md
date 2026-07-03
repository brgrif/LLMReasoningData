# Domains

Problems are drawn from a spread of subject-matter domains so datasets stay diverse and no reasoning type is tied to one vocabulary. Domains are grouped by verifiability, which sets the ceiling on how strong a verifier the domain can support.

Because surface-cue scrubbing (RULES.md, Rule 5) requires that the same structural problem be expressible in more than one domain, domains here are also a menu of skins for a shared underlying structure. A single deductive template might appear as a logic puzzle, a program-behavior question, and a troubleshooting scenario. Analogical reasoning depends on this breadth most of all (see the note at the end).

## Formal domains

Highest verifiability. These are the backbone of the corpus because their answers can be checked by a symbolic solver or by running code.

- **Logic puzzles.** Constraint-satisfaction, ordering, and grouping problems stated as explicit rules. Contributes clean, decontaminable deductive material with tunable depth. Best exercises deductive and counterfactual reasoning. Verification ceiling: symbolic solver.
- **Mathematics.** Arithmetic, algebra, number theory, combinatorics, and probability posed as generative problems with exact answers. Contributes precise, scalable material and the numeric backbone for probabilistic work. Best exercises deductive, probabilistic, and inductive reasoning. Verification ceiling: symbolic solver or code execution.
- **Program behavior.** Predict the output, final state, or invariant of a short, self-contained program. Contributes traces whose ground truth is obtained by execution. Best exercises deductive, causal, and counterfactual reasoning. Verification ceiling: code execution.
- **Algorithms and program analysis.** Reason about invariants, termination, and complexity of an algorithm rather than one concrete run. Contributes deductive and inductive material whose claims can be proved or checked empirically. Best exercises deductive, inductive, and counterfactual reasoning. Verification ceiling: symbolic solver for invariants, code execution for behavior.
- **Formal grammars and symbol systems.** String rewriting, encodings, ciphers, and rule-governed symbol manipulation. Contributes clean inductive and analogical material because the rules are explicit, executable, and can be dressed in any alphabet. Best exercises inductive, deductive, and analogical reasoning. Verification ceiling: code execution.

## Structured real-world domains

Medium verifiability, high transfer value. Ground truth exists but is model- or graph-derived rather than purely formal, so verification often tops out at answer-matching or process checks.

- **Medicine-style diagnosis.** Reason from findings to the most plausible underlying cause, ruling out alternatives. Contributes realistic abductive material. Best exercises abductive and probabilistic reasoning. Verification ceiling: answer-match against a planted cause, or process check.
- **Mechanical / systems troubleshooting.** Localize a fault in a described mechanical or software system from symptoms. Contributes causal and abductive material with a checkable fault location. Best exercises causal, abductive, and deductive reasoning. Verification ceiling: answer-match against the planted fault.
- **Incident and root-cause analysis.** From a timeline of events and signals, identify the root cause and the chain to the failure. Contributes causal-chain material with confounders. Best exercises causal and abductive reasoning. Verification ceiling: answer-match against a known causal graph or process check.
- **Finance and business operations.** Quantitative and policy-driven reasoning over budgets, pricing, and operational rules. Contributes deductive and probabilistic material grounded in computation. Best exercises deductive, probabilistic, and counterfactual reasoning. Verification ceiling: code execution for the quantitative parts, process check otherwise.
- **Science.** Reason from a stated model or data to a prediction or explanation, including simple experiment design. Contributes causal and inductive material with intervention reasoning. Best exercises causal, inductive, and counterfactual reasoning. Verification ceiling: answer-match against a known model, or process check.
- **Law and regulation.** Apply explicit rules, statutes, and precedent to a fact pattern, and map new facts onto governing cases. Contributes deductive and analogical material grounded in rule application and case-to-case transfer. Best exercises deductive and analogical reasoning. Verification ceiling: answer-match against the governing rule, or process check.
- **Engineering and physical systems.** Circuits, structures, mechanisms, and control loops reasoned about from stated principles. A rich source of relational templates (flow driven by a potential difference, feedback toward a setpoint), which makes it a workhorse for analogical data. Best exercises analogical, causal, and counterfactual reasoning. Verification ceiling: code execution or answer-match against a model.
- **Biology and ecology.** Homeostasis, feedback regulation, food webs, and inheritance. Contributes causal and analogical material where the same regulatory structure recurs across scales, from a cell to an ecosystem. Best exercises causal, analogical, and abductive reasoning. Verification ceiling: answer-match against a known model, or process check.
- **Economics and markets.** Incentives, supply and demand, pricing, and equilibria. Contributes causal, counterfactual, and analogical material with a computable quantitative core and clear intervention questions. Best exercises causal, counterfactual, and analogical reasoning. Verification ceiling: code execution for the quantitative parts, process check otherwise.
- **Chemistry.** Stoichiometry, equilibria, and reaction prediction from stated rules. Contributes deductive and probabilistic material with computable answers. Best exercises deductive, causal, and probabilistic reasoning. Verification ceiling: code execution.

## Open domains

Low verifiability. No formal ground truth, so these are rubric-judged only and never enter verifiable-reward training.

- **Everyday planning.** Multi-step plans under practical constraints (time, resources, ordering). Contributes decompositional and counterfactual behavior in natural settings. Best exercises counterfactual and (cross-cutting) decompositional reasoning. Verification ceiling: rubric judge.
- **Social situations.** Reason about intentions, obligations, and outcomes among people. Contributes analogical and abductive material in a human setting. Best exercises analogical and abductive reasoning. Verification ceiling: rubric judge.
- **Ethics.** Dilemmas with multiple defensible positions across ethical frameworks. Contributes the moral-ethical corpus. Best exercises moral-ethical reasoning. Verification ceiling: rubric judge, scoring reasoning quality rather than a fixed conclusion.
- **Negotiation and interpersonal strategy.** Reason about interests, offers, and trade-offs among people with partly aligned goals. Contributes analogical and abductive material in a human setting. Best exercises analogical, abductive, and counterfactual reasoning. Verification ceiling: rubric judge. Full multi-agent strategic play with payoffs is deferred (see ROADMAP.md); this domain covers the human-reasoning surface only.
- **Narrative and discourse.** Infer motives, causes, and structure from stories and arguments. Contributes abductive and analogical material where the relational skeleton of a plot or argument transfers across settings. Best exercises abductive and analogical reasoning. Verification ceiling: rubric judge.

## A note on analogical breadth and shared templates

Analogical reasoning is the domain-hungriest type. It works by transferring a relational template from a source domain to a target, so its data quality scales directly with how many domains are on the menu: more domains means more source-target pairs and more room to hide surface similarity. A handful of relational templates recur across the domains above, and analogical problems are built by pairing domains that share one:

- **Flow driven by a potential difference:** circuits (voltage), fluid systems (pressure), rivers (elevation), diffusion (concentration), price gradients (economics).
- **Feedback toward a setpoint:** a thermostat (engineering), homeostasis (biology), inventory control (operations), monetary policy (economics).
- **Equilibrium of opposing forces:** chemical equilibria, supply and demand, predator-prey balance.
- **Hierarchy and inheritance:** class hierarchies (program behavior), biological taxonomy, legal precedent.

Because the same structure can be expressed in any of these skins, this breadth is also what powers surface-cue scrubbing (RULES.md, Rule 5): the Critic can re-skin a problem into an unrelated domain while preserving the relation, so vocabulary shortcuts stop working and only the structure carries the answer. New domains should be added when they either raise a reasoning type's verification ceiling or supply a relational template the corpus does not yet cover.
