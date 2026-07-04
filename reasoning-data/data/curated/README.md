# data/curated/

Items that passed every gate. Positives only: `is_correct` true and `verification.passed` true. Split into `{type}.train.jsonl` and `{type}.eval.jsonl`; eval problems are never used as generation seeds and never appear in a train file. This is the only set eligible for SFT positive targets and RLVR. See DATA.md and TRAINING.md.

Current contents: most reasoning types hold 1,300 examples (1,040 train / 260 eval); analogical holds 6,226 (6,000 train / 226 eval) in this season and is designed to scale much further across seasons (see below), spanning many domains and several distinct question formats per type. The verifiable types (deductive, inductive, probabilistic, counterfactual, causal) were checked by executing their verifier; abductive and moral-ethical are template-authored with their verification method recorded, awaiting an independent Solver/judge pass.

Analogical is a **seed-driven sampling engine**, not a fixed set. `--per-type` sets train volume and each `--seed` is a fresh "season" that yields new, non-overlapping items (dedup is enforced against everything already on disk), so the corpus grows to tens of thousands by appending seasons:

```
python tools/generate.py --only analogical --replace --per-type 6000 --seed 7  --write   # season 1 (fresh eval + train)
python tools/generate.py --only analogical           --per-type 6000 --seed 8  --write   # season 2 (appends new train; eval saturates)
python tools/generate.py --only analogical           --per-type 6000 --seed 9  --write   # season 3 ...
```

Two layers keep volume honest (no memorization):
- **Breadth layer (bounded, deterministic).** Every distinct answer-identity in the banks -- ~43 relations, 12 cross-domain structures across ~19 domains, and the trap skins -- is emitted once, with a phrasing and register fixed by a stable hash of its identity, so it renders identically every season and never re-appears under a new surface.
- **Volume layer (unbounded, parametric).** Numeric analogies, a power-law scaling trap, the binding-constraint trap, and the competing-relations trap draw rng numbers that are recomputed and verified in every trace, so each item has a genuinely different answer. This is what scales to tens of thousands without repeating answers.

Analogical was rebuilt for cross-domain transfer, which is the type's whole point (see DOMAINS.md, "analogical breadth and shared templates"). It has three layers:

1. **Cross-domain isomorphism (the core, ~36% of train).** A library of abstract relational structures (flow driven by a potential difference, feedback to a setpoint, equilibrium of opposing forces, a bottleneck capping output, a buffer absorbing shocks, a selective gatekeeper, part-whole composition, compounding feedback), each instantiated across many *unrelated* domains. Each item asks "what plays the same role in system B that X plays in system A," so source and target are always different domains and only the structure -- never shared vocabulary (a build-time leakage guard enforces this) -- yields the answer. Two question formats (role-map and proportional) and difficulty-4 distractor variants.
2. **Difficulty-4/5 trap kernels.** Series blockage, binding-constraint-by-rate, exception-to-exception, and competing relations, each skinned across domains, each planting a surface distractor or competing relation and rejecting it by role; the numeric traps are recomputed in the trace.
3. **Single-relation completion (difficulty 1-3).** Foundational relation induction across ~35 relations with explicit role labels; traces show the actual mapping, not boilerplate. Each analogy appears once, with the phrasing rotated across items for format variety without duplication.

Anti-memorization: train and eval draw from disjoint answer pools; whole relations, whole trap-skins, and held-out target systems/domains are reserved for eval, so eval measures applying a known structure to a *new* domain, not recall. Difficulty spans 1-5 in both splits (39% of train is difficulty 3+), across 19 of the 20 canonical domains. Reproduce with `python tools/generate.py --only analogical --replace --write` (deterministic).
