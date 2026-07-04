# data/curated/

Items that passed every gate. Positives only: `is_correct` true and `verification.passed` true. Split into `{type}.train.jsonl` and `{type}.eval.jsonl`; eval problems are never used as generation seeds and never appear in a train file. This is the only set eligible for SFT positive targets and RLVR. See DATA.md and TRAINING.md.

Current contents (train / eval / paired negatives):

| type | train | eval | negatives |
| --- | --- | --- | --- |
| deductive | 100,000 | 120 | 25,000 |
| inductive | 100,000 | 86 | 25,000 |
| counterfactual | 100,000 | 92 | 25,000 |
| metacognitive | 100,000 | 82 | -- |
| analogical | 100,000 | 548 | -- |
| abductive | 100,000 | 156 | -- |
| probabilistic | 80,000 | 110 | 20,000 |
| causal | 45,000 | 102 | 11,249 |
| moral-ethical | 6,000 | 240 | -- |

Each type is scaled to **the top of its honest limit** (measured, not padded):

- **100,000** for the six types whose parametric/compound problem space is
  effectively unbounded (deductive ~10^12 chains, inductive/counterfactual/
  metacognitive/analogical parametric, abductive over adjective x noun components).
- **80,000 / 45,000** for probabilistic and causal, which have *fixed*
  combinatorial grids (measured ceilings ~97,920 and ~55,000); held just under
  the ceiling so the season never has to enumerate the whole space.
- **6,000** for moral-ethical, which is bank-bound (dilemmas x 15 analysis
  modes, ceiling ~6,030) -- extended honestly rather than padded to a round
  number, since RULES.md forbids padding with near-duplicate answers.

Every split is leakage-free (0 train/eval overlap; 0 structural-signature +
answer leakage, analogical 2 by coincidence) and spans difficulty 1-5
(moral-ethical 2-5). Total corpus: ~838k records.

Splits larger than ~95 MiB are sharded (`{type}.train.jsonl` +
`{type}.train.NNN.jsonl`) so no file exceeds the 100 MiB host limit; the shards
are logically one split (see SCHEMA.md). The tooling reads them transparently.

**All nine types are now seed-driven with a deterministic, held-out eval** (not just analogical). Each `build_<type>` emits a fixed benchmark drawn from a *reserved* parameter/entity region (disjoint from train) plus a parametric/bank train layer, so: train and eval never share a problem; the (structural-signature, final_answer) leakage between them is zero (analogical: 1-2, coincidence); and the eval is identical every season (it saturates on append while train grows). Difficulty spans 1-5 for every type (moral-ethical spans 2-5: it has no trivial dilemmas). Every record is `generation_method: procedural` with an honest `provenance.source` -- nothing is labeled `human_expert`/`hand-authored`, because the whole corpus is produced by `tools/generate.py`. The verifiable types (deductive, inductive, probabilistic, counterfactual, causal) are checked by executing their verifier in Python and only passing items are emitted; analogical/abductive/metacognitive/moral-ethical carry their verification method (process_check / answer_match / rubric_judge) by construction and await an independent Solver/judge pass.

Reproduce the corpus (deterministic, per the exact commands in the season log below):

```
# season 1 (fresh eval + train); season 2 appends new train, eval saturates
python tools/generate.py --only deductive     --replace --per-type 3000 --seed 100 --write
python tools/generate.py --only deductive               --per-type 3000 --seed 101 --write
# ... analogical/inductive/probabilistic/counterfactual/causal/abductive/metacognitive similarly;
# moral-ethical is bank-bound: one --replace season at --per-type 2000 reaches its ceiling.
```

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

## Audit before generator rewrites

Before rebuilding any non-analogical reasoning type, run the read-only split
audit so the rewrite has concrete before/after numbers and no new data is
created during the audit phase:

```
python tools/audit_reasoning_types.py --types deductive inductive abductive causal counterfactual probabilistic metacognitive moral-ethical
```

The audit reports exact train/eval problem overlap, first-step/answer and
structural-signature/answer leakage, distinct step text counts by trace index,
difficulty coverage (including whether levels 4/5 are absent), canonical-domain
coverage, generation/provenance/verification labels, and a small spot-check
metadata sample. Use those numbers as the baseline for the one-type-at-a-time
rewrite acceptance checks.
