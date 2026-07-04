# tools/

Reusable tooling for the corpus.

- `generate.py` — procedural/parametric batch generator (verifiable types are
  executed and checked). Documented below.
- `deal_season.py` — the GENERATOR.md dealer. Shuffles the season decks and
  deals one spec per item across the nine diversity dimensions, enforcing the
  composition targets, the no-consecutive-repeat rule, the per-domain cap, and
  the isomorphism quota. The decks ship large (~115 surface domains, 26
  registers, 20 kernels, 11 twists) so per-item collisions stay rare. Emits
  `specs.json` (`{"salt":.., "items":[..]}`) that drives the authoring pass.
  Change the decks in the file to launch a new season; the run salt is drawn
  fresh each run. `python deal_season.py specs.json`. Note: this path produces
  **pre-gate** candidates for `data/raw/`, not curated records.
- `validate_batch.py` — validates an authored GENERATOR.md batch against
  SCHEMA.md and its dealt specs, scans the season ban list, and runs the
  end-of-batch near-duplicate self-audit. `python validate_batch.py specs.json
  data/raw/'*.raw.jsonl'`. Exits non-zero on any record error.

## generate.py

Produces a batch of new reasoning traces (default 500 per type - a "500-shot"
batch), verifies the verifiable types by executing their verifier, and appends
to `data/curated/` (and `data/negatives/`), continuing IDs above the global
per-type maximum and de-duplicating against whatever is already on disk. Every
emitted record validates against SCHEMA.md. This is the tool that grows
`curated/`; the GENERATOR.md path (above) grows only `raw/`.

### Usage

```
python tools/generate.py                       # dry run, 500/type, prints a report
python tools/generate.py --write               # append a 500-shot batch
python tools/generate.py --per-type 200 --seed 42 --write
python tools/generate.py --report-only         # current corpus stats, no generation
```

Safe by default: it prints a report and writes nothing unless you pass
`--write`.

### Options

| flag | default | meaning |
| --- | --- | --- |
| `--per-type` | 500 | new records to add per reasoning type this run |
| `--seed` | 7 | RNG seed; change per season for fresh items |
| `--train-frac` | 0.8 | fraction of new records routed to train (rest to eval) |
| `--neg-frac` | 0.25 | paired negatives per verifiable type, as a fraction of its new train records |
| `--created` | today | ISO date stamped on `provenance.created` |
| `--root` | repo root | directory holding `DOMAINS.md` and `data/` |
| `--write` | off | actually append (otherwise dry run) |
| `--report-only` | off | just print current corpus stats |

### How it serves the goal

The corpus aims at a transferable logical foundation, not memorized patterns.
The generator is organized around four levers:

1. **Verifiability.** Deductive, inductive, probabilistic, counterfactual, and
   causal items are generated *and checked* by running the verifier in Python
   (truth-table entailment, executed rules, computed posteriors, run
   world-models, causal-graph analysis). An item is emitted only if it passes.
2. **Format invariance.** Each type rotates through several question formats so
   the model learns the operation, not one phrasing.
3. **Domain width.** The same logical structure is skinned across many DOMAINS.md
   domains, so structure - not vocabulary - solves the problem (RULES.md Rule 5).
4. **Contrastive signal.** Verifiable types emit paired negatives encoding the
   type's characteristic fallacy (RULES.md Rule 9), for preference data.

Difficulty is tagged 1-5 for curriculum ordering.

### Reuse across seasons

- **New season:** change `--seed`. The verifiable and metacognitive generators
  are randomized/parametric, so a new seed produces genuinely new items with no
  code change.
- **Bank-limited types:** the authored types (abductive, moral-ethical) draw
  from `BANK`-marked lists in `generate.py`. When the report shows one produced
  fewer than requested (`<-- SHORT`), extend the relevant bank; the tool never
  pads a shortfall with duplicates.
- **Analogical** is a seed-driven SAMPLING ENGINE (`build_analogical`), not a
  fixed set. `--per-type` sets train volume; each `--seed` is a fresh season that
  yields new, non-overlapping items (dedup against everything on disk). Two
  layers:
  - *Breadth (bounded, deterministic):* `ANA_REL` (single-relation completion,
    ~43 relations with role labels) + `STRUCTURES` (the cross-domain isomorphism
    engine: 12 abstract structures instantiated across ~19 domains, asked as
    "what plays the same role in system B as X in system A", with a leakage guard
    blocking any item whose answer appears in the source) + the T1/T3 trap skins.
    Each answer-identity is emitted once, with phrasing/register fixed by a stable
    hash of its identity, so it renders identically every season and never
    reappears under a new surface.
  - *Volume (unbounded, parametric):* `_mk_num` (clean numeric analogy, d2-3),
    `_mk_scale` (power-law transfer, d4), `_rand_t2_nums`/`_mk_t2`
    (binding-constraint, d5), `_rand_t4`/`_mk_t4` (competing relations, d5). Every
    item draws rng numbers that are recomputed and verified in the trace, so each
    has a different answer -- this scales to tens of thousands without repeating
    answers.
  - Train/eval never overlap: eval is a deterministic, held-out benchmark
    (`_analogical_eval`) of unseen relations / trap-skins / target systems, in the
    plain register, so it is identical every season and saturates on append.
  - Season 1 (fresh eval + train): `--only analogical --replace --per-type N --seed S --write`.
    Later seasons (append train only): drop `--replace`, change `--seed`.
    To grow structural/domain BREADTH, extend `ANA_REL`/`STRUCTURES`; the engine
    fans each addition across domains, formats, and registers automatically.
- The tool dedups against existing problems and allocates new IDs above the
  global per-type maximum (across `curated/`, `raw/`, and `negatives/`), so
  re-running is safe, additive, and never collides with pre-gate raw candidates.
- If a type's whole candidate space is already on disk, the report prints
  `ADDED NOTHING to curated` for it and writes nothing for that type - change
  `--seed` (parametric types) or extend its `BANK` (authored types) to grow it.

### Honesty

Verifiable types are genuinely checked. The rubric/ranking types
(abductive = `answer_match` by ranking, analogical = `process_check`,
moral-ethical = `rubric_judge`) are authored so the planted ground truth matches
by construction; they carry their verification method but await an independent
Solver/judge pass (PIPELINE.md). The report marks any type whose achievable
count fell short, so coverage is never overstated.
