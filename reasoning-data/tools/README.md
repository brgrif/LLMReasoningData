# tools/

Reusable tooling for the corpus.

- `generate.py` — procedural/parametric batch generator (verifiable types are
  executed and checked). Documented below.
- `deal_season.py` — the GENERATOR.md dealer. Shuffles the season decks and
  deals one spec per item across the nine diversity dimensions, enforcing the
  composition targets, the no-consecutive-repeat rule, the per-domain cap, and
  the isomorphism quota. Emits `specs.json` (`{"salt":.., "items":[..]}`) that
  drives the authoring pass. Change the decks in the file to launch a new
  season; the run salt is drawn fresh each run. `python deal_season.py specs.json`
- `validate_batch.py` — validates an authored GENERATOR.md batch against
  SCHEMA.md and its dealt specs, scans the season ban list, and runs the
  end-of-batch near-duplicate self-audit. `python validate_batch.py specs.json
  data/raw/'*.raw.jsonl'`. Exits non-zero on any record error.

## generate.py

## generate.py

Produces a batch of new reasoning traces (default 500 per type - a "500-shot"
batch), verifies the verifiable types by executing their verifier, appends to
`data/curated/` (and `data/negatives/`), continuing IDs and de-duplicating
against whatever is already on disk. Every emitted record validates against
SCHEMA.md.

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
- **Bank-limited types:** the authored types (abductive, analogical,
  moral-ethical) draw from `BANK`-marked lists in `generate.py`. When the report
  shows one produced fewer than requested (`<-- SHORT`), extend the relevant
  bank; the tool never pads a shortfall with duplicates.
- The tool dedups against existing problems and continues IDs, so re-running is
  safe and additive.

### Honesty

Verifiable types are genuinely checked. The rubric/ranking types
(abductive = `answer_match` by ranking, analogical = `process_check`,
moral-ethical = `rubric_judge`) are authored so the planted ground truth matches
by construction; they carry their verification method but await an independent
Solver/judge pass (PIPELINE.md). The report marks any type whose achievable
count fell short, so coverage is never overstated.
