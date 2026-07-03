# Data

Where the actual datasets live. This defines the on-disk layout, the lifecycle a record moves through, and the invariants each location enforces. An initial batch of 100 curated examples per reasoning type (80 train / 20 eval, 900 total) plus 100 paired negatives for the verifiable types now populates `data/`.

## Layout

```
reasoning-data/
  data/
    seeds/       {type}.seeds.jsonl       hand-written bootstrap problems
    raw/         {type}.raw.jsonl         generated candidates, pre-gate
    curated/     {type}.train.jsonl       passed all gates, positives only
                 {type}.eval.jsonl        held-out, never used as seeds
    negatives/   {type}.negatives.jsonl   retained incorrect traces
```

`{type}` is the full reasoning-type name (deductive, inductive, abductive, analogical, causal, counterfactual, probabilistic, metacognitive, moral-ethical), matching `reasoning_type` in SCHEMA.md. There is one file per reasoning type per location, and per split within curated.

## Lifecycle

A record moves left to right through the pipeline (PIPELINE.md):

1. **seeds** bootstrap the Generator. Small, hand-written, high quality. Never drawn from eval.
2. **raw** holds Generator/Solver/Critic output awaiting the deterministic gates. May contain unverified or failing records.
3. The gates run: formal verification, dedup, difficulty tagging, diversity check, decontamination audit, human spot-check.
4. **curated** receives items that pass every gate. Positives only.
5. **negatives** receives items that fail with an informative error. `is_correct` false.

## Invariants

- **curated/** — every record has `is_correct` true and `verification.passed` true. This is the only set eligible for SFT positive targets and RLVR (TRAINING.md). It is split into train and eval; eval problems are never used as generation seeds and never appear in any train file.
- **negatives/** — every record has `is_correct` false. This is the contrastive signal for preference optimization and the source material for metacognitive data (RULES.md, Rule 9). A negative inherits the split of its paired positive, so a negative paired to an eval item never enters training.
- **raw/** and **negatives/** — the only locations where a `verification.passed` false record may appear (SCHEMA.md validation rules). curated never contains a failed record.
- **seeds/** — hand-written; `provenance.source` records this, and items generated from a seed reference it via `provenance.seed_id`.
- Every record in every location conforms to SCHEMA.md. A record's location changes its training eligibility, never its schema.

## Decontamination and derived data

Decontamination audits run against all evaluation sets before any batch is promoted to curated, and again after any synthesis step (RULES.md, Rule 1). Decontamination hashes, token counts, and chat-template formatting are computed by tooling at audit or training time and are never written into these files (SCHEMA.md).
