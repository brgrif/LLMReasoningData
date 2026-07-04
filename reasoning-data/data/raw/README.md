# data/raw/

Generated candidates awaiting the deterministic gates (`{type}.raw.jsonl`). May contain unverified or failing records. Nothing here is training-eligible until it is promoted to `curated/` or moved to `negatives/`. Along with `negatives/`, this is one of only two locations where a `verification.passed` false record may appear. See DATA.md.

Current contents: one 500-record candidate batch from the GENERATOR.md engine (run salt 338103), split per type as `{type}.raw.jsonl` (IDs continue the corpus at 005160+). It is diversity-engineered rather than procedural: 76 surface domains, all 20 registers, the full L1-L5 difficulty ladder, and 20 isomorphism clusters. See `BATCH-338103.report.md` for the full diversity report. These are pre-gate: they carry authored `verification.passed` true but await the independent Solver/Critic and formal gate before any promotion to `curated/`.
