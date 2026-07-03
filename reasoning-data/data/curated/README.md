# data/curated/

Items that passed every gate. Positives only: `is_correct` true and `verification.passed` true. Split into `{type}.train.jsonl` and `{type}.eval.jsonl`; eval problems are never used as generation seeds and never appear in a train file. This is the only set eligible for SFT positive targets and RLVR. See DATA.md and TRAINING.md.

Current contents: 800 examples per reasoning type (640 train / 160 eval), 7,200 total, spanning many domains and several distinct question formats per type. The verifiable types (deductive, inductive, probabilistic, counterfactual, causal) were checked by executing their verifier; abductive, analogical, and moral-ethical are template-authored with their verification method recorded, awaiting an independent Solver/judge pass.
