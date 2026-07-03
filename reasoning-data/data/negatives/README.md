# data/negatives/

Retained incorrect traces (`{type}.negatives.jsonl`), every record with `is_correct` false. Contrastive signal for preference optimization and source material for metacognitive data (RULES.md, Rule 9). A negative inherits the split of its paired positive, so a negative paired to an eval item never enters training. See DATA.md.

Current contents: 20 paired negatives for each verifiable type (deductive, inductive, probabilistic, counterfactual, causal), 100 total. Each is paired to a train positive via its `-neg` id and a note, and encodes that type's characteristic fallacy.
