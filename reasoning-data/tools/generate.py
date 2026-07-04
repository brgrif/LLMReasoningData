#!/usr/bin/env python3
"""
generate.py - reusable batch generator for the reasoning-data corpus.

WHAT IT DOES
    Produces a batch of new reasoning traces (default 500 per type - a
    "500-shot" batch), appends them to data/curated/ (and data/negatives/ for
    the verifiable types), continuing IDs above the global per-type maximum
    (so an appended batch never collides with a pre-gate raw candidate) and
    de-duplicating against whatever is already on disk. Every record it emits
    validates against SCHEMA.md.

DESIGN (why it is built this way)
    The goal of this corpus is a transferable *logical foundation* for LLMs -
    reasoning the model performs, not patterns it memorizes. Four levers serve
    that goal, and the generator is organized around them:

      1. Verifiability. Verifiable types (deductive, inductive, probabilistic,
         counterfactual, causal) are generated *and checked* by running the
         actual verifier here in Python (truth-table entailment, executed
         rules, computed posteriors, run world-models, causal-graph analysis).
         An item is only emitted if its check passes.
      2. Format invariance. Each type rotates through several distinct question
         FORMATS, so the model learns the operation rather than one phrasing.
      3. Domain width. The same logical structure is skinned across many of the
         subject domains in DOMAINS.md, so only the structure - not vocabulary
         co-occurrence - solves the problem (RULES.md Rule 5).
      4. Contrastive signal. Verifiable types emit paired NEGATIVES encoding the
         type's characteristic fallacy (RULES.md Rule 9), for preference data.

    Difficulty is tagged 1-5 for curriculum ordering. Broad, clean,
    compositional coverage across difficulty is what tends to produce emergent
    generalization, so breadth and correctness are prioritized over raw volume.

SEED-DRIVEN, WITH A HELD-OUT EVAL FOR EVERY TYPE
    - Every build_<type> emits a DETERMINISTIC held-out eval from a RESERVED
      parameter/entity region (disjoint from train) plus a parametric/bank train
      layer. So train and eval never share a problem, the (structural-signature,
      final_answer) leakage is ~0, and the eval is identical every season (it
      saturates on append while train grows). Difficulty spans 1-5 per type.
    - Change --seed for a fresh season. All parametric generators (the five
      verifiable types, metacognitive, abductive's localization layer, analogical)
      yield genuinely new train items on a new seed. `causal` is parametric too
      (confounder / mediation / RCT / collider / Simpson families).
    - moral-ethical is bank-bound (MORAL_BANK x analysis modes); scale it by
      extending MORAL_BANK. abductive mixes a CAPPED bank layer (each base once,
      stem fixed by a stable hash) with a parametric fault-localization layer.
    - Idempotent-friendly: dedups against existing problems and continues IDs,
      so re-running never creates duplicate problems.

HONESTY
    Every record is generation_method=procedural (this script produces all of
    it; nothing is labeled human_expert/multi_agent/hand-authored). The five
    verifiable types are genuinely checked -- the verifier runs and only passing
    items are emitted. The rubric/ranking types (abductive=answer_match,
    analogical/metacognitive=process_check, moral-ethical=rubric_judge) match
    their planted ground truth by construction and await an independent
    Solver/judge pass (see PIPELINE.md). The generator never pads a shortfall:
    if a type cannot produce the requested count of new, distinct problems it
    emits fewer and says so (`<-- SHORT`) in the report.

USAGE
    python tools/generate.py                      # dry run, 500/type, prints report
    python tools/generate.py --write              # actually append a 500-shot batch
    python tools/generate.py --per-type 200 --seed 42 --write
    python tools/generate.py --report-only        # just show current corpus stats
"""
import argparse, datetime, glob, json, os, random, re
from itertools import product, combinations

# --------------------------------------------------------------------------
# Schema constants (mirror SCHEMA.md)
# --------------------------------------------------------------------------
RT = ["deductive","inductive","abductive","analogical","causal",
      "counterfactual","probabilistic","metacognitive","moral-ethical"]
PREFIX = {"deductive":"ded","inductive":"ind","abductive":"abd","analogical":"ana",
          "causal":"cau","counterfactual":"cfa","probabilistic":"prb",
          "metacognitive":"met","moral-ethical":"mor"}
VERIFIABLE = {"deductive","inductive","probabilistic","counterfactual","causal"}
GEN_METHODS = {"procedural","teacher_distillation","self_instruct","multi_agent","human_expert"}
VMETHODS = {"symbolic_solver","code_execution","answer_match","process_check","rubric_judge"}
LABELS = {"valid","invalid","unverified"}
ID_RE = re.compile(r'^(ded|ind|abd|ana|cau|cfa|prb|met|mor)-\d{6}(-neg)?$')
REQ = ["id","reasoning_type","domain","problem","reasoning_trace","final_answer",
       "is_correct","difficulty","generation_method","verification","provenance"]

def load_canonical_domains(root):
    txt = open(os.path.join(root, "DOMAINS.md")).read()
    return set(re.findall(r'^- \*\*(.+?)\.\*\*', txt, re.M))

def cap(s): return s[0].upper() + s[1:] if s else s

# --------------------------------------------------------------------------
# Verification engines
# --------------------------------------------------------------------------
def bayes(prior, sens, spec):
    return sens*prior / (sens*prior + (1-spec)*(1-prior))

def entails(varnames, premises, conclusion):
    """Propositional entailment by truth-table enumeration."""
    for vals in product([False, True], repeat=len(varnames)):
        env = dict(zip(varnames, vals))
        if all(eval(p, {}, env) for p in premises) and not eval(conclusion, {}, env):
            return False
    return True

# --------------------------------------------------------------------------
# Record builder + validator
# --------------------------------------------------------------------------
def make(rid, rt, domain, problem, steps, final, is_correct, difficulty, gen,
         vm, vp, vd, source, seed_id, created, confidence=None, notes=None):
    trace = [{"step": i+1, "text": t, "label": l} for i, (t, l) in enumerate(steps)]
    r = {"id": rid, "reasoning_type": rt, "domain": domain, "problem": problem,
         "reasoning_trace": trace, "final_answer": final, "is_correct": is_correct,
         "difficulty": difficulty, "generation_method": gen,
         "verification": {"method": vm, "passed": vp, "details": vd},
         "provenance": {"source": source, "seed_id": seed_id, "created": created}}
    if confidence is not None:
        out = {}
        for k, v in r.items():
            out[k] = v
            if k == "is_correct":
                out["confidence"] = confidence
        r = out
    if notes is not None:
        r["notes"] = notes
    return r

def validate(r, location, canon):
    e = []
    for k in REQ:
        if k not in r: e.append(f"missing {k}")
    if not ID_RE.match(r.get("id","")): e.append(f"bad id {r.get('id')}")
    if r["reasoning_type"] not in RT: e.append("bad reasoning_type")
    if r["domain"] not in canon: e.append(f"noncanonical domain {r['domain']!r}")
    steps = [s["step"] for s in r["reasoning_trace"]]
    if steps != list(range(1, len(steps)+1)): e.append("steps not 1..n")
    if any(s["label"] not in LABELS for s in r["reasoning_trace"]): e.append("bad label")
    if r["generation_method"] not in GEN_METHODS: e.append("bad generation_method")
    if r["verification"]["method"] not in VMETHODS: e.append("bad verification method")
    if not isinstance(r["is_correct"], bool): e.append("is_correct not bool")
    if not (1 <= r["difficulty"] <= 5): e.append("difficulty out of range")
    if "confidence" in r and not (0.0 <= r["confidence"] <= 1.0): e.append("confidence out of range")
    if r["reasoning_type"] == "moral-ethical" and r["verification"]["method"] != "rubric_judge":
        e.append("moral-ethical must use rubric_judge")
    if location == "curated" and (not r["is_correct"] or not r["verification"]["passed"]):
        e.append("curated impurity")
    if location == "negatives" and r["is_correct"]: e.append("negative is_correct true")
    return e

# --------------------------------------------------------------------------
# Wide subject/domain banks (shared)
# --------------------------------------------------------------------------
# Subjects grouped by DOMAINS.md domain -> deductive/metacognitive skin the same
# formal structure across all of these.
DOMAIN_SUBJECTS = {
    "logic puzzles": ["token","tile","marker","card","switch","lever","glyph","peg","cog","dial","rune","chit"],
    "program behavior": ["record","request","job","event","session","message","packet","thread","handle","socket","token","frame"],
    "finance and business operations": ["invoice","transaction","account","order","claim","payment","ledger entry","receipt","refund","voucher","remittance","statement"],
    "mechanical / systems troubleshooting": ["part","unit","valve","motor","panel","gauge","relay","bearing","actuator","coupling","fuse","seal"],
    "chemistry": ["sample","batch","compound","solution","vial","reagent","aliquot","precipitate","buffer","titrant","isolate","distillate"],
    "medicine-style diagnosis": ["chart","specimen","dose","case","scan","sample","culture","panel","biopsy","referral","swab","record"],
    "incident and root-cause analysis": ["alert","ticket","deploy","signal","log entry","trace","incident","page","runbook","postmortem","alarm","span"],
    "science": ["reading","measurement","trial","observation","specimen","dataset","sample","assay","run","survey","record","probe"],
    "law and regulation": ["filing","contract","permit","case file","statute","affidavit","motion","brief","subpoena","clause","ordinance","docket"],
    "economics and markets": ["asset","listing","position","order","contract","lot","bid","quote","tranche","instrument","holding","warrant"],
    "engineering and physical systems": ["beam","circuit","sensor","joint","module","bearing","strut","node","gasket","conduit","rotor","truss"],
    "biology and ecology": ["culture","specimen","colony","plot","sample","population","isolate","transect","clutch","graft","strain","quadrat"],
    "formal grammars and symbol systems": ["string","token","symbol","expression","rule","glyph","lexeme","production","cipher","word","tape","node"],
    "algorithms and program analysis": ["node","invariant","subroutine","iteration","state","predicate","branch","loop","assertion","register","frame","edge"],
}
# Generic state predicates for formal chains (domain-neutral, so any subject fits).
PREDICATES = ["flagged","queued","archived","approved","sealed","escalated","audited",
              "locked","expired","tagged","routed","verified","quarantined","published",
              "indexed","frozen","released","logged","signed","cleared","batched","held",
              "certified","deferred","encrypted","mirrored","notarized","pinned","ranked",
              "reconciled","redacted","reserved","stamped","suspended","syndicated","throttled",
              "vaulted","whitelisted","annotated","bonded","chartered","dispatched","embargoed",
              "endorsed","forwarded","gated","harmonized","impounded","licensed","migrated",
              "provisioned","ratified","sandboxed","staged","tokenized","triaged","vetted"]

def _domain_subject(rng):
    dom = rng.choice(list(DOMAIN_SUBJECTS))
    return dom, rng.choice(DOMAIN_SUBJECTS[dom])

# Held-out predicate vocabulary: the last 16 PREDICATES are reserved for eval
# ONLY (deductive + metacognitive), so an eval item's answer/relation predicate
# never appears in any train item -> train/eval answer pools are disjoint by
# construction and (structural-signature, answer) leakage is impossible.
PRED_EVAL = PREDICATES[-16:]
PRED_TRAIN = PREDICATES[:-16]

# ==========================================================================
# GENERATORS  (build_<type>(need, rng, exclude) -> list[instance dict])
# Each instance: {domain, problem, steps, final, difficulty, vm, vd,
#                 optional: confidence, and raw fields for the negative builder}
# ==========================================================================

DED_FMTS = 4
def _entails_chain(preds, hops):
    """Run the symbolic verifier: forward implication chain + fact entails last."""
    vs = [f"c{i}" for i in range(len(preds))]
    prem = [f"(not {vs[i]}) or {vs[i+1]}" for i in range(hops)]
    return entails(vs, prem + [vs[0]], vs[-1])

def _mk_deductive(dom, subj, preds, hops, fmt, distract, distractor_pred, name, split):
    """Build one deductive item. difficulty == hops (1..5); the surface
    distractor is flavor only and does NOT change difficulty (so level 3 is
    reachable). Answers are the terminal predicate `preds[-1]`."""
    first, last = preds[0], preds[-1]
    def rule_line(i, lead):
        subjref = f"the {subj}" if i == 0 and lead else "it"
        return f"If {subjref} is {preds[i]}, then it is {preds[i+1]}."
    prose = "Rules: " + " ".join(rule_line(i, i == 0) for i in range(hops))
    dtx = (f" Separately, if it is {distractor_pred}, then a receipt prints." if distract else "")
    if fmt == 0:
        prob = prose + dtx + f" Fact: the {subj} is {first}. Given only these rules, is it necessarily {last}?"
        final = f"Yes, necessarily {last}."
    elif fmt == 1:
        bul = "Consider these rules:\n" + "\n".join(f"- if the {subj} is {preds[i]} then it is {preds[i+1]}" for i in range(hops))
        prob = bul + dtx + f"\nObserved: the {subj} is {first}. Does it follow that it is {last}?"
        final = f"Yes, it follows that the {subj} is {last}."
    elif fmt == 2:
        prob = prose + dtx + f" The {subj} is {first}. {name} concludes it is {last}. Is that conclusion valid?"
        final = f"Yes, {name}'s conclusion is valid: the {subj} is {last}."
    else:
        prob = prose + dtx + f" Fact: the {subj} is {first}. Following the rules to the end, what must be true of the {subj}?"
        final = f"The {subj} is {last}."
    steps = [(f"Premise {i+1}: {preds[i]} -> {preds[i+1]}.", "valid") for i in range(hops)]
    if distract: steps.append(("The receipt rule shares no terms with the chain; a distractor, unused.", "valid"))
    steps.append((f"Fact: the {subj} is {first}.", "valid")); cur = first
    for i in range(hops):
        steps.append((f"From '{cur}' and premise {i+1}, conclude '{preds[i+1]}' (modus ponens).", "valid")); cur = preds[i+1]
    steps.append(("Check: only the premises and valid modus ponens are used.", "valid"))
    return dict(domain=dom, problem=prob, steps=steps, final=final,
                difficulty=min(max(1, hops), 5), vm="symbolic_solver",
                vd=f"Encoded a length-{hops} implication chain and the fact; truth-table check confirms entailment.",
                split=split, subj=subj, preds=preds, hops=hops)

# Deterministic held-out eval: reserved predicate vocabulary (PRED_EVAL) only,
# so eval answers are disjoint from every train answer. Stable across seasons.
DED_EVAL_DOMS = [("logic puzzles", "glyph"), ("program behavior", "packet"),
                 ("law and regulation", "statute"), ("chemistry", "reagent"),
                 ("science", "reading"), ("finance and business operations", "ledger entry")]
def _deductive_eval(seen):
    out = []
    def add(it):
        if it and it["problem"] not in seen:
            seen.add(it["problem"]); out.append(it)
    ring = PRED_EVAL + PRED_EVAL
    idx = 0
    for di, (dom, subj) in enumerate(DED_EVAL_DOMS):
        for hops in (1, 2, 3, 4, 5):
            for fmt in range(DED_FMTS):
                start = (idx * 3) % len(PRED_EVAL)
                preds = ring[start:start + hops + 1]
                if not _entails_chain(preds, hops):
                    idx += 1; continue
                distract = hops >= 3
                dp = ring[(start + hops + 1) % len(PRED_EVAL)]
                nm = NAMES[(di * 20 + hops * 4 + fmt) % len(NAMES)]
                add(_mk_deductive(dom, subj, preds, hops, fmt, distract, dp, nm, "eval"))
                idx += 1
    return out

def build_deductive(need, rng, exclude):
    out, seen = [], set(exclude)
    for it in _deductive_eval(seen):
        out.append(it)                      # deterministic held-out eval (saturates)
    made, attempts = 0, 0
    while made < need and attempts < need*40 + 4000:
        attempts += 1
        dom, subj = _domain_subject(rng)
        hops = rng.randint(1, 5)
        preds = rng.sample(PRED_TRAIN, hops+1)   # train draws ONLY train-pool predicates
        if not _entails_chain(preds, hops):      # run verifier; drop failures
            continue
        distract = hops >= 2 and rng.random() < 0.4   # flavor only, difficulty == hops
        dp = rng.choice([p for p in PRED_TRAIN if p not in preds])
        fmt = rng.randrange(DED_FMTS); nm = rng.choice(NAMES)
        it = _mk_deductive(dom, subj, preds, hops, fmt, distract, dp, nm, "train")
        if it["problem"] in seen:
            continue
        seen.add(it["problem"]); out.append(it); made += 1
    return out

def neg_deductive(tid, it):
    # A proper negative COMMITS the fallacy and reaches the WRONG answer (SCHEMA.md):
    # given the chain first -> ... -> last, it observes `last` and wrongly concludes
    # `first` (affirming the consequent), answering "Yes" when the truth is "No".
    subj, preds, hops = it["subj"], it["preds"], it["hops"]
    first, last = preds[0], preds[-1]
    prob = ("Rules: " + " ".join(f"If the {subj} is {preds[i]}, then it is {preds[i+1]}." for i in range(hops))
            + f" Observed: the {subj} is {last}. Someone concludes the {subj} must therefore be {first}. Is that inference valid?")
    return make(tid+"-neg", "deductive", it["domain"], prob,
        [(f"Observed: the {subj} is {last}.", "valid"),
         (f"Read the final rule backward: since '{preds[hops-1]} -> {last}', treat {last} as giving back '{preds[hops-1]}'.", "invalid"),
         (f"Chain that backward reading to the start and land on '{first}'.", "invalid"),
         (f"Conclude the {subj} must be {first}.", "invalid")],
        f"Yes, the {subj} must be {first}.", False, it["difficulty"], "procedural",
        "symbolic_solver", False,
        f"Affirms the consequent: observing '{last}' is consistent with '{first}' being false, so '{last}' does not entail '{first}'. The correct answer is No.",
        "procedural-gen, verified", None, it["_created"],
        notes=f"paired positive: {tid}. Fallacy: affirming the consequent.")

# Inductive: infer the rule behind shown pairs, then apply it. Numeric families
# span difficulty 1..5 (shift/scale/affine -> quad -> quad2/cubic -> full
# quadratic). Answers are rule-revealing (include the inferred coefficients), so
# reserving a disjoint coefficient+query region for eval makes train/eval answer
# pools disjoint and eval stable across seasons.
IND_DIFF = {"shift": 1, "scale": 2, "affine": 2, "quad": 3, "quad2": 4, "cubic": 4, "polyfull": 5}
def _ind_fn(name, cf):
    a = cf.get("a", 1); b = cf.get("b", 0); c = cf.get("c", 0)
    if name == "shift":    return (lambda x: x + b),          f"input + {b}",              f"f = lambda x: x + {b}",              f"f(x) = x + {b}"
    if name == "scale":    return (lambda x: a * x),          f"{a}*input",                f"f = lambda x: {a}*x",                f"f(x) = {a}x"
    if name == "affine":   return (lambda x: a * x + b),      f"{a}*input + {b}",          f"f = lambda x: {a}*x + {b}",          f"f(x) = {a}x + {b}"
    if name == "quad":     return (lambda x: x * x + b),      f"input^2 + {b}",            f"f = lambda x: x*x + {b}",            f"f(x) = x^2 + {b}"
    if name == "quad2":    return (lambda x: a * x * x + b),  f"{a}*input^2 + {b}",        f"f = lambda x: {a}*x*x + {b}",        f"f(x) = {a}x^2 + {b}"
    if name == "cubic":    return (lambda x: a * x**3 + b),   f"{a}*input^3 + {b}",        f"f = lambda x: {a}*x**3 + {b}",       f"f(x) = {a}x^3 + {b}"
    return (lambda x: a * x * x + b * x + c), f"{a}*input^2 + {b}*input + {c}", f"f = lambda x: {a}*x*x + {b}*x + {c}", f"f(x) = {a}x^2 + {b}x + {c}"

IND_NUM_DOMS = ["mathematics", "science", "finance and business operations", "economics and markets",
                "chemistry", "engineering and physical systems", "algorithms and program analysis"]
def _mk_ind_num(name, cf, shown_xs, held_xs, q, fmt, dom, split):
    f, dsc, code, namestr = _ind_fn(name, cf)
    shown = [(x, f(x)) for x in shown_xs]; held = [(x, f(x)) for x in held_xs]
    pairs = ", ".join(f"{x}->{y}" for x, y in shown)
    if fmt == 0:
        prob = f"From these pairs, state the rule and apply it to {q}: {pairs}."
        final = f"Rule: {namestr}. f({q}) = {f(q)}."
    elif fmt == 1:
        prob = f"A function produces: {pairs}. What does it output for input {q}?"
        final = f"f({q}) = {f(q)} (rule {namestr})."
    else:
        prob = f"Infer the rule behind these examples, write it as code, then evaluate at {q}: {pairs}."
        final = f"{code}; result {f(q)}."
    steps = [(f"Fit output = {dsc}.", "valid"),
             ("Check shown pairs: " + ", ".join(f"f({x})={y}" for x, y in shown) + ". All hold.", "valid"),
             (f"As code: {code}.", "valid"), (f"Apply to {q}: {f(q)}.", "valid"),
             ("Held-out check " + ", ".join(f"{x}->{y}" for x, y in held) + ": consistent.", "valid")]
    return dict(domain=dom, problem=prob, steps=steps, final=final, difficulty=IND_DIFF[name],
        vm="code_execution", vd=f"Executed {namestr} on held-out {[x for x,_ in held]}; matched and f({q})={f(q)}.",
        split=split, neg=dict(kind="num", first_in=shown[0][0], first_out=shown[0][1],
                              snd_in=shown[1][0], snd_out=shown[1][1], query=q))

def _mk_ind_seq(name, cf, split):
    f, dsc, code, namestr = _ind_fn(name, cf)
    seq = [f(i) for i in range(1, 5)]; nxt = f(5); nname = namestr.replace("x", "n")
    prob = "Find the next number in the sequence and state the rule: %s, ..." % (", ".join(map(str, seq)))
    return dict(domain="mathematics", problem=prob, difficulty=IND_DIFF[name], vm="code_execution", split=split,
        steps=[(f"Terms follow position n via {nname}.", "valid"),
               ("Check: " + ", ".join(f"n={i}->{f(i)}" for i in range(1, 5)) + ".", "valid"),
               (f"Next term (n=5): {nxt}.", "valid")],
        final=f"Next term {nxt}; rule {nname}.",
        vd=f"Executed {namestr} at n=1..5; sequence matches and the next term is {nxt}.")

STR_TX = [("reverse", lambda s: s[::-1], "reverse the string"),
          ("upper", lambda s: s.upper(), "uppercase the string"),
          ("double", lambda s: s + s, "repeat the string twice"),
          ("first-cap", lambda s: s.capitalize(), "capitalize the first letter")]
# WORDS is defined lower in the banks section; slice lazily to reserve the last
# 16 words as eval-only query words (disjoint from train query words).
def _words_eval():  return WORDS[-16:]
def _words_train(): return WORDS[:-16]
def _mk_ind_str(nm, fn, txt, demos, q, fmt, split):
    shown = [(w, fn(w)) for w in demos]
    pairs = ", ".join(f"{a}->{b}" for a, b in shown)
    prob = (f"Infer the transformation and apply it to '{q}': {pairs}." if fmt == 0
            else f"These strings follow one rule: {pairs}. What is the output for '{q}'?")
    return dict(domain="formal grammars and symbol systems", problem=prob, difficulty=2,
        vm="code_execution", split=split,
        steps=[(f"Each output applies: {txt}.", "valid"), ("Check shown pairs: consistent.", "valid"),
               (f"Apply to '{q}': '{fn(q)}'.", "valid")],
        final=f"'{fn(q)}' (rule: {txt})",
        vd=f"Executed the '{nm}' transform on shown inputs and the query; all match.",
        neg=dict(kind="str", first_in=shown[0][0], first_out=shown[0][1],
                 snd_in=shown[1][0], snd_out=shown[1][1], query=q))

def _inductive_eval(seen):
    """Deterministic held-out eval: reserved coefficients (a>=8, b>=13, c>=7),
    reserved query points (>=16), and reserved query words; rule-revealing
    answers -> disjoint from train answers."""
    out = []
    def add(it):
        if it and it["problem"] not in seen:
            seen.add(it["problem"]); out.append(it)
    combos = ([("shift", {"b": b}) for b in (13, 15, 17, 19)]
              + [("scale", {"a": a}) for a in (8, 9, 11, 12)]
              + [("affine", {"a": a, "b": b}) for a in (8, 9, 11) for b in (13, 15, 17)]
              + [("quad", {"b": b}) for b in (14, 16, 18, 20)]
              + [("quad2", {"a": a, "b": b}) for a in (8, 10) for b in (13, 17)]
              + [("cubic", {"a": a, "b": b}) for a in (8, 9) for b in (13, 20)]
              + [("polyfull", {"a": a, "b": b, "c": c}) for (a, b, c) in
                 ((8, 13, 7), (9, 15, 11), (11, 17, 9), (12, 19, 12))])
    for i, (name, cf) in enumerate(combos):
        dom = IND_NUM_DOMS[i % len(IND_NUM_DOMS)]
        for fmt, q in ((0, [17, 19, 21, 23, 25, 26][i % 6]), (2, [16, 18, 20, 22, 24, 26][i % 6] + 1)):
            add(_mk_ind_num(name, cf, [16, 18, 20], [22, 24], q, fmt, dom, "eval"))
    for name, cf in [("affine", {"a": 8, "b": 13}), ("affine", {"a": 11, "b": 19}), ("scale", {"a": 9}),
                     ("scale", {"a": 12}), ("quad", {"b": 14}), ("quad", {"b": 20}),
                     ("quad2", {"a": 8, "b": 13}), ("shift", {"b": 15})]:
        add(_mk_ind_seq(name, cf, "eval"))
    ew = _words_eval()
    for i, (nm, fn, txt) in enumerate(STR_TX):
        for k in range(3):
            demos = [ew[(i + k) % 16], ew[(i + k + 4) % 16], ew[(i + k + 8) % 16]]
            q = ew[(i + k + 12) % 16]
            add(_mk_ind_str(nm, fn, txt, demos, q, (i + k) % 2, "eval"))
    return out

def _rand_coeffs(rng, name):
    """Train-region coefficients (disjoint from the reserved eval region)."""
    cf = {}
    if name in ("scale", "affine", "quad2", "cubic", "polyfull"):
        cf["a"] = rng.randint(2, 7)
    if name in ("shift", "affine", "quad", "quad2", "cubic"):
        cf["b"] = rng.randint(1, 10) * rng.choice([1, -1])
    if name == "polyfull":
        cf["b"] = rng.randint(1, 6) * rng.choice([1, -1]); cf["c"] = rng.randint(1, 6)
    return cf

def build_inductive(need, rng, exclude):
    out, seen = [], set(exclude)
    for it in _inductive_eval(seen):
        out.append(it)
    made, attempts = 0, 0
    fams = (["shift"] + ["scale"] * 2 + ["affine"] * 2 + ["string"] * 3 + ["quad"] * 2
            + ["seq"] + ["quad2"] * 2 + ["cubic"] * 2 + ["polyfull"] * 2)
    while made < need and attempts < need * 60 + 4000:
        attempts += 1
        fam = rng.choice(fams)
        if fam == "string":
            nm, fn, txt = rng.choice(STR_TX)
            ws = rng.sample(_words_train(), 4)
            it = _mk_ind_str(nm, fn, txt, ws[:3], ws[3], rng.randrange(2), "train")
        elif fam == "seq":
            name = rng.choice(["shift", "scale", "affine", "quad", "quad2"])
            it = _mk_ind_seq(name, _rand_coeffs(rng, name), "train")
        else:
            cf = _rand_coeffs(rng, fam)
            xs = rng.sample(range(0, 15), 6)
            it = _mk_ind_num(fam, cf, xs[:3], xs[3:5], xs[5], rng.randrange(3),
                             rng.choice(IND_NUM_DOMS) if fam != "shift" else "mathematics", "train")
        if it["problem"] in seen:
            continue
        seen.add(it["problem"]); out.append(it); made += 1
    return out

def neg_inductive(tid, it):
    # Coherent overfit negative: MEMORIZE the first shown example and echo its
    # output for everything, which fits pair 1 but fails the next pair. Only the
    # pair-style families (numeric, string) carry `neg`; sequence items skip.
    n = it.get("neg")
    if n is None:
        return None
    if n["kind"] == "str":
        fi, fo, si, so, q = (f"'{n['first_in']}'", f"'{n['first_out']}'",
                             f"'{n['snd_in']}'", f"'{n['snd_out']}'", f"'{n['query']}'")
        ans = f"'{n['first_out']}'"
    else:
        fi, fo, si, so, q = (n["first_in"], n["first_out"], n["snd_in"], n["snd_out"], n["query"])
        ans = f"{n['first_out']}"
    return make(tid+"-neg", "inductive", it["domain"], it["problem"],
        [(f"Look only at the first example, {fi} -> {fo}.", "invalid"),
         (f"Guess that the output is always {fo}, without checking the other pairs.", "invalid"),
         (f"So for {q}, answer {fo}.", "invalid")],
        ans, False, it["difficulty"], "procedural", "code_execution", False,
        f"Memorizes the first example and returns {fo} regardless of input; this already fails the "
        f"next shown pair {si} -> {so}. A rule checked against every pair would have been rejected.",
        "procedural-gen, verified", None, it["_created"],
        notes=f"paired positive: {tid}. Fallacy: overfitting/memorizing one example instead of the rule.")

PRB_FRAMINGS = [
    ("medicine-style diagnosis", lambda se,sp,pv: f"A medical test is {se}% sensitive and {sp}% specific for a condition with {pv}% prevalence.", "the person has the condition"),
    ("program behavior",         lambda se,sp,pv: f"A spam filter flags {se}% of spam and wrongly flags {100-sp}% of legitimate mail; {pv}% of incoming mail is spam.", "the message is spam"),
    ("mechanical / systems troubleshooting", lambda se,sp,pv: f"A scanner catches {se}% of defective parts and falsely flags {100-sp}% of good parts; {pv}% of parts are defective.", "the part is defective"),
    ("incident and root-cause analysis", lambda se,sp,pv: f"An intrusion detector alerts on {se}% of real attacks and false-alarms on {100-sp}% of normal sessions; {pv}% of sessions are attacks.", "it is a real attack"),
    ("science",                  lambda se,sp,pv: f"A field survey detects a species in {se}% of sites where it lives and gives a false positive in {100-sp}% of sites where it does not; the species is present at {pv}% of sites.", "the species is present"),
    ("finance and business operations", lambda se,sp,pv: f"A fraud screen flags {se}% of fraudulent charges and wrongly flags {100-sp}% of legitimate ones; {pv}% of charges are fraudulent.", "the charge is fraudulent"),
    ("biology and ecology",      lambda se,sp,pv: f"A genetic assay is positive in {se}% of animals carrying a trait and in {100-sp}% of those without it; {pv}% of the population carries the trait.", "the animal carries the trait"),
    ("law and regulation",       lambda se,sp,pv: f"A compliance audit flags {se}% of filings that truly breach a rule and {100-sp}% of compliant ones; {pv}% of filings actually breach it.", "the filing breaches the rule"),
    ("chemistry",                lambda se,sp,pv: f"A spectrometer detects a contaminant in {se}% of tainted batches and reads positive on {100-sp}% of clean ones; {pv}% of batches are tainted.", "the batch is contaminated"),
    ("engineering and physical systems", lambda se,sp,pv: f"A weld-inspection sensor catches {se}% of cracked joints and false-alarms on {100-sp}% of sound ones; {pv}% of joints are cracked.", "the joint is cracked"),
]
QSTEM_P = ["Given a positive result, what is the probability that {H}?",
           "A positive result comes back. How likely is it that {H}?",
           "After a positive result, give the posterior probability that {H} (to three decimals)."]
# Eval uses its OWN stem strings (never used in train), so the delexicalized
# problem signature of any eval item differs from every train item -> the
# (structural-signature, answer) leakage is zero by construction, on top of the
# reserved held-out parameter region below.
QSTEM_P_EVAL = ["Report the posterior probability that {H}, to three decimals, from a single positive result.",
                "One positive result is observed; state the posterior probability that {H} (three decimals)."]
PRB_PREVS_TRAIN = [round(x, 3) for x in [0.001,0.002,0.003,0.005,0.008,0.01,0.02,0.03,0.04,0.05,0.07,0.1,0.12,0.15,0.2,0.25,0.3]]
PRB_RATES_TRAIN = [0.7,0.72,0.75,0.8,0.82,0.85,0.88,0.9,0.92,0.95,0.97,0.99]
PRB_PREVS_EVAL = [0.004,0.006,0.009,0.015,0.018,0.06,0.09,0.11,0.18,0.28]     # disjoint from train
PRB_RATES_EVAL = [0.71,0.73,0.77,0.81,0.83,0.87,0.91,0.93,0.96,0.98]          # disjoint from train

def _prb_diff(prev):
    return 1 if prev >= 0.25 else 2 if prev >= 0.1 else 3 if prev >= 0.02 else 4

def _mk_prb_single(dom, ctx, H, prev, se, sp, stem, split):
    p = bayes(prev, se, sp); den = se*prev + (1-sp)*(1-prev)
    prob = ctx(int(round(se*100)), int(round(sp*100)), f"{prev*100:g}") + " " + stem.format(H=H)
    return dict(domain=dom, problem=prob, difficulty=_prb_diff(prev), confidence=0.95,
        vm="code_execution", split=split,
        steps=[(f"Prior = {prev:g}; complement = {1-prev:g}.", "valid"),
               (f"True-positive rate {se:g}; false-positive rate {round(1-sp,3):g}.", "valid"),
               (f"P(positive) = {se:g}*{prev:g} + {round(1-sp,3):g}*{1-prev:g} = {den:.5f}.", "valid"),
               (f"Posterior = {se*prev:.5f} / {den:.5f} = {p:.4f}.", "valid"),
               ("Check: consistent with Bayes' rule.", "valid")],
        final=f"{p:.3f}",
        vd=f"Computed ({se:g}*{prev:g})/({se:g}*{prev:g}+{round(1-sp,3):g}*{1-prev:g}) = {p:.5f}; matches {p:.3f}.",
        se=int(round(se*100)))

PRB_DOUBLE_TAIL_TRAIN = " The test is run twice independently on the same subject and BOTH results are positive. Give the posterior probability that {H}, to three decimals."
PRB_DOUBLE_TAIL_EVAL = " Two independent runs of the test on the same subject are BOTH positive. Report the posterior probability that {H} to three decimals."
def _mk_prb_double(dom, ctx, H, prev, se, sp, split):
    """Difficulty 5: two INDEPENDENT positive results -> update with se^2, (1-sp)^2."""
    num = se*se*prev; den = num + (1-sp)*(1-sp)*(1-prev); p = num/den
    tail = PRB_DOUBLE_TAIL_EVAL if split == "eval" else PRB_DOUBLE_TAIL_TRAIN
    prob = ctx(int(round(se*100)), int(round(sp*100)), f"{prev*100:g}") + tail.format(H=H)
    return dict(domain=dom, problem=prob, difficulty=5, confidence=0.9,
        vm="code_execution", split=split,
        steps=[(f"Prior = {prev:g}; complement = {1-prev:g}.", "valid"),
               (f"Two independent positives multiply the likelihoods: P(++|H) = {se:g}^2, P(++|not H) = {round(1-sp,3):g}^2.", "valid"),
               (f"P(++) = {se:g}^2*{prev:g} + {round(1-sp,3):g}^2*{1-prev:g} = {den:.6f}.", "valid"),
               (f"Posterior = {num:.6f} / {den:.6f} = {p:.4f}.", "valid"),
               ("Check: sequential Bayes with conditionally independent tests.", "valid")],
        final=f"{p:.3f}",
        vd=f"Two-test update: ({se:g}^2*{prev:g})/({se:g}^2*{prev:g}+{round(1-sp,3):g}^2*{1-prev:g}) = {p:.6f}; matches {p:.3f}.",
        se=int(round(se*100)))

def _probabilistic_eval(seen):
    out = []
    def add(it):
        if it and it["problem"] not in seen:
            seen.add(it["problem"]); out.append(it)
    for i, (dom, ctx, H) in enumerate(PRB_FRAMINGS):
        for j in range(9):                       # reserved params + eval-only stem
            prev = PRB_PREVS_EVAL[(i + j) % len(PRB_PREVS_EVAL)]
            se = PRB_RATES_EVAL[(i + 2 * j) % len(PRB_RATES_EVAL)]
            sp = PRB_RATES_EVAL[(i + 2 * j + 3) % len(PRB_RATES_EVAL)]
            add(_mk_prb_single(dom, ctx, H, prev, se, sp, QSTEM_P_EVAL[j % len(QSTEM_P_EVAL)], "eval"))
        for j in range(2):
            prev = PRB_PREVS_EVAL[(i + j) % len(PRB_PREVS_EVAL)]
            se = PRB_RATES_EVAL[(i + 1 + j) % len(PRB_RATES_EVAL)]; sp = PRB_RATES_EVAL[(i + 4 + j) % len(PRB_RATES_EVAL)]
            add(_mk_prb_double(dom, ctx, H, prev, se, sp, "eval"))
    return out

def build_probabilistic(need, rng, exclude):
    out, seen = [], set(exclude)
    for it in _probabilistic_eval(seen):
        out.append(it)
    made, attempts = 0, 0
    while made < need and attempts < need*40 + 3000:
        attempts += 1
        prev = rng.choice(PRB_PREVS_TRAIN); se = rng.choice(PRB_RATES_TRAIN); sp = rng.choice(PRB_RATES_TRAIN)
        dom, ctx, H = rng.choice(PRB_FRAMINGS)
        if rng.random() < 0.15:                  # difficulty-5 two-test family
            it = _mk_prb_double(dom, ctx, H, prev, se, sp, "train")
        else:
            it = _mk_prb_single(dom, ctx, H, prev, se, sp, rng.choice(QSTEM_P), "train")
        if it["problem"] in seen:
            continue
        seen.add(it["problem"]); out.append(it); made += 1
    return out

def neg_probabilistic(tid, it):
    se = it.get("se")
    if se is None: return None
    return make(tid+"-neg", "probabilistic", it["domain"], it["problem"],
        [(f"The true-positive rate is {se}%.", "valid"),
         (f"So a positive result means a {se}% chance of the hypothesis.", "invalid"),
         ("Ignore the base rate.", "invalid")],
        f"{se/100:.3f}", False, it["difficulty"], "procedural", "code_execution", False,
        f"Reports the sensitivity instead of the posterior and neglects the base rate; correct posterior is {it['final']}.",
        "procedural-gen, verified", None, it["_created"], confidence=0.95,
        notes=f"paired positive: {tid}. Fallacy: base-rate neglect.")

CFA_STEMS = ["what would {Y} have been?", "compute the resulting {Y}.", "what would {Y} be instead?"]
CFA_STEMS_EVAL = ["state the resulting {Y}.", "what would {Y} come to instead?"]
def _money(x):
    return str(int(x)) if x == int(x) else f"{x:.2f}"

# One-factor linear skins: baseline = p0*q, intervene p0->p1 (q held fixed).
# (domain, difficulty, model-label, setup format(p0,p1,q,b), Y-label)
CFA_LINEAR = [
    ("program behavior", 2, "volume = rate * time",
     "A tank fills at {p0} L/min for {q} min, reaching {b} L. If the rate had been {p1} L/min for the same {q} min,", "the volume"),
    ("engineering and physical systems", 2, "area = length * width",
     "A plot is {q} by {p0} (area {b}). If the width were {p1} with the same length {q},", "the area"),
    ("science", 2, "distance = speed * time",
     "A train goes {p0} km/h for {q} h, covering {b} km. At {p1} km/h for the same {q} h,", "the distance"),
    ("medicine-style diagnosis", 2, "dose = rate * weight",
     "A drug is dosed at {q} mg/kg; a {p0} kg patient received {b} mg. For a {p1} kg patient at the same rate,", "the dose"),
    ("engineering and physical systems", 2, "energy = power * hours",
     "A heater at {p0} kW ran {q} h, using {b} kWh. At {p1} kW for the same {q} h,", "the energy"),
    ("engineering and physical systems", 3, "voltage = current * resistance",
     "A resistor of {p0} ohms carries {q} A, dropping {b} V. If it were {p1} ohms at the same {q} A,", "the voltage"),
    ("program behavior", 2, "total = rate * seconds",
     "A service handling {p0} requests/s for {q} s served {b} requests. At {p1} requests/s for the same {q} s,", "the total served"),
    ("biology and ecology", 2, "harvest = plots * per-plot",
     "A farm with {p0} plots yielding {q} kg each harvested {b} kg. With {p1} plots at the same {q} kg each,", "the harvest"),
    ("economics and markets", 2, "pay = hours * rate",
     "A worker paid {q} per hour for {p0} hours earned {b}. For {p1} hours at the same {q} per hour,", "the pay"),
    ("everyday planning", 2, "flour = servings * per-serving",
     "A recipe for {p0} servings uses {q} cups per serving, {b} cups in all. For {p1} servings at the same {q} cups each,", "the flour"),
]
def _mk_cfa_linear(skin, p0, p1, q, stem, split):
    dom, diff, model, setup_fmt, Y = skin
    b, c = p0 * q, p1 * q
    prob = setup_fmt.format(p0=p0, p1=p1, q=q, b=b) + " " + stem.format(Y=Y)
    return dict(domain=dom, problem=prob, difficulty=diff, vm="code_execution", split=split, base=b,
        steps=[(f"Model: {model}.", "valid"), (f"Baseline: {p0}*{q} = {b}.", "valid"),
               (f"Intervene: set the varied factor to {p1}, holding the other at {q}.", "valid"),
               (f"Run the model: {p1}*{q} = {c}.", "valid"),
               (f"Check: {c} differs from the factual baseline {b}, as an intervention should.", "valid")],
        final=str(c), vd=f"{model}: intervened factor {p0}->{p1}, other={q} => {c}.")

# Percentage / money skins (difficulty 3): baseline uses rate r0, intervene to r1.
CFA_PCT = [
    ("finance and business operations", "interest = principal * rate * years", "interest",
     "A deposit of {P} earns simple interest at {pr0}% for {y} years, yielding {b}. At {pr1}% for the same {y} years,"),
    ("finance and business operations", "tax = income * rate", "tax",
     "Income of {P} taxed at {pr0}% owes {b}. At a {pr1}% rate on the same income,"),
]
def _mk_cfa_pct(skin, P, r0, r1, y, stem, split):
    dom, model, Y, setup_fmt = skin
    b = round(P * r0 * y, 2); c = round(P * r1 * y, 2)
    prob = setup_fmt.format(P=P, pr0=round(r0*100, 4), pr1=round(r1*100, 4), y=y, b=_money(b)) + " " + stem.format(Y="the " + Y)
    return dict(domain=dom, problem=prob, difficulty=3, vm="code_execution", split=split, base=_money(b),
        steps=[(f"Model: {model}.", "valid"), (f"Baseline: {P}*{r0:g}*{y} = {_money(b)}.", "valid"),
               (f"Intervene: change the rate to {r1:g}, holding principal and term.", "valid"),
               (f"Run: {P}*{r1:g}*{y} = {_money(c)}.", "valid"),
               (f"Check: the factual baseline {_money(b)} cannot answer the intervention.", "valid")],
        final=_money(c), vd=f"{model}: rate {r0:g}->{r1:g} => {_money(c)}.")

def _mk_cfa_simple(n0, n1, v, stem, split):     # difficulty 1
    b, c = n0 * v, n1 * v
    prob = (f"A shelf holds {n0} boxes of {v} items each, {b} items in all. "
            f"If it held {n1} boxes of {v} items each instead, " + stem.format(Y="the item count"))
    return dict(domain="everyday planning", problem=prob, difficulty=1, vm="code_execution", split=split, base=b,
        steps=[("Model: items = boxes * per-box.", "valid"), (f"Baseline: {n0}*{v} = {b}.", "valid"),
               (f"Intervene: boxes = {n1}, per-box unchanged at {v}.", "valid"),
               (f"Run: {n1}*{v} = {c}.", "valid"), (f"Check: {c} != baseline {b}.", "valid")],
        final=str(c), vd=f"items=boxes*per-box: {n1}*{v} = {c}.")

def _mk_cfa_two(s0, s1, t0, t1, stem, split):   # difficulty 4: two simultaneous interventions
    b, c = s0 * t0, s1 * t1
    prob = (f"A pump moves water at {s0} L/min for {t0} min, moving {b} L. "
            f"If BOTH the rate had been {s1} L/min AND it had run for {t1} min, " + stem.format(Y="the volume moved"))
    return dict(domain="engineering and physical systems", problem=prob, difficulty=4,
        vm="code_execution", split=split, base=b,
        steps=[("Model: volume = rate * time.", "valid"), (f"Baseline: {s0}*{t0} = {b}.", "valid"),
               (f"Intervene on BOTH factors: rate {s0}->{s1} and time {t0}->{t1}.", "valid"),
               (f"Run with both changes: {s1}*{t1} = {c}.", "valid"),
               (f"Check: applying only one change would give {s1*t0} or {s0*t1}; both together give {c}.", "valid")],
        final=str(c), vd=f"two interventions: {s1}*{t1} = {c}; single-change values {s1*t0}/{s0*t1} rejected.")

def _mk_cfa_compound(P, r0, r1, n, stem, split):    # difficulty 5: nonlinear compounding
    b = round(P * (1 + r0) ** n, 2); c = round(P * (1 + r1) ** n, 2)
    prob = (f"{P} is invested at {round(r0*100,4):g}% compounded annually for {n} years, growing to {_money(b)}. "
            f"If the annual rate had been {round(r1*100,4):g}% over the same {n} years, " + stem.format(Y="the final balance"))
    return dict(domain="finance and business operations", problem=prob, difficulty=5,
        vm="code_execution", split=split, base=_money(b),
        steps=[("Model: balance = principal * (1 + rate)^years (compound, not linear).", "valid"),
               (f"Baseline: {P}*(1+{r0:g})^{n} = {_money(b)}.", "valid"),
               (f"Intervene: rate {r0:g}->{r1:g}, term held at {n} years.", "valid"),
               (f"Run: {P}*(1+{r1:g})^{n} = {_money(c)}.", "valid"),
               (f"Check: compounding is nonlinear, so a linear scaling of the baseline is wrong; recomputed to {_money(c)}.", "valid")],
        final=_money(c), vd=f"compound: {P}*(1+{r1:g})^{n} = {_money(c)}.")

def _mk_cfa_chain(units, pr0, pr1, tax, stem, split):   # difficulty 5: two-stage chain
    b = round(units * pr0 * (1 - tax), 2); c = round(units * pr1 * (1 - tax), 2)
    prob = (f"Selling {units} units at {pr0} each and then paying {int(round(tax*100))}% tax nets {_money(b)}. "
            f"If the price had been {pr1} each, same units and tax, " + stem.format(Y="the after-tax revenue"))
    return dict(domain="finance and business operations", problem=prob, difficulty=5,
        vm="code_execution", split=split, base=_money(b),
        steps=[("Model (two stages): revenue = units*price, then after-tax = revenue*(1-tax).", "valid"),
               (f"Baseline: {units}*{pr0}*(1-{tax:g}) = {_money(b)}.", "valid"),
               (f"Intervene: price {pr0}->{pr1}; propagate through both stages.", "valid"),
               (f"Run: {units}*{pr1}*(1-{tax:g}) = {_money(c)}.", "valid"),
               (f"Check: the intervention flows through revenue into after-tax revenue; recomputed to {_money(c)}.", "valid")],
        final=_money(c), vd=f"two-stage: {units}*{pr1}*(1-{tax:g}) = {_money(c)}.")

def _counterfactual_eval(seen):
    """Deterministic held-out eval: reserved parameter region (linear factors
    >= 26, disjoint from train) + eval-only stems -> disjoint answers & signatures."""
    out = []
    def add(it):
        if it and it["problem"] not in seen:
            seen.add(it["problem"]); out.append(it)
    stems = CFA_STEMS_EVAL
    for i, sk in enumerate(CFA_LINEAR):
        for j in range(6):
            p0 = 31 + ((i + j) % 9); p1 = 41 + ((i + 2 * j) % 9); q = 26 + ((i + j) % 6)
            add(_mk_cfa_linear(sk, p0, p1, q, stems[j % len(stems)], "eval"))
    for i, sk in enumerate(CFA_PCT):
        for j, (P, r0, r1, y) in enumerate([(7000, 0.011, 0.061, 6), (8000, 0.013, 0.071, 7),
                                            (9000, 0.017, 0.081, 8), (7500, 0.019, 0.091, 9),
                                            (8800, 0.014, 0.066, 6), (9600, 0.016, 0.076, 7)]):
            add(_mk_cfa_pct(sk, P, r0, r1, y, stems[j % len(stems)], "eval"))
    for j, (n0, n1, v) in enumerate([(31, 41, 26), (33, 44, 27), (35, 47, 28), (37, 49, 29), (39, 46, 31), (32, 48, 30)]):
        add(_mk_cfa_simple(n0, n1, v, stems[j % len(stems)], "eval"))
    for j, (s0, s1, t0, t1) in enumerate([(31, 41, 26, 33), (34, 45, 27, 31), (37, 48, 29, 35),
                                          (33, 47, 28, 34), (39, 44, 30, 36), (36, 49, 31, 32)]):
        add(_mk_cfa_two(s0, s1, t0, t1, stems[j % len(stems)], "eval"))
    for j, (P, r0, r1, n) in enumerate([(7000, 0.031, 0.081, 9), (8500, 0.041, 0.091, 11),
                                        (7700, 0.036, 0.086, 10), (9200, 0.046, 0.096, 12)]):
        add(_mk_cfa_compound(P, r0, r1, n, stems[j % len(stems)], "eval"))
    for j, (u, p0, p1, tx) in enumerate([(310, 41, 61, 0.19), (330, 44, 64, 0.23),
                                         (350, 47, 67, 0.17), (370, 49, 69, 0.21)]):
        add(_mk_cfa_chain(u, p0, p1, tx, stems[j % len(stems)], "eval"))
    return out

def build_counterfactual(need, rng, exclude):
    out, seen = [], set(exclude)
    for it in _counterfactual_eval(seen):
        out.append(it)
    made, attempts = 0, 0
    fams = (["linear"] * 10 + ["pct"] * 3 + ["simple"] * 2 + ["two"] * 3 + ["compound"] * 2 + ["chain"] * 2)
    while made < need and attempts < need * 60 + 4000:
        attempts += 1
        fam = rng.choice(fams); stem = rng.choice(CFA_STEMS)
        if fam == "linear":
            sk = rng.choice(CFA_LINEAR)
            p0 = rng.randint(2, 25); p1 = rng.randint(2, 25); q = rng.randint(2, 25)
            if p1 == p0: p1 = p0 + 1
            it = _mk_cfa_linear(sk, p0, p1, q, stem, "train")
        elif fam == "pct":
            sk = rng.choice(CFA_PCT); P = rng.choice([1000, 1500, 2000, 3000, 4000, 5000, 6000])
            r0 = rng.choice([0.02, 0.03, 0.04, 0.05]); r1 = rng.choice([0.06, 0.07, 0.08, 0.1, 0.12])
            it = _mk_cfa_pct(sk, P, r0, r1, rng.randint(2, 9), stem, "train")
        elif fam == "simple":
            n0 = rng.randint(2, 12); n1 = rng.randint(2, 15); v = rng.randint(2, 12)
            if n1 == n0: n1 = n0 + 1
            it = _mk_cfa_simple(n0, n1, v, stem, "train")
        elif fam == "two":
            s0 = rng.randint(2, 20); s1 = rng.randint(2, 20); t0 = rng.randint(2, 20); t1 = rng.randint(2, 20)
            if s1 == s0: s1 = s0 + 1
            if t1 == t0: t1 = t0 + 1
            it = _mk_cfa_two(s0, s1, t0, t1, stem, "train")
        elif fam == "compound":
            P = rng.choice([1000, 2000, 3000, 5000]); r0 = rng.choice([0.02, 0.03, 0.04]); r1 = rng.choice([0.05, 0.06, 0.08])
            it = _mk_cfa_compound(P, r0, r1, rng.randint(3, 12), stem, "train")
        else:
            u = rng.randint(20, 200); p0 = rng.randint(5, 40); p1 = rng.randint(5, 40)
            if p1 == p0: p1 = p0 + 1
            it = _mk_cfa_chain(u, p0, p1, rng.choice([0.1, 0.15, 0.2, 0.25]), stem, "train")
        if it["problem"] in seen:
            continue
        seen.add(it["problem"]); out.append(it); made += 1
    return out

def neg_counterfactual(tid, it):
    base = it.get("base")
    if base is None: return None
    return make(tid+"-neg", "counterfactual", it["domain"], it["problem"],
        [("Model identified.", "valid"), (f"The factual outcome was {base}.", "valid"),
         (f"So the answer would be {base}.", "invalid")],
        str(base), False, it["difficulty"], "procedural", "code_execution", False,
        f"Reports the factual baseline {base} and ignores the intervention; correct counterfactual is {it['final']}.",
        "procedural-gen, verified", None, it["_created"],
        notes=f"paired positive: {tid}. Fallacy: answering from the factual baseline.")

# Causal: parametric families whose ANSWER is determined by the planted graph
# and the numbers (partial correlation, effect decomposition, trial size), so
# each item does real causal reasoning and scales freely with the seed. The
# verification ceiling for causal is answer-match against a known causal graph
# (DOMAINS.md), which is what these encode.
CAU_Q_TRAIN = ["Does the first quantity cause the second on this evidence, and what would settle it?",
               "Is the causal claim warranted here? Explain.",
               "Assess whether this establishes causation, and how you would confirm it."]
CAU_Q_EVAL = ["Does this evidence establish that the first causes the second?",
              "State whether the causal claim holds and what would settle it."]
# X, Y independent causes of a collider C (conditioning on C induces spurious correlation).
CAU_COLLIDER = [
    ("athletic talent", "academic ability", "admission to a selective scholarship program", "social situations"),
    ("acting skill", "physical attractiveness", "being cast as a working actor", "social situations"),
    ("code quality", "marketing spend", "a startup getting acquired", "finance and business operations"),
    ("engine power", "fuel efficiency", "a car passing a strict certification", "engineering and physical systems"),
    ("kindness", "competence", "getting hired after interviews", "social situations"),
    ("symptom severity", "test positivity", "being admitted to the study ward", "medicine-style diagnosis"),
    ("manuscript novelty", "writing polish", "a paper clearing peer review", "narrative and discourse"),
    ("soil richness", "rainfall", "a plot being chosen for the trial", "biology and ecology"),
]
# Treatment X lowers outcome Y within every subgroup, but the pooled data reverse (Simpson).
CAU_SIMPSON = [
    ("a new treatment", "the recovery rate", "case severity", "medicine-style diagnosis"),
    ("an ad campaign", "the conversion rate", "customer segment", "economics and markets"),
    ("a study method", "the pass rate", "prior preparation", "science"),
    ("a code-review policy", "the defect rate", "module complexity", "program behavior"),
    ("a fertilizer", "the yield", "soil type", "biology and ecology"),
    ("a staffing change", "the resolution rate", "ticket difficulty", "incident and root-cause analysis"),
]
def _cau_triples(drivers):
    out = []
    for z, (acts, dom) in drivers.items():
        for x, y in combinations(acts, 2):
            out.append((x, y, z, dom))
    return out

def _mk_cau_confound(x, y, z, dom, r, pr, q, split):
    confounded = pr < 0.15
    if confounded:
        final = (f"Not supported: controlling for {z} collapses the association (r = {r:g} -> partial r = {pr:g}, "
                 f"near zero), so {z} is a confounder and the evidence does not show that {x} causes {y}.")
        last = "Near zero: the raw correlation was almost entirely the common cause, so no direct effect is established."
        verdict = "not supported (confounded)"
    else:
        final = (f"Supported as a direct effect: the association largely survives controlling for {z} "
                 f"(r = {r:g} -> partial r = {pr:g}), so beyond the shared driver there is a direct {x} -> {y} link.")
        last = "Still substantial: part of the association remains after removing the common cause, indicating a direct effect."
        verdict = "supported (direct effect persists)"
    prob = (f"Across many observations, {x} and {y} are correlated (r = {r:g}); both also track {z}. "
            f"After statistically controlling for {z}, the partial correlation between them is {pr:g}. " + q)
    return dict(domain=dom, problem=prob, difficulty=(3 if (pr < 0.08 or pr >= 0.45) else 4),
        vm="answer_match", split=split, is_conf=confounded,
        steps=[(f"Observed: {x} and {y} correlate at r = {r:g}; both move with {z}.", "valid"),
               (f"{cap(z)} is a candidate common cause (confounder) of both.", "valid"),
               (f"Control for {z}: the partial correlation between {x} and {y} is {pr:g}.", "valid"),
               (last, "valid"),
               (f"Check: the verdict follows from whether the association survives conditioning on {z} -- {verdict}.", "valid")],
        final=final, vd=f"Common cause {z}; partial r={pr:g} => {verdict}.")

def _mk_cau_mediation(cause, eff, med, dom, te, de, q, split):
    ind = round(te - de, 2); full = de < 0.08
    if full:
        final = (f"Yes, but entirely indirectly: the total effect ({te:g}) of {cause} on {eff} runs through {med} "
                 f"(direct effect ~{de:g}); blocking {med} would remove it.")
        last = f"Direct part is ~0, so the effect is fully mediated by {med}."
    else:
        final = (f"Yes, partly directly and partly through {med}: total effect {te:g} = direct {de:g} + indirect {ind:g} via {med}.")
        last = f"Both paths carry effect: a direct {cause} -> {eff} arrow plus the path through {med}."
    prob = (f"A study finds {cause} is associated with {eff} (total effect {te:g}). Holding {med} fixed, the "
            f"direct effect of {cause} on {eff} is {de:g}; {cause} also changes {med}, which changes {eff}. " + q)
    return dict(domain=dom, problem=prob, difficulty=(3 if full else 4), vm="answer_match", split=split, is_conf=False,
        steps=[(f"Graph: {cause} -> {med} -> {eff}, possibly plus a direct {cause} -> {eff}.", "valid"),
               (f"Total effect measured: {te:g}.", "valid"),
               (f"Direct effect (holding {med} fixed): {de:g}; indirect via {med} = {te:g} - {de:g} = {ind:g}.", "valid"),
               (last, "valid"), ("Check: total = direct + indirect decomposition of the causal effect.", "valid")],
        final=final, vd=f"Effect decomposition: total {te:g} = direct {de:g} + indirect {ind:g} via {med}.")

def _mk_cau_rct(cause, eff, dom, delta, n, q, split):
    prob = (f"A randomized controlled trial with {n} subjects varies {cause} alone (random assignment) and finds {eff} "
            f"shifts by {delta:g} in the treated group. " + q)
    return dict(domain=dom, problem=prob, difficulty=2, vm="answer_match", split=split, is_conf=False,
        steps=[(f"Design: randomized assignment of {cause}, so treated and control groups differ only in {cause}.", "valid"),
               ("Randomization breaks any back-door path, ruling out confounders in expectation.", "valid"),
               (f"{cap(eff)} responds by {delta:g} under the manipulation.", "valid"),
               ("Check: a responsive randomized intervention isolates the cause.", "valid")],
        final=f"Supported: the randomized trial (n = {n}) isolates {cause}, and {eff} responds by {delta:g}, so the causal claim holds.",
        vd=f"RCT n={n}, effect {delta:g}; randomization rules out confounders.")

def _mk_cau_toggle(cause, eff, dom, k, q, split):
    prob = (f"Over {k} separate trials, {cause} is switched on and off while nothing else is changed, and {eff} follows "
            f"every single time. " + q)
    return dict(domain=dom, problem=prob, difficulty=1, vm="answer_match", split=split, is_conf=False,
        steps=[(f"Each trial is a controlled manipulation of {cause} with all else held fixed.", "valid"),
               (f"{cap(eff)} tracks the manipulation across all {k} trials with no exceptions.", "valid"),
               ("Repeated responsive manipulation with nothing else varying establishes the cause.", "valid"),
               ("Check: this is intervention, not mere correlation.", "valid")],
        final=f"Yes: manipulating {cause} toggles {eff} every time across {k} controlled trials, so {cause} causes {eff}.",
        vd=f"{k} controlled on/off manipulations; effect every time.")

def _mk_cau_collider(x, y, c, dom, rc, q, split):
    prob = (f"In the general population {x} and {y} are independent. But among cases selected by {c}, they show a "
            f"correlation of r = {rc:g}. Someone concludes {x} affects {y}. " + q)
    return dict(domain=dom, problem=prob, difficulty=5, vm="answer_match", split=split, is_conf=False,
        steps=[(f"{cap(c)} is a common EFFECT of both {x} and {y} (a collider), not a cause.", "valid"),
               (f"Conditioning on {c} (selecting on it) opens a spurious path between {x} and {y}.", "valid"),
               (f"So the r = {rc:g} appears only inside the {c}-selected sample; in the population they are independent.", "valid"),
               ("Check: the association is collider (selection) bias, not a causal link.", "valid")],
        final=(f"Not causal: {x} and {y} are independent in the population; the r = {rc:g} is collider bias from "
               f"selecting on {c} (a common effect), so it does not show {x} affects {y}."),
        vd=f"Collider {c}; conditioning induces spurious r={rc:g}; no causal link.")

def _mk_cau_simpson(tr, out_, sub, dom, lo, hi, q, split):
    prob = (f"In every subgroup of {sub}, {tr} is associated with a LOWER {out_} (by about {lo:g} points). Yet in the "
            f"pooled data, {tr} shows a HIGHER {out_} (by about {hi:g} points), because {sub} is distributed unevenly "
            f"across the groups. " + q)
    return dict(domain=dom, problem=prob, difficulty=5, vm="answer_match", split=split, is_conf=False,
        steps=[(f"Within every {sub} subgroup, {tr} lowers {out_} by ~{lo:g}: a consistent within-group effect.", "valid"),
               (f"The pooled +{hi:g} reversal comes from unequal {sub} mix between treated and untreated (a confounder).", "valid"),
               (f"{cap(sub)} confounds the pooled comparison; the subgroup (adjusted) effect is the causal one.", "valid"),
               ("Check: Simpson's paradox -- trust the stratified estimate, not the aggregate.", "valid")],
        final=(f"The causal effect is the within-subgroup one: {tr} lowers {out_}. The pooled increase is Simpson's "
               f"paradox from an uneven {sub} mix, not a real positive effect."),
        vd=f"Simpson's paradox: within-{sub} effect negative, pooled positive from confounding by {sub}.")

CAU_R_TRAIN = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
CAU_PR_TRAIN = [0.02, 0.05, 0.08, 0.1, 0.12, 0.35, 0.45, 0.55]
CAU_R_EVAL = [0.58, 0.63, 0.72, 0.83, 0.88]
CAU_PR_EVAL = [0.03, 0.06, 0.11, 0.4, 0.5]
def _causal_eval(seen):
    """Deterministic held-out eval: reserved driver/entity banks + reserved
    numeric params + eval-only question stems -> disjoint answers & signatures."""
    out = []
    def add(it):
        if it and it["problem"] not in seen:
            seen.add(it["problem"]); out.append(it)
    ev_triples = _cau_triples({k: v for k, v in list(CAUSAL_DRIVERS.items())[-3:]})
    for i, (x, y, z, dom) in enumerate(ev_triples[:40]):
        for j in range(2):
            r = CAU_R_EVAL[(i + j) % len(CAU_R_EVAL)]; pr = CAU_PR_EVAL[(i + 2 * j) % len(CAU_PR_EVAL)]
            add(_mk_cau_confound(x, y, z, dom, r, pr, CAU_Q_EVAL[(i + j) % 2], "eval"))
    for i, (cause, eff, med, dom) in enumerate(CAUSAL_MED[-6:]):
        for te, de in [(0.5, 0.0), (0.6, 0.3), (0.8, 0.0)]:
            add(_mk_cau_mediation(cause, eff, med, dom, te, de, CAU_Q_EVAL[i % 2], "eval"))
    for i, (cause, eff, dom) in enumerate(CAUSAL_DIRECT[-6:]):
        add(_mk_cau_rct(cause, eff, dom, [0.35, 0.5, 0.4, 0.6, 0.45, 0.55][i % 6], [220, 340, 480, 610, 290, 520][i % 6], CAU_Q_EVAL[i % 2], "eval"))
        add(_mk_cau_toggle(cause, eff, dom, [12, 15, 18, 21, 14, 24][i % 6], CAU_Q_EVAL[(i + 1) % 2], "eval"))
    for i, (x, y, c, dom) in enumerate(CAU_COLLIDER[-3:]):
        for rc in [0.42, 0.55]:
            add(_mk_cau_collider(x, y, c, dom, rc, CAU_Q_EVAL[i % 2], "eval"))
    for i, (tr, o, sub, dom) in enumerate(CAU_SIMPSON[-3:]):
        for lo, hi in [(6, 4), (8, 5)]:
            add(_mk_cau_simpson(tr, o, sub, dom, lo, hi, CAU_Q_EVAL[i % 2], "eval"))
    return out

def build_causal(need, rng, exclude):
    out, seen = [], set(exclude)
    for it in _causal_eval(seen):
        out.append(it)
    train_drivers = {k: v for k, v in list(CAUSAL_DRIVERS.items())[:-3]}
    conf_triples = _cau_triples(train_drivers)
    med_train = CAUSAL_MED[:-6]; direct_train = CAUSAL_DIRECT[:-6]
    coll_train = CAU_COLLIDER[:-3]; simp_train = CAU_SIMPSON[:-3]
    made, attempts = 0, 0
    fams = (["confound"] * 6 + ["mediation"] * 3 + ["rct"] * 2 + ["toggle"] * 2 + ["collider"] * 2 + ["simpson"] * 2)
    while made < need and attempts < need * 60 + 5000:
        attempts += 1
        fam = rng.choice(fams); q = rng.choice(CAU_Q_TRAIN)
        if fam == "confound":
            x, y, z, dom = rng.choice(conf_triples)
            it = _mk_cau_confound(x, y, z, dom, rng.choice(CAU_R_TRAIN), rng.choice(CAU_PR_TRAIN), q, "train")
        elif fam == "mediation":
            cause, eff, med, dom = rng.choice(med_train)
            te = rng.choice([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]); de = rng.choice([0.0, 0.05, 0.2, 0.3, 0.4])
            if de >= te: de = 0.0
            it = _mk_cau_mediation(cause, eff, med, dom, te, de, q, "train")
        elif fam == "rct":
            cause, eff, dom = rng.choice(direct_train)
            it = _mk_cau_rct(cause, eff, dom, rng.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), rng.choice(range(120, 900, 20)), q, "train")
        elif fam == "toggle":
            cause, eff, dom = rng.choice(direct_train)
            it = _mk_cau_toggle(cause, eff, dom, rng.randint(6, 40), q, "train")
        elif fam == "collider":
            x, y, c, dom = rng.choice(coll_train)
            it = _mk_cau_collider(x, y, c, dom, rng.choice([0.3, 0.35, 0.45, 0.5, 0.6]), q, "train")
        else:
            tr, o, sub, dom = rng.choice(simp_train)
            it = _mk_cau_simpson(tr, o, sub, dom, rng.randint(4, 12), rng.randint(3, 9), q, "train")
        if it["problem"] in seen:
            continue
        seen.add(it["problem"]); out.append(it); made += 1
    return out

def neg_causal(tid, it):
    if not it.get("is_conf"): return None
    return make(tid+"-neg", "causal", it["domain"], it["problem"],
        [("The two variables move together.", "valid"),
         ("Because they move together, one must cause the other.", "invalid"),
         ("Conclude the causal claim is supported.", "invalid")],
        "Yes, the causal claim is supported.", False, it["difficulty"], "procedural", "answer_match", False,
        "A common cause already explains the correlation, so concluding causation from co-movement alone is unjustified; the claim is not supported by this evidence.",
        "procedural-gen, verified", None, it["_created"],
        notes=f"paired positive: {tid}. Fallacy: correlation mistaken for causation.")

Q_MET = ["Audit this reasoning and correct it: '{T}'",
         "Find the first error in this argument, or say it is valid: '{T}'",
         "Is this argument valid? If not, identify the flaw: '{T}'",
         "A student wrote: '{T}'. Grade it and fix any mistake."]
# Metacognitive: audit a short argument, find the planted error (or certify it
# clean). Family builders take explicit params + a stem index so they are
# deterministic; train draws params by rng from the TRAIN region, eval
# enumerates a fixed set from a disjoint EVAL region -> answers are disjoint and
# eval is stable across seasons. Every "clean" family's answer includes its own
# numbers, so the correct-case answer is not a constant string (no leakage).
def _met_arith(a, b, qi, split):    # difficulty 1: certify a correct addition
    c = a + b
    prob = Q_MET[qi].format(T=f"{a} + {b} = {c}.")
    return dict(domain="mathematics", problem=prob, difficulty=1, vm="process_check", split=split,
        steps=[(f"Add: {a} + {b} = {c}.", "valid"), (f"The stated sum is {c}; it matches.", "valid"),
               ("No error is present.", "valid")],
        final=f"No errors. {a} + {b} = {c} is correct.", vd="Certified a correct one-step addition.")

def _met_half(N, qi, split):        # difficulty 2: arithmetic slip
    half = N // 2; slip = half - 2; qq = N // 4
    prob = Q_MET[qi].format(T=f"Half of {N} is {half}, and half of {half} is {slip}, so a quarter of {N} is {slip}.")
    return dict(domain="mathematics", problem=prob, difficulty=2, vm="process_check", split=split,
        steps=[(f"Half of {N} is {half}. Correct.", "valid"),
               (f"Half of {half} is {half//2}, not {slip}: an arithmetic slip.", "valid"),
               (f"Correction: a quarter of {N} is {qq}.", "valid")],
        final=f"Error: half of {half} is {half//2}, not {slip}. A quarter of {N} is {qq}.",
        vd="Planted arithmetic slip; recomputed.")

def _met_affirm(dom, subj, preds, qi, split):   # difficulty scales with chain length
    L = len(preds) - 1
    T = "Rules: " + " ".join(f"If the {subj} is {preds[i]}, then it is {preds[i+1]}." for i in range(L)) + \
        f" Observed: the {subj} is {preds[-1]}. Therefore it is {preds[0]}."
    prob = Q_MET[qi].format(T=T)
    return dict(domain=dom, problem=prob, difficulty=min(1 + L, 5), vm="process_check", split=split,
        steps=[(f"The forward chain '{preds[0]}' -> ... -> '{preds[-1]}' ({L} steps) is fine.", "valid"),
               (f"But it then infers '{preds[0]}' from '{preds[-1]}': affirming the consequent.", "valid"),
               (f"'{preds[-1]}' can hold for other reasons, so it does not entail '{preds[0]}'.", "valid"),
               (f"Correction: '{preds[0]}' is not justified by the observation.", "valid")],
        final=f"Error: affirming the consequent. '{preds[-1]}' does not entail '{preds[0]}'.",
        vd=f"Planted error: affirming the consequent across a length-{L} chain.")

def _met_root(k, qi, split):        # difficulty 3: skipped negative root
    sq = k * k
    prob = Q_MET[qi].format(T=f"x^2 = {sq}, so x = {k}.")
    return dict(domain="mathematics", problem=prob, difficulty=3, vm="process_check", split=split,
        steps=[(f"x = {k} satisfies x^2 = {sq}.", "valid"), ("It skips the negative root.", "valid"),
               (f"x = -{k} also works.", "valid")],
        final=f"Error: a skipped case. x = {k} or x = -{k}.", vd="Planted skipped negative root.")

def _met_clean_div(d, mult, qi, split):     # difficulty 3: valid, no error
    n = d * mult
    prob = Q_MET[qi].format(T=f"If a number is divisible by {d} then it is even, since {d} is even. {n} is divisible by {d}, so {n} is even.")
    return dict(domain="mathematics", problem=prob, difficulty=3, vm="process_check", split=split,
        steps=[(f"Divisible by {d} implies even, since {d} is even. Valid.", "valid"),
               (f"{n} is divisible by {d}.", "valid"), (f"So {n} is even. Valid.", "valid"),
               ("No error is present.", "valid")],
        final=f"No errors. The reasoning is valid: {n} is even.", vd="Certified a clean divisibility argument.")

def _met_clean_odd(nn, qi, split):  # difficulty 3: valid, no error (answer varies with nn)
    s = nn * nn; odds = ", ".join(str(2 * i + 1) for i in range(nn))
    prob = Q_MET[qi].format(T=f"The sum of the first n odd numbers is n^2. The first {nn} odd numbers are {odds}, summing to {s} = {nn}^2.")
    return dict(domain="mathematics", problem=prob, difficulty=3, vm="process_check", split=split,
        steps=[(f"{odds.replace(', ', ' + ')} = {s}.", "valid"), (f"{s} = {nn}^2.", "valid"),
               (f"Matches the identity for n = {nn}.", "valid"), ("No error is present.", "valid")],
        final=f"No errors. The first {nn} odd numbers sum to {s} = {nn}^2, as claimed.",
        vd="Certified a clean sum-of-odds identity instance.")

def _met_offby(n, qi, split):       # difficulty 4: off-by in a triangular sum
    correct = n * (n + 1) // 2; wrong = correct + n
    prob = Q_MET[qi].format(T=f"The sum 1 + 2 + ... + {n} equals n(n+1)/2. For n = {n} that gives {wrong}.")
    return dict(domain="mathematics", problem=prob, difficulty=4, vm="process_check", split=split,
        steps=[("The formula n(n+1)/2 for the sum 1..n is correct.", "valid"),
               (f"But n(n+1)/2 at n = {n} is {n}*{n+1}/2 = {correct}, not {wrong}.", "valid"),
               (f"The arithmetic was off by {n}.", "valid"), (f"Correction: the sum is {correct}.", "valid")],
        final=f"Error: n(n+1)/2 at n = {n} is {correct}, not {wrong}.",
        vd=f"Planted evaluation error; recomputed {n}(n+1)/2 = {correct}.")

def _met_false_identity(n, qi, split):      # difficulty 5: a plausible but WRONG general formula
    correct = n * (n + 1) // 2; claimed = (n * n) // 2
    prob = Q_MET[qi].format(T=f"Claim: the sum of the first n positive integers equals n^2/2. Check at n = {n}: n^2/2 = {claimed}.")
    return dict(domain="mathematics", problem=prob, difficulty=5, vm="process_check", split=split,
        steps=[("The proposed identity is sum(1..n) = n^2/2.", "valid"),
               (f"The correct closed form is n(n+1)/2, which at n = {n} is {correct}.", "valid"),
               (f"The claim gives n^2/2 = {claimed} at n = {n}, differing from {correct}: the formula is wrong (it drops the +n/2 term).", "valid"),
               ("The error is a wrong general identity, not a slip; it fails for every n > 0.", "valid")],
        final=f"Error: the formula is wrong. Sum of 1..{n} is {correct} (n(n+1)/2), not n^2/2 = {claimed}.",
        vd=f"Refuted a false general identity; correct sum n(n+1)/2 = {correct} vs claimed {claimed}.")

def _metacognitive_eval(seen):
    """Deterministic held-out eval over reserved parameter regions."""
    out = []
    def add(it):
        if it and it["problem"] not in seen:
            seen.add(it["problem"]); out.append(it)
    for i, (a, b) in enumerate([(41, 58), (47, 66), (53, 74), (61, 89), (72, 95)]):
        add(_met_arith(a, b, i % len(Q_MET), "eval"))
    for i, N in enumerate(range(604, 900, 28)):        # reserved half range
        add(_met_half(N, i % len(Q_MET), "eval"))
    ev_doms = [("logic puzzles", "glyph"), ("program behavior", "packet"), ("chemistry", "reagent")]
    ring = PRED_EVAL + PRED_EVAL
    idx = 0
    for L in (1, 2, 3, 4):
        for j in range(3):
            dom, subj = ev_doms[j % len(ev_doms)]
            preds = ring[idx:idx + L + 1]; idx += 1
            add(_met_affirm(dom, subj, preds, (L + j) % len(Q_MET), "eval"))
    for i, k in enumerate(range(45, 61)):              # reserved root range
        add(_met_root(k, i % len(Q_MET), "eval"))
    for i, d in enumerate(range(58, 80, 2)):           # reserved clean_div: product > train max (504)
        add(_met_clean_div(d, 10 + (i % 4), i % len(Q_MET), "eval"))
    for i, nn in enumerate(range(10, 15)):             # reserved clean_odd range
        add(_met_clean_odd(nn, i % len(Q_MET), "eval"))
    for i, n in enumerate(range(31, 42)):              # reserved offby range
        add(_met_offby(n, i % len(Q_MET), "eval"))
    for i, n in enumerate(range(31, 42)):              # reserved false-identity range
        add(_met_false_identity(n, i % len(Q_MET), "eval"))
    return out

def build_metacognitive(need, rng, exclude):
    out, seen = [], set(exclude)
    for it in _metacognitive_eval(seen):
        out.append(it)
    made, attempts = 0, 0
    # weighted family mix spreads difficulty 1..5 across the volume
    fams = (["arith"] * 2 + ["half"] * 2 + ["affirm1"] + ["affirm2"] + ["affirm3"] + ["affirm4"]
            + ["root"] * 2 + ["clean_div"] * 2 + ["clean_odd"] * 2 + ["offby"] * 2 + ["false_id"] * 2)
    while made < need and attempts < need * 60 + 4000:
        attempts += 1
        fam = rng.choice(fams); qi = rng.randrange(len(Q_MET))
        if fam == "arith":
            it = _met_arith(rng.randint(2, 39), rng.randint(2, 39), qi, "train")   # reserved-disjoint range
        elif fam == "half":
            it = _met_half(rng.choice(range(44, 600, 4)), qi, "train")
        elif fam.startswith("affirm"):
            L = int(fam[-1]); dom, subj = _domain_subject(rng); preds = rng.sample(PRED_TRAIN, L + 1)
            it = _met_affirm(dom, subj, preds, qi, "train")
        elif fam == "root":
            it = _met_root(rng.randint(6, 44), qi, "train")
        elif fam == "clean_div":
            it = _met_clean_div(rng.choice(range(4, 58, 2)), rng.randint(3, 9), qi, "train")
        elif fam == "clean_odd":
            it = _met_clean_odd(rng.randint(3, 9), qi, "train")
        elif fam == "offby":
            it = _met_offby(rng.randint(5, 30), qi, "train")
        else:
            it = _met_false_identity(rng.randint(5, 30), qi, "train")
        if it["problem"] in seen:
            continue
        seen.add(it["problem"]); out.append(it); made += 1
    return out

# --- authored types (bank x format); grow BANKS for larger seasons ----------
Q_ABD = ["What most plausibly explains this?", "Rank the possible explanations and justify the most likely.",
         "Which explanation best fits, and why are the alternatives weaker?", "You are diagnosing this. What is the single most likely cause?",
         "State your leading hypothesis and the single observation that most rules out the runner-up.",
         "Give the most probable cause and briefly say why each alternative is less likely.",
         "As the technician on call, what is your leading diagnosis and reasoning?",
         "Infer the best explanation and note your confidence in it."]
Q_ABD_EVAL = ["Which single cause best explains this, and why are the alternatives weaker?"]
def _abd_bank_bases(entries):
    bases = []
    for dom, subject, sigs in entries:
        for sig, cause, alts in sigs:
            bases.append((dom, subject, sig, cause, alts))
    return bases

def _mk_abd_rank(dom, subject, sig, cause, alts, stem, split):
    """Ranking-by-signature item; EACH base emitted ONCE (phrasing fixed by a
    stable hash of its identity) so an answer is never re-skinned across stems."""
    steps = [(f"Observation: {subject}, {sig}.", "valid")]
    for alt, why in alts:
        steps.append((f"Alternative '{alt}' is unlikely: {why}.", "valid"))
    steps.append((f"Best explanation: {cause}.", "valid"))
    steps.append(("Confidence moderate: not every competing factor was directly measured.", "valid"))
    diff = 2 if len(alts) <= 1 else 3 if len(alts) == 2 else 4
    return dict(domain=dom, problem=f"{cap(subject)}: {sig}. {stem}", steps=steps,
        final=cap(cause) + ".", difficulty=diff, confidence=0.65, vm="answer_match", split=split,
        vd=f"Planted cause: {cause}; the top explanation ranks above the listed alternatives by the distinguishing evidence.")

def _mk_abd_localize(dom, stages, jfail, distract, split):
    """Parametric fault-localization: from an OK/error/starved pattern in a
    series, localize the fault. The answer (the faulty component) genuinely
    varies with the stage set and fault position."""
    n = len(stages); faulty = stages[jfail]
    upstream = stages[:jfail]; downstream = stages[jfail + 1:]
    us = ", ".join(f"the {s}" for s in upstream) or "no earlier stage"
    ds = ", ".join(f"the {s}" for s in downstream) or "no later stage"
    prob = (f"A process runs in series through {n} stages: " + ", ".join(f"the {s}" for s in stages) + ". "
            f"Health checks show {us} reporting OK, the {faulty} reporting an error, and {ds} receiving nothing (starved). "
            + (f"Separately, a status label on the {downstream[-1]} is outdated. " if distract and downstream else "")
            + "Which single stage is the fault, and how are the others explained away?")
    steps = [
        (f"This is a {n}-stage series process, so a fault at one stage starves everything downstream while upstream stages stay healthy.", "valid"),
        (f"Upstream ({us}) reports OK, so the fault is not before the {faulty}.", "valid"),
        (f"Downstream ({ds}) receives nothing -- consistent with being starved by an upstream block, not with its own fault.", "valid"),
        (f"The {faulty} is the earliest stage whose own signal is anomalous, so it best explains the whole pattern.", "valid"),
    ]
    if distract and downstream:
        steps.append((f"The outdated label on the {downstream[-1]} is cosmetic and carries no failure signal; rule it out.", "valid"))
    steps.append(("Best explanation: the earliest stage that itself reports an anomaly.", "valid"))
    diff = min(max(1, n - 1) + (1 if distract else 0), 5)
    return dict(domain=dom, problem=prob, steps=steps, final=f"The fault is at the {faulty}.",
        difficulty=diff, confidence=0.8, vm="answer_match", split=split,
        vd=f"Localized to the {faulty} (stage {jfail+1}): upstream OK, downstream starved; earliest anomalous stage.")

# Component names for fault-localization: adjective x noun -> realistic, distinct
# stage names. Eval reserves the last 3 adjectives, so every eval component name
# (hence every eval fault-answer) is disjoint from train.
ABD_ADJ = ["intake", "primary", "secondary", "upstream", "relief", "bypass", "main", "auxiliary",
           "inlet", "outlet", "control", "feed", "return", "pilot", "booster", "trim", "isolation", "purge"]
ABD_NOUN = ["valve", "pump", "filter", "sensor", "regulator", "manifold", "coupling", "relay",
            "gate", "module", "junction", "buffer", "compressor", "exchanger", "actuator", "condenser"]
ABD_LOC_DOMS = ["engineering and physical systems", "mechanical / systems troubleshooting",
                "program behavior", "incident and root-cause analysis", "chemistry",
                "biology and ecology", "science", "finance and business operations",
                "algorithms and program analysis", "economics and markets",
                "law and regulation", "everyday planning"]
def _abd_parts(adjs, rng, n):
    return [f"{rng.choice(adjs)} {rng.choice(ABD_NOUN)}" for _ in range(n * 3)]

def _abductive_eval(seen):
    """Held-out eval: reserved bank entries (last 6) + reserved component adjectives
    + an eval-only stem -> disjoint problems and answers."""
    out = []
    def add(it):
        if it and it["problem"] not in seen:
            seen.add(it["problem"]); out.append(it)
    for (dom, subject, sig, cause, alts) in _abd_bank_bases(ABD_BANK[-6:]):
        add(_mk_abd_rank(dom, subject, sig, cause, alts, Q_ABD_EVAL[0], "eval"))
    ev_adj = ABD_ADJ[-3:]
    idx = 0
    for a in ev_adj:
        for nz in ABD_NOUN:
            for n in (3, 4):
                stages = [f"{a} {ABD_NOUN[(idx + k) % len(ABD_NOUN)]}" for k in range(n)]
                if len({*stages}) < n:
                    idx += 1; continue
                jfail = idx % n
                add(_mk_abd_localize(ABD_LOC_DOMS[idx % len(ABD_LOC_DOMS)], stages, jfail, jfail < n - 1, "eval"))
                idx += 1
    return out

def build_abductive(need, rng, exclude):
    out, seen = [], set(exclude)
    for it in _abductive_eval(seen):
        out.append(it)
    train_bases = _abd_bank_bases(ABD_BANK[:-6])
    # bank layer (capped: each base ONCE, stem fixed by a stable hash of identity)
    for (dom, subject, sig, cause, alts) in train_bases:
        stem = Q_ABD[_shash("A|" + subject + "|" + sig) % len(Q_ABD)]
        it = _mk_abd_rank(dom, subject, sig, cause, alts, stem, "train")
        if it["problem"] not in seen:
            seen.add(it["problem"]); out.append(it)
    made = sum(1 for it in out if it.get("split") == "train")
    # volume layer: parametric fault-localization over compound component names
    # (train adjectives only -> answers disjoint from the reserved eval set).
    train_adj = ABD_ADJ[:-3]
    attempts = 0
    while made < need and attempts < need * 60 + 5000:
        attempts += 1
        n = rng.randint(2, 6)
        parts = list(dict.fromkeys(_abd_parts(train_adj, rng, n)))   # distinct, order-preserving
        if len(parts) < n:
            continue
        stages = parts[:n]
        jfail = rng.randrange(n)
        distract = rng.random() < 0.4 and jfail < n - 1
        dom = ABD_LOC_DOMS[attempts % len(ABD_LOC_DOMS)]
        it = _mk_abd_localize(dom, stages, jfail, distract, "train")
        if it["problem"] in seen:
            continue
        seen.add(it["problem"]); out.append(it); made += 1
    return out

# --------------------------------------------------------------------------
# ANALOGICAL
#
# Design (see reasoning-types/analogical.md):
#   * Analogical reasoning transfers a RELATIONAL STRUCTURE from source to
#     target, mapping roles rather than surface features. So every trace names
#     the roles and performs the mapping; there is no boilerplate "instantiates
#     it" filler step.
#   * Surface content is varied while structure is held fixed, so only the
#     relation - not vocabulary co-occurrence - solves the item.
#   * The difficulty-4/5 items carry a real surface DISTRACTOR (or a competing
#     relation) that the trace must reject BY ROLE. These are the items that
#     force structural mapping; the earlier corpus lacked them entirely.
#
# Anti-leakage: train and eval draw from DISJOINT answer pools. For each
# relation the answer pairs are partitioned so an eval target answer never
# appears (as demo or answer) in any train item; several whole relations and
# whole trap-skins are held out for eval to test transfer to unseen relations
# too. Each instance carries an explicit "split" the writer honors, so the
# split is structural, not a blind slice of one ordered list.
# --------------------------------------------------------------------------
def _art(w):
    """Correct indefinite article for a word (handles unit/use/hour-style cases)."""
    wl = w.lower()
    if wl[:3] in ("uni", "use", "uti", "ubi", "eur") or wl.startswith("one"):
        return "a " + w
    if wl[:1] == "h" and wl[:5] in ("hour", "honor", "hones"):
        return "an " + w
    return ("an " if wl[:1] in "aeiou" else "a ") + w

# (relation phrase, left role, right role, domain, difficulty, eval_only, pairs)
ANA_REL = [
    ("animal to the sound it makes", "animal", "sound", "biology and ecology", 1, False,
     [("dog","bark"),("cat","meow"),("cow","moo"),("duck","quack"),("lion","roar"),("horse","neigh"),("sheep","bleat"),("frog","croak"),("bee","buzz"),("snake","hiss"),("owl","hoot"),("wolf","howl"),("pig","oink"),("crow","caw"),("hen","cluck")]),
    ("animal to its young", "creature", "young", "biology and ecology", 1, False,
     [("dog","puppy"),("cat","kitten"),("cow","calf"),("horse","foal"),("sheep","lamb"),("lion","cub"),("frog","tadpole"),("hen","chick"),("kangaroo","joey"),("bear","cub"),("deer","fawn"),("goat","kid"),("duck","duckling"),("fox","kit"),("eagle","eaglet")]),
    ("profession to its tool", "profession", "tool", "engineering and physical systems", 2, False,
     [("chef","knife"),("painter","brush"),("carpenter","hammer"),("writer","pen"),("surgeon","scalpel"),("photographer","camera"),("farmer","plow"),("tailor","needle"),("gardener","spade"),("blacksmith","anvil"),("dentist","drill"),("barber","scissors"),("mechanic","wrench"),("sculptor","chisel"),("cartographer","compass")]),
    ("object to its material", "thing", "material", "engineering and physical systems", 2, False,
     [("book","paper"),("window","glass"),("tire","rubber"),("wire","copper"),("ring","gold"),("bottle","plastic"),("table","wood"),("blade","steel"),("sweater","wool"),("brick","clay"),("candle","wax"),("rope","fiber"),("balloon","latex"),("crayon","wax"),("mug","ceramic")]),
    ("member to its category", "member", "category", "biology and ecology", 2, False,
     [("apple","fruit"),("car","vehicle"),("violin","instrument"),("salmon","fish"),("oak","tree"),("sparrow","bird"),("iron","metal"),("rose","flower"),("whale","mammal"),("triangle","shape"),("tennis","sport"),("oxygen","gas"),("ruby","gemstone"),("maple","tree"),("trumpet","instrument")]),
    ("word to a stronger-degree version", "word", "stronger form", "formal grammars and symbol systems", 2, False,
     [("warm","hot"),("big","huge"),("cool","cold"),("good","great"),("tired","exhausted"),("small","tiny"),("happy","ecstatic"),("bad","terrible"),("wet","soaked"),("hungry","starving"),("angry","furious"),("quiet","silent"),("pretty","gorgeous"),("funny","hilarious"),("sad","devastated")]),
    ("cause to its typical effect", "cause", "typical effect", "science", 2, False,
     [("spark","fire"),("virus","illness"),("rain","flood"),("study","knowledge"),("exercise","fitness"),("drought","famine"),("friction","heat"),("practice","skill"),("investment","growth"),("pollution","smog"),("training","endurance"),("sunlight","photosynthesis")]),
    ("active driver to the medium it moves", "driver", "medium moved", "engineering and physical systems", 2, False,
     [("heart","blood"),("pump","water"),("battery","charge"),("fan","air"),("turbine","steam"),("plunger","fluid"),("escalator","people"),("conveyor","packages"),("speaker","sound"),("windmill","grain")]),
    ("tool to its function", "tool", "function", "engineering and physical systems", 2, False,
     [("knife","cutting"),("pen","writing"),("key","unlocking"),("broom","sweeping"),("thermometer","measuring temperature"),("compass","finding direction"),("ruler","measuring length"),("clock","telling time"),("filter","removing impurities"),("brake","stopping"),("magnet","attracting iron"),("shovel","digging")]),
    ("country to its capital", "country", "capital", "science", 2, False,
     [("France","Paris"),("Japan","Tokyo"),("Egypt","Cairo"),("Peru","Lima"),("Kenya","Nairobi"),("Norway","Oslo"),("Cuba","Havana"),("Nepal","Kathmandu"),("Ghana","Accra"),("Chile","Santiago"),("Iraq","Baghdad"),("Greece","Athens")]),
    ("unit to what it measures", "unit", "measured quantity", "science", 2, False,
     [("meter","length"),("gram","mass"),("second","time"),("ampere","current"),("volt","voltage"),("watt","power"),("liter","volume"),("kelvin","temperature"),("pascal","pressure"),("hertz","frequency"),("joule","energy"),("mole","amount")]),
    ("element to its chemical symbol", "chemical element", "chemical symbol", "chemistry", 2, False,
     [("hydrogen","H"),("oxygen","O"),("carbon","C"),("sodium","Na"),("iron","Fe"),("gold","Au"),("helium","He"),("nitrogen","N"),("chlorine","Cl"),("potassium","K"),("calcium","Ca"),("silver","Ag"),("copper","Cu"),("lead","Pb")]),
    ("instrument to its family", "instrument", "family", "engineering and physical systems", 2, False,
     [("violin","strings"),("trumpet","brass"),("flute","woodwind"),("drum","percussion"),("cello","strings"),("clarinet","woodwind"),("trombone","brass"),("timpani","percussion"),("harp","strings"),("oboe","woodwind"),("tuba","brass"),("cymbal","percussion")]),
    ("profession to its workplace", "profession", "workplace", "economics and markets", 2, False,
     [("chef","kitchen"),("judge","courtroom"),("teacher","classroom"),("pilot","cockpit"),("surgeon","operating room"),("farmer","field"),("actor","stage"),("banker","bank"),("librarian","library"),("chemist","laboratory"),("barista","cafe"),("miner","mine")]),
    ("word to its opposite", "word", "opposite", "formal grammars and symbol systems", 1, False,
     [("hot","cold"),("up","down"),("fast","slow"),("open","closed"),("day","night"),("win","lose"),("push","pull"),("rise","fall"),("wet","dry"),("near","far"),("full","empty"),("begin","end"),("light","dark"),("buy","sell")]),
    ("animal to its habitat", "animal", "habitat", "biology and ecology", 2, False,
     [("fish","water"),("camel","desert"),("polar bear","the arctic"),("monkey","jungle"),("mole","underground"),("frog","pond"),("lion","savanna"),("penguin","antarctica"),("bat","cave"),("whale","ocean"),("owl","forest"),("crab","shore")]),
    ("whole to one of its parts", "whole", "part", "engineering and physical systems", 2, False,
     [("car","wheel"),("tree","branch"),("book","page"),("house","room"),("body","limb"),("computer","keyboard"),("bicycle","pedal"),("clock","hand"),("guitar","string"),("flower","petal"),("ship","deck"),("phone","screen")]),
    ("process to its product", "process", "product", "science", 2, False,
     [("photosynthesis","glucose"),("combustion","heat"),("digestion","nutrients"),("evaporation","vapor"),("fermentation","alcohol"),("erosion","sediment"),("condensation","water"),("respiration","energy"),("baking","bread"),("smelting","metal"),("distillation","spirit"),("weathering","soil")]),
    ("number to its square", "number", "square", "mathematics", 1, False,
     [("2","4"),("3","9"),("4","16"),("5","25"),("6","36"),("7","49"),("8","64"),("9","81"),("10","100"),("11","121"),("12","144"),("13","169")]),
    ("shape to its number of sides", "shape", "number of sides", "mathematics", 1, False,
     [("triangle","3"),("square","4"),("pentagon","5"),("hexagon","6"),("heptagon","7"),("octagon","8"),("nonagon","9"),("decagon","10"),("quadrilateral","4"),("dodecagon","12")]),
    ("programming construct to its purpose", "construct", "purpose", "program behavior", 2, False,
     [("loop","repetition"),("function","reuse"),("variable","storage"),("conditional","branching"),("array","indexed collection"),("pointer","indirection"),("exception","error handling"),("comment","documentation"),("constant","fixed value"),("recursion","self-reference")]),
    ("legal area to what it governs", "legal area", "subject governed", "law and regulation", 3, False,
     [("tort law","civil injuries"),("contract law","agreements"),("criminal law","offenses"),("property law","ownership"),("family law","domestic relations"),("tax law","levies"),("labor law","employment"),("maritime law","shipping"),("patent law","inventions"),("constitutional law","state powers")]),
    ("chemical formula to its common name", "formula", "common name", "chemistry", 2, False,
     [("NaCl","salt"),("H2O","water"),("CO2","carbon dioxide"),("CH4","methane"),("NH3","ammonia"),("O2","oxygen"),("C6H12O6","glucose"),("NaHCO3","baking soda"),("H2O2","hydrogen peroxide"),("CaCO3","limestone")]),
    ("planet to its order from the sun", "planet", "order from the sun", "science", 2, False,
     [("Mercury","first"),("Venus","second"),("Earth","third"),("Mars","fourth"),("Jupiter","fifth"),("Saturn","sixth"),("Uranus","seventh"),("Neptune","eighth")]),
    ("currency to its country", "currency", "country", "economics and markets", 2, False,
     [("yen","Japan"),("pound","Britain"),("rupee","India"),("peso","Mexico"),("won","South Korea"),("real","Brazil"),("rand","South Africa"),("lira","Turkey"),("baht","Thailand"),("zloty","Poland")]),
    ("metric prefix to its factor", "prefix", "factor", "science", 2, False,
     [("kilo","thousand"),("mega","million"),("giga","billion"),("milli","thousandth"),("micro","millionth"),("nano","billionth"),("centi","hundredth"),("deci","tenth"),("tera","trillion"),("hecto","hundred")]),
    ("verb to its past tense", "verb", "past tense", "formal grammars and symbol systems", 2, False,
     [("go","went"),("run","ran"),("eat","ate"),("see","saw"),("take","took"),("buy","bought"),("bring","brought"),("teach","taught"),("think","thought"),("catch","caught"),("build","built"),("sing","sang")]),
    ("sport to its playing surface", "sport", "playing surface", "social situations", 2, False,
     [("soccer","pitch"),("tennis","court"),("ice hockey","rink"),("golf","course"),("baseball","diamond"),("bowling","lane"),("swimming","pool"),("boxing","ring"),("cricket","pitch"),("track","oval")]),
    ("data structure to its access pattern", "data structure", "access pattern", "algorithms and program analysis", 3, False,
     [("stack","last-in first-out"),("queue","first-in first-out"),("array","random access"),("linked list","sequential access"),("hash map","key lookup"),("heap","priority order"),("binary search tree","sorted traversal"),("graph","adjacency")]),
    # --- relations reserved ENTIRELY for eval: test transfer to unseen relations ---
    ("worker to what they produce", "worker", "product", "economics and markets", 2, True,
     [("baker","bread"),("author","book"),("brewer","beer"),("mason","wall"),("weaver","cloth"),("potter","pottery"),("cobbler","shoes"),("vintner","wine"),("smith","tools"),("jeweler","jewelry")]),
    ("gas to a hazard it poses", "substance", "hazard", "chemistry", 3, True,
     [("methane","explosion"),("carbon monoxide","asphyxiation"),("chlorine","corrosion"),("radon","radiation"),("ammonia","burns"),("hydrogen","fire"),("ozone","lung irritation"),("sulfur dioxide","acid rain")]),
    ("polygon to its interior angle sum", "polygon", "interior angle sum", "mathematics", 3, True,
     [("triangle","180"),("quadrilateral","360"),("pentagon","540"),("hexagon","720"),("heptagon","900"),("octagon","1080"),("nonagon","1260"),("decagon","1440")]),
    ("device to the energy conversion it performs", "device", "energy conversion", "engineering and physical systems", 3, True,
     [("motor","electrical to mechanical"),("generator","mechanical to electrical"),("battery","chemical to electrical"),("solar cell","light to electrical"),("microphone","sound to electrical"),("speaker","electrical to sound"),("heater","electrical to thermal"),("turbine","kinetic to mechanical")]),
    # --- additional train relations (breadth) ---
    ("tree to its fruit", "tree", "fruit", "biology and ecology", 1, False,
     [("apple tree","apple"),("orange tree","orange"),("oak","acorn"),("vine","grape"),("olive tree","olive"),("cherry tree","cherry"),("fig tree","fig"),("almond tree","almond"),("peach tree","peach"),("lemon tree","lemon"),("coconut palm","coconut"),("chestnut tree","chestnut")]),
    ("liquid to its frozen form", "liquid", "frozen form", "science", 2, False,
     [("water","ice"),("lava","rock"),("candle wax","solid wax"),("milk","milk ice"),("molten glass","glass"),("mercury","solid mercury"),("juice","popsicle"),("cream","ice cream"),("honey","crystallized honey"),("steel","ingot")]),
    ("worker to their finished product", "maker", "product", "economics and markets", 2, False,
     [("carpenter","furniture"),("chef","meal"),("architect","building"),("composer","symphony"),("programmer","software"),("farmer","crop"),("sculptor","statue"),("playwright","play"),("tailor","garment"),("engineer","machine"),("baker","loaf"),("brewer","ale")]),
    ("celestial body to what orbits it", "body", "satellite", "science", 3, False,
     [("Earth","the Moon"),("the Sun","the planets"),("Jupiter","its moons"),("a nucleus","electrons"),("a galaxy","its stars"),("Mars","Phobos"),("Saturn","its rings"),("a planet","its atmosphere")]),
    ("action to the sense it uses", "action", "sense", "biology and ecology", 1, False,
     [("seeing","sight"),("hearing","hearing"),("tasting","taste"),("smelling","smell"),("touching","touch"),("reading","sight"),("listening","hearing"),("sniffing","smell")]),
    ("material to the craft that shapes it", "material", "craft", "engineering and physical systems", 2, False,
     [("clay","pottery"),("wood","carpentry"),("metal","smithing"),("glass","glassblowing"),("stone","masonry"),("cloth","tailoring"),("leather","tanning"),("gold","goldsmithing"),("paper","origami"),("wax","candlemaking")]),
    ("emotion to its facial sign", "feeling", "facial sign", "social situations", 2, False,
     [("happiness","a smile"),("sadness","a frown"),("surprise","raised brows"),("anger","a scowl"),("fear","wide eyes"),("disgust","a wrinkled nose"),("boredom","a yawn"),("confusion","a furrowed brow")]),
    ("disease to the organ it attacks", "disease", "organ", "medicine-style diagnosis", 3, False,
     [("hepatitis","the liver"),("nephritis","the kidney"),("pneumonia","the lungs"),("gastritis","the stomach"),("arthritis","the joints"),("meningitis","the brain lining"),("carditis","the heart"),("dermatitis","the skin")]),
    ("operation to its inverse", "operation", "inverse", "mathematics", 2, False,
     [("addition","subtraction"),("multiplication","division"),("squaring","square root"),("exponentiation","logarithm"),("differentiation","integration"),("encryption","decryption"),("folding","unfolding"),("freezing","melting")]),
    ("tool to the quantity it measures", "measuring tool", "quantity", "science", 2, False,
     [("thermometer","temperature"),("barometer","pressure"),("odometer","distance"),("voltmeter","voltage"),("scale","weight"),("clock","time"),("hygrometer","humidity"),("seismometer","ground motion"),("ammeter","current"),("speedometer","speed") ]),
    # --- added breadth relations (2026-07) ---
    ('algorithm to its complexity class', 'algorithm', 'complexity', 'algorithms and program analysis', 3, False,
     [('binary search', 'O(log n)'), ('linear scan', 'O(n)'), ('merge sort', 'O(n log n)'), ('bubble sort', 'O(n^2)'), ('hash-table lookup', 'O(1) average'), ('heap insertion', 'O(log n)'), ('tree traversal', 'O(n)'), ('breadth-first search', 'O(V+E)'), ('Euclid gcd', 'O(log n)'), ('subset enumeration', 'O(2^n)')]),
    ('file format to usual extension', 'format', 'extension', 'program behavior', 2, False,
     [('JSON', '.json'), ('CSV', '.csv'), ('HTML', '.html'), ('Markdown', '.md'), ('Python source', '.py'), ('JavaScript source', '.js'), ('YAML', '.yaml'), ('PDF', '.pdf'), ('PNG image', '.png'), ('JPEG image', '.jpg')]),
    ('HTTP status to meaning', 'status', 'meaning', 'program behavior', 2, False,
     [('200', 'OK'), ('201', 'Created'), ('301', 'Moved Permanently'), ('302', 'Found'), ('400', 'Bad Request'), ('401', 'Unauthorized'), ('403', 'Forbidden'), ('404', 'Not Found'), ('409', 'Conflict'), ('500', 'Internal Server Error')]),
    ('protocol to default port', 'protocol', 'default port', 'program behavior', 3, False,
     [('HTTP', '80'), ('HTTPS', '443'), ('SSH', '22'), ('DNS', '53'), ('SMTP', '25'), ('IMAP', '143'), ('POP3', '110'), ('LDAP', '389'), ('NTP', '123'), ('PostgreSQL', '5432')]),
    ('grammar term to role', 'grammar term', 'role', 'formal grammars and symbol systems', 2, False,
     [('noun', 'names a thing'), ('verb', 'states an action'), ('adjective', 'modifies a noun'), ('adverb', 'modifies a verb'), ('preposition', 'shows relation'), ('conjunction', 'joins elements'), ('pronoun', 'stands for a noun'), ('determiner', 'specifies a noun'), ('interjection', 'expresses emotion'), ('article', 'marks definiteness')]),
    ('rhetorical device to effect', 'device', 'effect', 'narrative and discourse', 3, False,
     [('metaphor', 'implicit comparison'), ('simile', 'explicit comparison'), ('irony', 'meaning reversal'), ('foreshadowing', 'future hint'), ('alliteration', 'repeated initial sound'), ('hyperbole', 'deliberate exaggeration'), ('understatement', 'deliberate downplaying'), ('anaphora', 'repeated opening'), ('chiasmus', 'reversed structure'), ('personification', 'human traits assigned')]),
    ('plot role to narrative function', 'plot role', 'function', 'narrative and discourse', 2, False,
     [('protagonist', 'central actor'), ('antagonist', 'opposing force'), ('mentor', 'guidance source'), ('foil', 'contrast figure'), ('confidant', 'private listener'), ('herald', 'call to change'), ('trickster', 'disruptive reversal'), ('narrator', 'story voice'), ('deuteragonist', 'secondary lead'), ('chorus', 'commentary voice')]),
    ('contract clause to purpose', 'clause', 'purpose', 'law and regulation', 3, False,
     [('indemnity', 'loss shifting'), ('arbitration', 'private dispute resolution'), ('confidentiality', 'secrecy duty'), ('force majeure', 'excused nonperformance'), ('assignment', 'transfer permission'), ('severability', 'survival of remainder'), ('termination', 'ending rights'), ('warranty', 'quality promise'), ('venue', 'forum selection'), ('notice', 'communication method')]),
    ('ethical framework to decision rule', 'framework', 'decision rule', 'ethics', 3, False,
     [('utilitarianism', 'maximize welfare'), ('deontology', 'follow duties'), ('virtue ethics', 'cultivate character'), ('care ethics', 'preserve relationships'), ('contractualism', 'justify to each person'), ('rights theory', 'respect entitlements'), ('principlism', 'balance principles'), ('casuistry', 'reason from cases'), ('egalitarianism', 'reduce unfair inequality'), ('consequentialism', 'judge outcomes')]),
    ('bioethical principle to focus', 'principle', 'focus', 'ethics', 2, False,
     [('autonomy', 'informed choice'), ('beneficence', 'patient benefit'), ('nonmaleficence', 'avoiding harm'), ('justice', 'fair distribution'), ('confidentiality', 'privacy protection'), ('fidelity', 'keeping promises'), ('veracity', 'truth telling'), ('dignity', 'respectful treatment'), ('proportionality', 'balanced burden'), ('transparency', 'open explanation')]),
    ('market structure to defining feature', 'market structure', 'feature', 'economics and markets', 3, False,
     [('perfect competition', 'many price takers'), ('monopoly', 'single seller'), ('monopsony', 'single buyer'), ('oligopoly', 'few sellers'), ('duopoly', 'two sellers'), ('cartel', 'coordinated sellers'), ('auction', 'bids set price'), ('natural monopoly', 'scale economies'), ('contestable market', 'entry threat'), ('two-sided market', 'two user groups')]),
    ('financial statement to reported content', 'statement', 'reported content', 'finance and business operations', 2, False,
     [('balance sheet', 'assets and liabilities'), ('income statement', 'profit or loss'), ('cash-flow statement', 'cash movements'), ('trial balance', 'ledger totals'), ('bank reconciliation', 'book-bank differences'), ('aging report', 'overdue receivables'), ('budget variance report', 'plan versus actual'), ('inventory report', 'stock on hand'), ('invoice', 'amount owed'), ('receipt', 'payment proof')]),
    ('medical sign to likely body system', 'sign', 'body system', 'medicine-style diagnosis', 3, False,
     [('wheezing', 'respiratory'), ('jaundice', 'hepatic'), ('hematuria', 'urinary'), ('bradycardia', 'cardiac'), ('aphasia', 'neurologic'), ('rash', 'dermatologic'), ('edema', 'circulatory'), ('vertigo', 'vestibular'), ('polyuria', 'endocrine'), ('melena', 'gastrointestinal')]),
    ('lab test to primary analyte', 'test', 'analyte', 'medicine-style diagnosis', 2, False,
     [('hemoglobin A1c', 'average glucose'), ('troponin', 'cardiac injury'), ('creatinine', 'kidney function'), ('bilirubin', 'bile pigment'), ('TSH', 'thyroid stimulation'), ('INR', 'clotting tendency'), ('D-dimer', 'fibrin breakdown'), ('CRP', 'inflammation'), ('ALT', 'liver injury'), ('BNP', 'heart strain')]),
    ('ecological interaction to relationship', 'interaction', 'relationship', 'biology and ecology', 3, False,
     [('mutualism', 'both benefit'), ('commensalism', 'one benefits one unaffected'), ('parasitism', 'one benefits one harmed'), ('predation', 'one eats another'), ('competition', 'both incur cost'), ('amensalism', 'one harmed one unaffected'), ('pollination', 'plant and pollinator benefit'), ('decomposition', 'consumer recycles remains'), ('herbivory', 'plant eaten'), ('facilitation', 'one eases another')]),
    ('cell organelle to function', 'organelle', 'function', 'biology and ecology', 2, False,
     [('nucleus', 'DNA storage'), ('mitochondrion', 'ATP production'), ('ribosome', 'protein synthesis'), ('Golgi apparatus', 'protein sorting'), ('lysosome', 'waste digestion'), ('chloroplast', 'photosynthesis'), ('cell membrane', 'selective barrier'), ('vacuole', 'storage'), ('cytoskeleton', 'structural support'), ('nucleolus', 'ribosome assembly')]),
    ('chemical process to driving change', 'process', 'driving change', 'chemistry', 3, False,
     [('acid-base neutralization', 'proton transfer'), ('oxidation', 'electron loss'), ('reduction', 'electron gain'), ('precipitation', 'insoluble solid formation'), ('combustion', 'reaction with oxygen'), ('hydrolysis', 'bond breaking by water'), ('polymerization', 'monomer joining'), ('crystallization', 'ordered solid formation'), ('distillation', 'boiling-point separation'), ('osmosis', 'solvent movement')]),
    ('chemical technique to separation basis', 'technique', 'basis', 'chemistry', 3, False,
     [('distillation', 'boiling point'), ('filtration', 'particle size'), ('chromatography', 'affinity differences'), ('centrifugation', 'density'), ('extraction', 'solubility'), ('crystallization', 'solubility change'), ('electrophoresis', 'charge and size'), ('decanting', 'settling'), ('dialysis', 'membrane cutoff'), ('evaporation', 'volatility')]),
    ('physics quantity to SI unit', 'quantity', 'SI unit', 'science', 2, False,
     [('force', 'newton'), ('energy', 'joule'), ('power', 'watt'), ('pressure', 'pascal'), ('frequency', 'hertz'), ('charge', 'coulomb'), ('resistance', 'ohm'), ('conductance', 'siemens'), ('capacitance', 'farad'), ('inductance', 'henry')]),
    ('geologic feature to forming process', 'feature', 'forming process', 'science', 3, False,
     [('delta', 'sediment deposition'), ('canyon', 'river erosion'), ('dune', 'wind deposition'), ('moraine', 'glacial deposition'), ('stalactite', 'mineral precipitation'), ('volcano', 'magma eruption'), ('fault scarp', 'tectonic displacement'), ('sea arch', 'wave erosion'), ('sinkhole', 'dissolution collapse'), ('fjord', 'glacial carving')]),
    ('circuit component to role', 'component', 'role', 'engineering and physical systems', 2, False,
     [('resistor', 'limits current'), ('capacitor', 'stores charge'), ('inductor', 'stores magnetic energy'), ('diode', 'allows unidirectional current'), ('transistor', 'switches or amplifies'), ('fuse', 'opens on overcurrent'), ('transformer', 'changes voltage'), ('LED', 'emits light'), ('switch', 'opens or closes circuit'), ('battery', 'supplies voltage')]),
    ('mechanical element to function', 'element', 'function', 'mechanical / systems troubleshooting', 2, False,
     [('bearing', 'reduces friction'), ('spring', 'stores elastic energy'), ('gear', 'transmits torque'), ('clutch', 'engages rotation'), ('brake', 'dissipates motion'), ('seal', 'prevents leakage'), ('valve', 'controls flow'), ('flywheel', 'stores rotational energy'), ('cam', 'converts rotation to lift'), ('filter', 'traps particles')]),
    ('incident signal to diagnostic source', 'signal', 'source', 'incident and root-cause analysis', 2, False,
     [('error-rate spike', 'application logs'), ('latency spike', 'tracing spans'), ('CPU saturation', 'host metrics'), ('memory leak', 'heap profile'), ('packet loss', 'network telemetry'), ('failed deploy', 'deployment log'), ('database locks', 'query monitor'), ('disk full', 'filesystem metrics'), ('cache misses', 'cache metrics'), ('auth failures', 'audit log')]),
    ('outage mitigation to primary effect', 'mitigation', 'effect', 'incident and root-cause analysis', 3, False,
     [('rollback', 'restore prior version'), ('rate limiting', 'reduce request load'), ('circuit breaker', 'stop cascading calls'), ('failover', 'move traffic'), ('cache flush', 'remove stale entries'), ('scaling out', 'add capacity'), ('feature flag off', 'disable risky path'), ('restart', 'clear transient state'), ('database index', 'speed queries'), ('queue drain', 'clear backlog')]),
    ('planning constraint to limited resource', 'constraint', 'limited resource', 'everyday planning', 2, False,
     [('deadline', 'time'), ('budget cap', 'money'), ('venue capacity', 'seats'), ('dietary rule', 'menu choices'), ('travel distance', 'route options'), ('quiet hours', 'noise'), ('childcare window', 'available hours'), ('parking limit', 'cars'), ('luggage allowance', 'baggage weight'), ('appointment slot', 'schedule')]),
    ('social cue to likely meaning', 'cue', 'meaning', 'social situations', 2, False,
     [('crossed arms', 'closed posture'), ('steady eye contact', 'attention'), ('looking at watch', 'impatience'), ('leaning in', 'interest'), ('long pause', 'hesitation'), ('raised hand', 'request to speak'), ('whispering', 'privacy seeking'), ('applause', 'approval'), ('averted gaze', 'discomfort'), ('nodding', 'acknowledgment')]),
    ('negotiation move to strategic purpose', 'move', 'purpose', 'negotiation and interpersonal strategy', 3, False,
     [('anchoring offer', 'set reference point'), ('BATNA disclosure', 'signal alternative'), ('concession', 'invite reciprocity'), ('package deal', 'trade across issues'), ('deadline', 'increase urgency'), ('objective criterion', 'legitimize proposal'), ('silence', 'draw information'), ('contingent contract', 'handle uncertainty'), ('walk-away', 'enforce limit'), ('logrolling', 'swap priorities')]),
    ('logical fallacy to its error', 'fallacy', 'error', 'logic puzzles', 3, False,
     [('affirming the consequent', 'reversing implication'), ('denying the antecedent', 'negating implication'), ('equivocation', 'shifting meaning'), ('false dilemma', 'excluding alternatives'), ('circular reasoning', 'assuming conclusion'), ('hasty generalization', 'too little evidence'), ('straw man', 'misrepresenting claim'), ('ad hominem', 'attacking person'), ('post hoc', 'confusing sequence with cause'), ('appeal to ignorance', 'absence as proof')]),
    ('database index type to lookup pattern', 'lookup index', 'lookup pattern', 'algorithms and program analysis', 3, False,
     [('hash index', 'equality lookup'), ('B-tree index', 'range lookup'), ('bitmap index', 'low-cardinality filters'), ('R-tree index', 'spatial lookup'), ('inverted index', 'term search'), ('covering index', 'table-free read'), ('partial index', 'predicate-filtered rows'), ('unique index', 'duplicate prevention'), ('clustered index', 'physical order'), ('composite index', 'multi-column lookup')]),
    ('graph term to meaning', 'graph term', 'meaning', 'algorithms and program analysis', 2, False,
     [('vertex', 'node'), ('edge', 'connection'), ('degree', 'incident edge count'), ('path', 'sequence of edges'), ('cycle', 'closed path'), ('component', 'connected subgraph'), ('tree', 'acyclic connected graph'), ('root', 'distinguished start node'), ('leaf', 'node with no children'), ('cut vertex', 'removal disconnects graph')]),
    ('cipher to key operation', 'cipher', 'key operation', 'formal grammars and symbol systems', 3, False,
     [('Caesar cipher', 'fixed shift'), ('Vigenere cipher', 'repeating keyword'), ('substitution cipher', 'symbol replacement'), ('transposition cipher', 'position rearrangement'), ('one-time pad', 'random key mixing'), ('RSA', 'modular exponentiation'), ('rail fence cipher', 'zigzag writing'), ('Atbash', 'alphabet reversal'), ('ROT13', 'thirteen-letter shift'), ('Playfair', 'digraph substitution')]),
    ('statistical plot to what it shows', 'plot', 'shown feature', 'mathematics', 2, False,
     [('histogram', 'distribution'), ('scatterplot', 'association'), ('boxplot', 'quartiles'), ('line chart', 'trend over time'), ('bar chart', 'category comparison'), ('residual plot', 'model errors'), ('heat map', 'matrix intensity'), ('Q-Q plot', 'distribution fit'), ('pie chart', 'part-whole share'), ('stem plot', 'individual values')]),
    ('probability distribution to typical use', 'distribution', 'typical use', 'mathematics', 3, False,
     [('binomial', 'success counts'), ('Poisson', 'event counts'), ('normal', 'bell-shaped variation'), ('exponential', 'waiting time'), ('uniform', 'equal likelihood'), ('geometric', 'trials until success'), ('beta', 'probability parameter'), ('gamma', 'positive waiting amount'), ('Bernoulli', 'single success flag'), ('hypergeometric', 'sampling without replacement')]),
    ('economic policy tool to immediate channel', 'policy tool', 'channel', 'economics and markets', 3, False,
     [('interest-rate cut', 'cheaper borrowing'), ('reserve requirement', 'bank lending capacity'), ('tariff', 'import price increase'), ('subsidy', 'lower producer cost'), ('price ceiling', 'maximum legal price'), ('price floor', 'minimum legal price'), ('quota', 'quantity restriction'), ('tax credit', 'after-tax incentive'), ('open-market purchase', 'more bank reserves'), ('carbon price', 'emissions cost')]),
    ('policy term to meaning', 'policy term', 'meaning', 'finance and business operations', 2, False,
     [('premium', 'price paid'), ('deductible', 'initial out-of-pocket amount'), ('copay', 'fixed service charge'), ('coinsurance', 'percentage share'), ('exclusion', 'uncovered condition'), ('rider', 'added coverage'), ('claim', 'payment request'), ('underwriting', 'risk assessment'), ('beneficiary', 'recipient of payout'), ('policy limit', 'maximum payout')]),
    ('court role to function', 'court role', 'function', 'law and regulation', 2, False,
     [('judge', 'rules on law'), ('jury', 'finds facts'), ('plaintiff', 'brings claim'), ('defendant', 'responds to claim'), ('bailiff', 'maintains order'), ('clerk', 'keeps records'), ('witness', 'gives testimony'), ('counsel', 'advocates position'), ('court reporter', 'creates transcript'), ('mediator', 'facilitates settlement')]),
    ('argument component to role', 'argument component', 'role', 'narrative and discourse', 2, False,
     [('claim', 'conclusion asserted'), ('premise', 'supporting reason'), ('warrant', 'linking principle'), ('evidence', 'supporting facts'), ('qualifier', 'strength marker'), ('rebuttal', 'exception condition'), ('counterclaim', 'opposing conclusion'), ('example', 'illustrative case'), ('definition', 'term clarification'), ('analogy', 'structural comparison')]),
    ('moral hazard setting to hidden behavior', 'setting', 'hidden behavior', 'ethics', 3, False,
     [('insured driver', 'riskier driving'), ('bailed-out bank', 'riskier lending'), ('monitored employee', 'shirking when unseen'), ('delegated official', 'self-serving discretion'), ('tenant with deposit waived', 'less careful upkeep'), ('platform moderator paid by volume', 'rushed judgments'), ('contractor on cost-plus terms', 'inflated costs'), ('student with unchecked collaboration', 'free riding'), ('agent handling funds', 'private spending'), ('doctor paid per procedure', 'overtreatment risk')]),

    # --- softer / artistic analogy relations (2026-07 follow-up) ---
    ('image detail to conveyed mood', 'detail', 'conveyed mood', 'narrative and discourse', 2, False,
     [('warm lamplight','comfort'),('long shadow','unease'),('open window','possibility'),('wilting flowers','neglect'),('fresh footprints','arrival'),('cracked teacup','fragility'),('rain-streaked glass','melancholy'),('crowded table','abundance'),('empty chair','absence'),('soft blanket','safety')]),
    ('story setting to emotional atmosphere', 'setting', 'emotional atmosphere', 'narrative and discourse', 2, False,
     [('abandoned station','loneliness'),('busy market','energy'),('quiet library','concentration'),('stormy coast','tension'),('sunlit kitchen','warmth'),('foggy alley','uncertainty'),('hospital corridor','anxiety'),('garden at dusk','reflection'),('festival street','celebration'),('locked attic','secrecy')]),
    ('musical feature to felt effect', 'feature', 'felt effect', 'narrative and discourse', 3, False,
     [('rising melody','anticipation'),('minor harmony','sadness'),('steady pulse','stability'),('sudden silence','suspense'),('repeated refrain','familiarity'),('syncopation','restlessness'),('soft dynamics','intimacy'),('brass fanfare','triumph'),('slow tempo','calm'),('dissonant chord','unease')]),
    ('visual composition choice to viewer effect', 'composition choice', 'viewer effect', 'narrative and discourse', 3, False,
     [('close framing','intimacy'),('wide framing','smallness'),('symmetry','order'),('asymmetry','tension'),('negative space','isolation'),('high contrast','drama'),('soft focus','dreaminess'),('diagonal lines','movement'),('low angle','power'),('muted palette','restraint')]),
    ('conversation gesture to relational repair', 'gesture', 'relational effect', 'social situations', 2, False,
     [('apology','acknowledgment'),('active listening','being heard'),('gentle question','openness'),('shared laugh','ease'),('thank-you note','appreciation'),('pause before replying','care'),('mirroring language','rapport'),('checking consent','respect'),('remembered detail','attentiveness'),('offering help','support')]),
    ('caregiving action to emotional need met', 'caregiving action', 'need met', 'social situations', 2, False,
     [('bringing soup','comfort'),('sitting quietly nearby','presence'),('making a call for someone','advocacy'),('keeping a promise','trust'),('lowering one\'s voice','calm'),('giving space','autonomy'),('walking together','companionship'),('organizing medicines','reliability'),('asking preferences','dignity'),('sharing a memory','continuity')]),
    ('ritual element to social function', 'ritual element', 'social function', 'social situations', 3, False,
     [('shared meal','belonging'),('moment of silence','respect'),('handshake','recognition'),('wedding vow','commitment'),('graduation walk','transition'),('birthday candle','celebration'),('farewell toast','closure'),('team chant','solidarity'),('welcome gift','hospitality'),('memorial name reading','remembrance')]),
    ('teaching move to learner effect', 'teaching move', 'learner effect', 'everyday planning', 2, False,
     [('worked example','orientation'),('leading question','discovery'),('timely feedback','correction'),('practice spacing','retention'),('analogy','transfer'),('encouragement','confidence'),('rubric','expectations'),('peer discussion','perspective'),('diagram','clarity'),('reflection prompt','metacognition')]),
    ('design choice to user feeling', 'design choice', 'user feeling', 'everyday planning', 2, False,
     [('rounded corners','approachability'),('clear labels','confidence'),('generous spacing','calm'),('tiny text','strain'),('consistent icons','familiarity'),('dark overlay','focus'),('progress indicator','reassurance'),('plain language','inclusion'),('soft color','gentleness'),('loud animation','urgency')]),
    ('ethical practice to trust-building effect', 'practice', 'trust effect', 'ethics', 3, False,
     [('disclosing uncertainty','honesty'),('asking consent','respect'),('sharing credit','fairness'),('admitting error','accountability'),('protecting privacy','safety'),('explaining reasons','transparency'),('avoiding conflicts','impartiality'),('listening first','dignity'),('following through','reliability'),('welcoming dissent','humility')]),
    ('narrative object to symbolic role', 'object', 'symbolic role', 'narrative and discourse', 3, False,
     [('broken watch','stopped time'),('locked door','barrier'),('bridge','connection'),('seed','potential'),('mirror','self-recognition'),('mask','hidden identity'),('lantern','guidance'),('thread','continuity'),('threshold','transition'),('letter','unspoken message')]),
    ('color to common expressive association', 'color', 'association', 'narrative and discourse', 2, False,
     [('red','urgency'),('blue','calm'),('green','renewal'),('gold','value'),('gray','ambiguity'),('white','simplicity'),('black','formality'),('purple','mystery'),('orange','warmth'),('silver','coolness')]),
    ('improvisation move to collaborative effect', 'move', 'collaborative effect', 'social situations', 3, False,
     [('yes-and response','building momentum'),('callback','shared memory'),('heightening','increased stakes'),('status shift','fresh tension'),('generous offer','partner support'),('active silence','room to respond'),('mirroring','coherence'),('reincorporation','unity'),('accepting mistake','playfulness'),('clear edit','closure')]),
    ('poetic device to reader effect', 'poetic device', 'reader effect', 'narrative and discourse', 3, False,
     [('enjambment','forward motion'),('caesura','pause'),('slant rhyme','subtle echo'),('assonance','musical texture'),('consonance','sound cohesion'),('image cluster','thematic density'),('plain diction','directness'),('repetition','emphasis'),('line break','surprise'),('white space','breathing room')]),
    ('conflict response to de-escalating function', 'response', 'de-escalating function', 'negotiation and interpersonal strategy', 3, False,
     [('naming the concern','validation'),('asking for interests','reframing'),('summarizing fairly','trust'),('lowering volume','calm'),('taking a break','cooling off'),('offering options','agency'),('separating person from problem','respect'),('using neutral criteria','legitimacy'),('acknowledging harm','repair'),('setting a boundary','safety')]),

    # --- abstract narrative and role-schema relations (2026-07 follow-up) ---
    ('fable role to strategic function', 'fable role', 'strategic function', 'narrative and discourse', 3, False,
     [('fox','clever exploiter'),('crow','flattered holder of value'),('tortoise','patient underdog'),('hare','overconfident favorite'),('wolf','predatory threat'),('shepherd boy','false alarm source'),('mouse','small rescuer'),('lion','powerful dependent'),('ant','prudent preparer'),('grasshopper','short-sighted consumer')]),
    ('allegorical figure to abstract force', 'symbolic figure', 'abstract force', 'narrative and discourse', 3, False,
     [('blindfolded judge','impartial justice'),('hourglass bearer','time pressure'),('masked stranger','hidden identity'),('crossroads traveler','choice under uncertainty'),('storm-bringer','disruption'),('bridge-builder','reconciliation'),('mirror-holder','self-knowledge'),('gatekeeper','selective access'),('torchbearer','guidance'),('weaver','interdependence')]),
    ('business-case role to strategic position', 'case role', 'strategic position', 'finance and business operations', 3, False,
     [('incumbent','defender of existing advantage'),('entrant','challenger seeking foothold'),('platform owner','rule setter'),('complementor','value amplifier'),('substitute product','outside threat'),('bottleneck supplier','constraint holder'),('lead customer','early signal'),('fast follower','imitator with timing advantage'),('loss leader','attention bait'),('switching-cost builder','retention architect')]),
    ('conflict archetype to event schema', 'story archetype', 'event schema', 'narrative and discourse', 3, False,
     [('underdog versus giant','asymmetric contest'),('betrayed ally','trust collapse'),('reluctant hero','forced responsibility'),('false prophet','misleading signal'),('sleeping giant','latent power awakened'),('pyrrhic victor','win with ruinous cost'),('kingmaker','decisive third party'),('poisoned gift','benefit carrying hidden cost'),('scapegoat','blame displaced from cause'),('broken truce','cooperation collapse')]),
    ('plot obstacle to strategic analogue', 'plot obstacle', 'strategic analogue', 'narrative and discourse', 3, False,
     [('locked gate','access barrier'),('riddle','information barrier'),('labyrinth','coordination complexity'),('storm','external shock'),('temptation','incentive conflict'),('curse','legacy constraint'),('deadline','time pressure'),('disguise','information asymmetry'),('traitor','insider risk'),('oracle ambiguity','uncertain forecast')]),
    ('coalition role to bargaining function', 'coalition role', 'bargaining function', 'negotiation and interpersonal strategy', 3, False,
     [('swing voter','decisive marginal supporter'),('bridge partner','cross-group connector'),('veto player','blocker with consent power'),('agenda setter','issue framer'),('honest broker','trusted mediator'),('spoiler','deal disruptor'),('outside patron','resource backer'),('core bloc','stable base'),('wavering ally','uncertain supporter'),('face-saving exit','dignified retreat path')]),
    ('governance role to systemic risk', 'governance role', 'systemic risk', 'law and regulation', 3, False,
     [('unchecked executive','overreach'),('captured regulator','biased enforcement'),('rubber-stamp board','weak oversight'),('opaque committee','accountability gap'),('conflicted adviser','skewed counsel'),('whistleblower','exposed hidden harm'),('independent auditor','verification backstop'),('emergency power','temporary authority risk'),('minority dissenter','ignored warning'),('public trustee','fiduciary duty')]),
    ('case-study signal to strategic implication', 'case-study signal', 'strategic implication', 'finance and business operations', 3, False,
     [('customer workaround','unmet need'),('falling renewal rate','weakening loyalty'),('supplier concentration','fragile dependency'),('copycat entrant','eroding differentiation'),('rising support tickets','product friction'),('employee workaround','process mismatch'),('pilot enthusiasm','adoption potential'),('silent churn','hidden dissatisfaction'),('regulatory inquiry','compliance exposure'),('partner hesitation','trust deficit')]),

    # --- formal rule and abstraction relations (2026-07 follow-up) ---
    ('letter string to successor-last-letter transform', 'string', 'transformed string', 'formal grammars and symbol systems', 3, False,
     [('abc','abd'),('klm','kln'),('pqr','pqs'),('uvw','uvx'),('def','deg'),('rst','rsu'),('mno','mnp'),('xyz','xya'),('ghi','ghj'),('bcd','bce')]),
    ('letter string to reversed order', 'string', 'reversal', 'formal grammars and symbol systems', 2, False,
     [('abc','cba'),('bdf','fdb'),('ace','eca'),('mnop','ponm'),('qrst','tsrq'),('wxy','yxw'),('abca','acba'),('lmno','onml'),('pqrs','srqp'),('defg','gfed')]),
    ('symbol pair to swapped pair', 'ordered pair', 'swapped pair', 'formal grammars and symbol systems', 2, False,
     [('A:B','B:A'),('red:blue','blue:red'),('left:right','right:left'),('input:output','output:input'),('key:value','value:key'),('cause:effect','effect:cause'),('north:south','south:north'),('parent:child','child:parent'),('sender:receiver','receiver:sender'),('source:sink','sink:source')]),
    ('abstract operation to inverse operation', 'operation', 'inverse operation', 'mathematics', 3, False,
     [('add three','subtract three'),('double','halve'),('rotate clockwise','rotate counterclockwise'),('encrypt','decrypt'),('push onto stack','pop from stack'),('open gate','close gate'),('compose with f','compose with f inverse'),('sort ascending','reverse sorted order'),('wrap','unwrap'),('encode','decode')]),
    ('matrix pattern to governing rule', 'matrix pattern', 'governing rule', 'logic puzzles', 3, False,
     [('row totals match','conservation across row'),('columns alternate colors','alternation by column'),('diagonal repeats shape','diagonal invariance'),('corners rotate arrows','quarter-turn progression'),('center equals overlap','intersection rule'),('right cell combines left and middle','composition rule'),('bottom row mirrors top row','reflection rule'),('third symbol differs from first two','exclusive-or rule'),('size increases left to right','monotone progression'),('missing cell completes cycle','cyclic completion')]),
    ('game position role to strategic concept', 'position role', 'strategic concept', 'logic puzzles', 3, False,
     [('fork','two threats at once'),('pin','immobilized defender'),('zugzwang','harmful forced move'),('sacrifice','short-term loss for leverage'),('tempo gain','initiative advantage'),('gambit','risk for development'),('blockade','mobility restriction'),('ladder threat','forced sequence'),('bluff','false strength signal'),('endgame opposition','space control')]),
    ('network property to robustness effect', 'network property', 'robustness effect', 'engineering and physical systems', 3, False,
     [('redundant path','survives one link failure'),('single cut vertex','fragile disconnection point'),('high clustering','local resilience'),('hub dominance','efficient but attack-sensitive'),('modularity','failure containment'),('long-range shortcut','shorter paths'),('load balancing','reduced hotspot'),('isolation boundary','blast-radius limit'),('mesh topology','many alternate routes'),('ring topology','two directions around break')]),
    ('flow law to analogous gradient', 'flow law', 'driving gradient', 'science', 3, False,
     [('heat conduction','temperature difference'),('diffusion','concentration difference'),('electric current','voltage difference'),('fluid flow','pressure difference'),('cash movement','return difference'),('traffic diversion','travel-time difference'),('osmosis','water-potential difference'),('labor migration','wage difference'),('information spread','attention difference'),('sediment transport','slope difference')]),
]

# Verbiage diversity: a large deck of problem phrasings. Each analogy is emitted
# once with ONE sampled phrasing (surface variety without duplicating answers).
Q_ANA_TRAIN = [
    "{demo}. Each pair shares one relation: {rel}. By the same relation, complete: {t0} : ?",
    "Relation held constant ({rel}): {demo}. What completes the pair {t0} : ?",
    "In every pair {aleft} maps to its {right} ({demo}). Give the {right} for {t0}.",
    "Study the pairs {demo}. They all follow the relation '{rel}'. Now do {t0} : ?",
    "Pattern ({rel}): {demo}. Extend it: {t0} -> ?",
    "Here {aleft} is paired with its {right}: {demo}. What should pair with {t0}?",
    "Each of these maps {aleft} to its {right}: {demo}. Fill in {t0} -> ?",
    "Following the single relation '{rel}' shown by {demo}, complete {t0} -> ?",
    "The examples {demo} share one rule ({rel}). Apply that rule to {t0}.",
    "Analogy: as in {demo} ({rel}), give the match for {t0}.",
    "Work out the {right} of {t0}, using the relation '{rel}' from {demo}.",
    "{demo}: one relation runs through all of these ({rel}). What completes {t0} : ?",
]
Q_ANA_EVAL = [
    "These pairs share exactly one relation ({rel}): {demo}. Supply the term for {t0}.",
]

# REGISTER deck: content-preserving voices/framings applied once per TRAIN item
# for verbiage diversity. Eval items stay in the plain register (stable
# benchmark). Each entry transforms the problem string without altering content.
def _reg_plain(p): return p
def _reg_memo(p): return "MEMO -- reasoning drill.\n" + p
def _reg_consider(p): return "Consider the following. " + p
def _reg_q(p): return "Q. " + p
def _reg_puzzle(p): return "A little puzzle: " + p
def _reg_colleague(p): return "A colleague asks: " + p
def _reg_exam(p): return "Exam item. " + p
def _reg_warmup(p): return "Warm-up. " + p
def _reg_fieldnote(p): return "[field notebook] " + p
def _reg_challenge(p): return "Analogy challenge -- " + p
def _reg_stepwise(p): return p + "\nReason it through step by step before answering."
def _reg_briefly(p): return p + " (Explain the mapping, then give the answer.)"
def _reg_ticket(p): return "TICKET #-- please resolve: " + p
def _reg_socratic(p): return "Let's reason together. " + p
REGISTERS = [_reg_plain, _reg_memo, _reg_consider, _reg_q, _reg_puzzle, _reg_colleague,
             _reg_exam, _reg_warmup, _reg_fieldnote, _reg_challenge, _reg_stepwise,
             _reg_briefly, _reg_ticket, _reg_socratic]

def _register(rng, text):
    return rng.choice(REGISTERS)(text)

def _mk_simple(rel, left, right, dom, diff, t0, ans, others, tmpl, split, prob=None):
    """Construct one single-relation item (caller supplies demos and phrasing)."""
    d1, d2, d3 = others
    demo = "; ".join(f"{a} -> {b}" for a, b in others)
    steps = [
        (f"Relation being transferred: {_art(left)} maps to its {right}. Demonstrated by {d1[0]} -> {d1[1]}, {d2[0]} -> {d2[1]}, {d3[0]} -> {d3[1]}.", "valid"),
        (f"Hold that relation fixed and change only the {left}: apply it to '{t0}'.", "valid"),
        (f"The {right} of {t0} is {ans}.", "valid"),
        (f"Check: the demonstrations and '{t0}' share the relation but differ in surface content, so the structure -- not word overlap -- fixes the answer as {ans}.", "valid"),
    ]
    if prob is None:
        prob = tmpl.format(demo=demo, rel=rel, t0=t0, left=left, right=right, aleft=_art(left))
    return dict(domain=dom, problem=prob, steps=steps, final=cap(str(ans)) + ".",
                difficulty=diff, vm="process_check", split=split,
                vd=f"Answer fills the '{right}' role of the '{rel}' relation for '{t0}'; mapping shown across the demonstrations, surface content varied.")

def _simple_pools(rel_tuple):
    """Return (train_pairs, eval_pairs) for a relation with disjoint answers."""
    relphrase, left, right, dom, diff, eval_only, pairs = rel_tuple
    if eval_only:
        return [], list(pairs)
    eval_pairs = pairs[-4:]
    eval_ans = {b for _, b in eval_pairs}
    train_pairs = [(a, b) for (a, b) in pairs[:-4] if b not in eval_ans]
    return train_pairs, eval_pairs

# ---- difficulty-4/5 trap kernels ----------------------------------------
# T1: series flow blockage with a local-damage surface distractor.
# (target system, medium, stage-noun, [n1,n2,n3], connector, failure verb,
#  distractor clause, distractor short label, domain)
T1_SKINS = [
    ("a building's climate duct", "conditioned air", "gallery", ["Aldermoor","Brightwell","Corvane"], "damper", "jammed shut",
     "a painting in the first gallery has visibly faded", "faded painting", "engineering and physical systems"),
    ("a series lighting string", "current", "fixture", ["Ash","Birch","Cedar"], "connector", "corroded open",
     "the first fixture's shade is cracked", "cracked shade", "engineering and physical systems"),
    ("a garden irrigation line", "water", "bed", ["North bed","Mid bed","South bed"], "valve", "seized closed",
     "a plant in the first bed has yellow leaves", "yellow leaves", "biology and ecology"),
    ("a factory conveyor feed", "parts", "station", ["Cutting","Welding","Painting"], "gate", "locked shut",
     "the first station's guard rail is scratched", "scratched rail", "engineering and physical systems"),
    ("a municipal water main", "supply", "district", ["Harbor","Midtown","Ridge"], "shutoff", "failed closed",
     "a hydrant in the first district is repainted", "repainted hydrant", "engineering and physical systems"),
    ("a data pipeline", "records", "stage", ["Ingest","Transform","Load"], "gate", "blocking",
     "the ingest dashboard has a cosmetic label typo", "label typo", "program behavior"),
    ("a district heating loop", "hot water", "block", ["Elm","Fir","Grove"], "isolation valve", "stuck shut",
     "a radiator in the first block is dented", "dented radiator", "engineering and physical systems"),
    ("a supply chain leg", "goods", "hub", ["Port","Depot","Store"], "checkpoint", "closed",
     "the port's signage is outdated", "outdated signage", "finance and business operations"),
]
def _mk_t1(skin, bi, split):
    """Construct one series-blockage item; bi is the blocked node index (0 or 1)."""
    sys_, medium, stage, nodes, conn, fail, distr, distr_lbl, dom = skin
    n1, n2, n3 = nodes
    blocked = nodes[bi]
    downstream = nodes[bi + 1:]
    upstream = nodes[:bi + 1]
    ds = ", ".join(downstream)
    us = ", ".join(upstream)
    us_sit, us_keep, us_stay = ("sits", "keeps", "stays") if len(upstream) == 1 else ("sit", "keep", "stay")
    ds_lie, ds_lose = ("lies", "loses") if len(downstream) == 1 else ("lie", "lose")
    prob = (f"Reference relation: in a chain where {medium} flows in series, "
            f"if one link is blocked, everything downstream of the block loses supply while everything upstream keeps it. "
            f"Present case: {sys_} carries {medium} in series through {n1}, then {n2}, then {n3}; "
            f"the {conn} at {blocked} has {fail}. Note also that {distr}. "
            f"By the same relation, which element plays the blocked-link role, and which {_plural(stage)} lose {medium}? "
            f"(One observation is a surface look-alike; decide by role, not appearance.)")
    steps = [
        (f"Relational template: {medium} flows in series through a chain; a single point-block cuts off everything downstream of it while upstream stages are unaffected.", "valid"),
        (f"Map roles: the series chain -> {n1}, then {n2}, then {n3}; the blocked link (the thing that stops onward flow) -> {_art(conn)} that has failed.", "valid"),
        (f"Locate the block in the target: the {conn} at {blocked} has {fail}, so it fills the blocked-link role.", "valid"),
        (f"Propagate along the structure: {us} {us_sit} at or before the block and {us_keep} {medium}; {ds} {ds_lie} downstream of it and {ds_lose} {medium}.", "valid"),
        (f"Reject the surface distractor: '{distr}' resembles damage, but the template concerns flow blockage, not local cosmetic harm, so it does not fill the blocked-link role.", "valid"),
        (f"Check: the mapping preserves the relation (a point-block cuts downstream flow); the answer follows from the series structure, not from surface resemblance.", "valid"),
    ]
    final = (f"The {conn} at {blocked} plays the blocked-link role; {ds} {ds_lose} {medium} while {us} {us_stay} supplied. "
             f"The {distr_lbl} is a surface look-alike, not the blockage.")
    return dict(domain=dom, problem=prob, steps=steps, final=final, difficulty=4,
                vm="process_check", split=split,
                vd="Answer fills the blocked-link role by the series-flow structure; the cosmetic observation is rejected as a surface look-alike.")

# T2: binding constraint = min(stock / per-unit need), NOT min(stock).
# (domain, register, item-noun, [r1,r2,r3], unit)
T2_SKINS = [
    ("chemistry", "a perfume bench", "bottle", ["essence X", "essence Y", "essence Z"], "drops"),
    ("biology and ecology", "a plant-breeding program", "cross", ["line L-1", "line L-2", "line L-3"], "pollen units"),
    ("finance and business operations", "a workshop", "gift box", ["ribbon", "cards", "beads"], "units"),
    ("engineering and physical systems", "an assembly cell", "kit", ["bolts", "brackets", "gaskets"], "pieces"),
    ("economics and markets", "a bakery", "cake", ["flour", "eggs", "sugar"], "grams"),
    ("chemistry", "a lab prep", "batch", ["reagent A", "reagent B", "reagent C"], "mL"),
]
def _plural(noun):
    """English plural for the simple item nouns used here."""
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and noun[-2:-1] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"

def _rand_t2_nums(rng):
    """Random (stocks, rates) where the binding resource (min stock/rate) is NOT
    the raw-scarcest (min stock) -- the trap. Verified before returning."""
    for _ in range(40):
        rates = [rng.randint(2, 12) for _ in range(3)]
        caps = [rng.randint(4, 20) for _ in range(3)]      # supported counts
        stocks = [c * r + rng.randint(0, r - 1) for c, r in zip(caps, rates)]  # stock >= cap*rate
        binding = min(range(3), key=lambda i: stocks[i] / rates[i])
        raw_min = min(range(3), key=lambda i: stocks[i])
        caps = [s // r for s, r in zip(stocks, rates)]
        if binding != raw_min and len({caps[binding]}) and caps[binding] == min(caps) \
           and sum(1 for c in caps if c == min(caps)) == 1:
            return stocks, rates
    return None

def _mk_t2(skin, stocks, rates, split):
    dom, reg, item, res, unit = skin
    items = _plural(item)
    caps = [s // r for s, r in zip(stocks, rates)]
    binding = min(range(3), key=lambda i: stocks[i] / rates[i])
    raw_min = min(range(3), key=lambda i: stocks[i])
    mincount = caps[binding]
    r1, r2, r3 = res; s1, s2, s3 = stocks; q1, q2, q3 = rates
    prob = (f"Rule for {reg}: the number of {items} you can complete is set by the resource with the smallest "
            f"stock-divided-by-per-{item} need -- which is frequently NOT the one you have least of overall. "
            f"Apply that same relation here. Each {item} needs {q1} {unit} of {r1}, {q2} of {r2}, and {q3} of {r3}; "
            f"in stock you have {s1} {unit} of {r1}, {s2} of {r2}, and {s3} of {r3}. "
            f"Which resource runs out first (caps the count), and how many {items} do you get? "
            f"(The resource you have least of by raw amount is a distractor.)")
    steps = [
        (f"Relation to transfer: the binding resource minimizes stock / per-{item} need, not raw stock.", "valid"),
        (f"Compute the supported count for each: {r1} = {s1}/{q1} = {caps[0]}; {r2} = {s2}/{q2} = {caps[1]}; {r3} = {s3}/{q3} = {caps[2]} {items}.", "valid"),
        (f"Name the distractor: by raw amount the scarcest is {res[raw_min]} ({stocks[raw_min]} {unit}), which the surface reading would wrongly pick.", "valid"),
        (f"By the intended relation the minimum supported count is {mincount}, at {res[binding]}, even though its raw stock ({stocks[binding]}) is not the smallest.", "valid"),
        (f"Check: {mincount} {items} consume {mincount*q1} of {r1} (<= {s1}), {mincount*q2} of {r2} (<= {s2}), {mincount*q3} of {r3} (<= {s3}); one more would need {(mincount+1)*rates[binding]} of {res[binding]}, exceeding {stocks[binding]}. Mapping confirmed.", "valid"),
    ]
    final = f"{cap(res[binding])} runs out first; you can complete {mincount} {items}."
    return dict(domain=dom, problem=prob, steps=steps, final=final, difficulty=5,
                vm="process_check", split=split,
                vd=f"Binding constraint recomputed: min(stock/need) = {mincount} at {res[binding]}; the raw-scarcest resource {res[raw_min]} is correctly rejected as a distractor.")

# T3: exception-to-the-exception reinstates the base rule; a relabeled item is
# a surface distractor that stays in the plain-exception role.
# (domain, all-noun, base outcome verb-phrase, exempt subclass, reinstating
#  condition, distractor condition)
T3_SKINS = [
    ("program behavior", "event", "journaled", "keep-alive event", "tagged 'reconcile'", "rerouted to a neighbouring node"),
    ("law and regulation", "shipment", "taxed at the border", "in-transit shipment", "unpacked for local sale", "relabeled with a new tracking code"),
    ("finance and business operations", "account", "charged a monthly fee", "student account", "overdrawn past the limit", "renamed after a profile update"),
    ("program behavior", "request", "logged", "health-check request", "carrying an error flag", "redirected to a mirror endpoint"),
    ("law and regulation", "building", "subject to the noise ordinance", "place of worship", "hosting a ticketed concert", "repainted a different colour"),
    ("biology and ecology", "cell", "flagged for apoptosis", "stem cell", "showing DNA damage", "relocated to another tissue"),
]
def _mk_t3(skin, split):
    dom, alln, base, exempt, reinstate, distr = skin
    prob = (f"Structural rule (three levels): every {alln} is {base}; {_art(exempt)} is an exception and is NOT {base}; "
            f"but {_art(exempt)} that is {reinstate} is {base} again -- the base outcome is reinstated at the third level. "
            f"Careful: some {exempt}s are merely {distr}, which superficially looks like the reinstated case but is not. "
            f"By the same three-level relation, which class of {alln} has the base outcome ({base}) reinstated?")
    steps = [
        (f"Relational template: a base rule R applies to all items; an exception E suspends R for a subclass; an exception-to-E reinstates R for a sub-subclass.", "valid"),
        (f"Map the levels: R = '{base}' (all {alln}s); E = {exempt}s (R suspended); the role to fill is the sub-subclass where R returns.", "valid"),
        (f"The sub-subclass whose base outcome returns is {exempt}s that are {reinstate}: they are {base} again.", "valid"),
        (f"Reject the distractor: {exempt}s that are merely {distr} still are NOT {base}; being relabeled or moved does not reinstate R, so they occupy role E, not the exception-to-E role.", "valid"),
        (f"Check: the answer fills the exact third-level role (base outcome reinstated); the relation matches even though the vocabulary differs from the source.", "valid"),
    ]
    final = f"The {exempt}s that are {reinstate} -- they have the base outcome ({base}) reinstated."
    return dict(domain=dom, problem=prob, steps=steps, final=final, difficulty=4,
                vm="process_check", split=split,
                vd="Answer fills the exception-to-exception role; the merely-relabeled subclass is rejected as a surface distractor that remains in role E.")

# T4: competing relations -- two rules each fit the FIRST shown pair, only one
# fits ALL pairs; transfer the consistent one. Parametric + numerically verified.
def _sq(n): return n * n
def _cube(n): return n * n * n
def _dbl(n): return 2 * n
def _tpl(n): return 3 * n
def _tri(n): return n * (n + 1) // 2  # running total 1..n
# (op, op-label, competitor, competitor-name, crossing x0 where op(x0)==comp(x0))
T4_PAIRS = [
    (_sq, "each number maps to its square", _tpl, "tripling", 3),
    (_sq, "each number maps to its square", _dbl, "doubling", 2),
    (_sq, "each number maps to its square", (lambda n: n + 2), "adding 2", 2),
    (_dbl, "each number maps to its double", (lambda n: n + 3), "adding 3", 3),
    (_tri, "each number maps to the running total 1..n", _dbl, "doubling", 3),
    (_cube, "each number maps to its cube", (lambda n: 7 * n - 6), "the rule 7n-6", 1),
]
T4_DOMAINS = ["mathematics", "science", "economics and markets", "program behavior", "algorithms and program analysis"]

def _rand_t4(rng):
    """Random competing-relations spec: (op,label,comp,comp_name,xs,tx,dom).
    x0 (the crossing point) is shown first so the competitor fits the first pair;
    the other shown x's diverge, exposing the competitor as wrong."""
    for _ in range(30):
        op, label, comp, cname, x0 = rng.choice(T4_PAIRS)
        pool = [x for x in range(2, 13) if x != x0 and op(x) != comp(x)]
        if len(pool) < 3:
            continue
        extras = rng.sample(pool, 2)
        xs = [x0] + sorted(extras)
        tx = rng.choice([x for x in pool if x not in extras])
        dom = rng.choice(T4_DOMAINS)
        if op(x0) != comp(x0):
            continue
        return op, label, comp, cname, xs, tx, dom
    return None

def _mk_t4(spec, split):
    op, label, comp, cname, xs, tx, dom = spec
    demo = "; ".join(f"{x} -> {op(x)}" for x in xs)
    ans = op(tx); wrong = comp(tx)
    diverge = next(x for x in xs[1:] if op(x) != comp(x))
    steps = [
        (f"Two relations each fit the first pair {xs[0]} -> {op(xs[0])}: the intended one ({label}) and a competitor ({cname}). Only one can be the shared relation.", "valid"),
        (f"Test the competitor against a later pair: {cname} predicts {comp(diverge)} for {diverge}, but the pair shows {op(diverge)}. So {cname} is not the shared relation.", "valid"),
        (f"The intended relation ({label}) fits every demonstration: {demo}.", "valid"),
        (f"Apply the intended relation to {tx}: the answer is {ans}.", "valid"),
        (f"Check: {wrong} (from the {cname} distractor) is rejected because that rule failed an earlier pair; {ans} is the value under the relation that holds throughout.", "valid"),
    ]
    prob = (f"Each pair follows one and the same relation, and more than one rule appears to fit at first glance: {demo}. "
            f"The intended relation is: {label}. Ignore the competing pattern that only fits the first pair. "
            f"What value completes {tx} -> ?")
    return dict(domain=dom, problem=prob, steps=steps, final=f"{ans}.", difficulty=5,
                vm="process_check", split=split,
                vd=f"Intended relation verified on all demos; competitor '{cname}' fails at {diverge} ({comp(diverge)} vs {op(diverge)}); answer {ans} = op({tx}).")

# ---- CROSS-DOMAIN ISOMORPHISM ENGINE ------------------------------------
# The core of cross-domain transfer. Each STRUCTURE is one abstract relational
# schema with a single mappable ROLE; it is instantiated across many unrelated
# domains. Items ask "what plays the same role in system B that X plays in
# system A", so the SOURCE and TARGET are always different domains and only the
# structure -- never shared vocabulary -- yields the answer. Fillers are chosen
# so the answer word does not appear in the source text (a leakage guard also
# enforces this at build time).
#
# Split: each structure reserves some instances (ev=True) as EVAL TARGETS, in
# systems/domains never used as a train target for that structure, with answers
# disjoint from every train answer. Sources are always drawn from train
# instances, so eval measures "apply a structure learned in known domains to a
# new domain" -- transfer, not recall.
#
# inst = (domain, system, filler[=answer], context, distractor|None, ev)
STRUCTURES = [
    dict(name="flow driven by a potential difference",
         role="supplies the driving force that moves the medium through the system",
         diff=3, insts=[
            ("engineering and physical systems", "an electrical circuit", "the battery", "it pushes current through the components", "a resistor", False),
            ("engineering and physical systems", "a plumbing loop", "the pump", "it pushes water through the pipes", "a valve", False),
            ("biology and ecology", "the body's circulation", "the heart", "it pushes blood through the vessels", "a capillary", False),
            ("economics and markets", "a two-region market", "the price gap", "it moves goods from the cheaper region to the dearer one", "a warehouse", False),
            ("mechanical / systems troubleshooting", "a hydraulic brake system", "the master cylinder", "it pushes brake fluid to the calipers", "a brake pad", False),
            ("science", "a river system", "the elevation drop (gravity)", "it moves water downhill", "a boulder", True),
            ("chemistry", "diffusion across a membrane", "the concentration gradient", "it drives molecules from high to low concentration", "the membrane", True),
         ]),
    dict(name="negative feedback holding a variable near a setpoint",
         role="acts as the regulator that holds the controlled variable near a target",
         diff=3, insts=[
            ("engineering and physical systems", "a heated room", "the thermostat", "it holds room temperature near a set value", "the heater", False),
            ("biology and ecology", "the human body", "the hypothalamus", "it holds core temperature near 37C", "the skin", False),
            ("economics and markets", "a national economy", "the central bank", "it holds the inflation rate near a target", "a commercial bank", False),
            ("engineering and physical systems", "a car on cruise control", "the cruise controller", "it holds vehicle speed near the set speed", "the engine", False),
            ("finance and business operations", "a warehouse", "the reorder policy", "it holds the stock level near a target", "a shelf", False),
            ("biology and ecology", "the bloodstream", "the pancreas", "it holds blood sugar near a healthy level", "the liver", True),
            ("incident and root-cause analysis", "an autoscaled service", "the autoscaler", "it holds CPU utilization near a target", "a server", True),
         ]),
    dict(name="equilibrium of two opposing influences",
         role="is the opposing influence that balances the first",
         diff=3, opposing=True, insts=[
            # for this structure filler = the OPPOSING force; context names the FIRST force
            ("economics and markets", "a market's price", "supply from sellers", "demand from buyers pushes the price up", "a warehouse", False),
            ("chemistry", "a reversible reaction", "the reverse reaction", "the forward reaction builds up product", "a catalyst", False),
            ("biology and ecology", "a population's size", "the death rate", "the birth rate pushes the size up", "the habitat", False),
            ("engineering and physical systems", "a load hanging on a spring", "the spring's restoring force", "gravity pulls the load down", "the ceiling", False),
            ("science", "a floating object", "buoyancy pushing up", "gravity pulls the object down", "the water", False),
            ("negotiation and interpersonal strategy", "a price negotiation", "the buyer's walk-away limit", "the seller's asking price pushes the number up", "the meeting room", False),
            ("economics and markets", "the labor market's wage", "labor supply from workers", "employer demand pushes the wage up", "a factory", True),
            ("science", "a planet's stable size", "outward radiation pressure", "gravity pulls the gas inward", "a moon", True),
         ]),
    dict(name="a single bottleneck caps the whole system's output",
         role="is the limiting element that caps the whole system's output",
         diff=3, insts=[
            ("chemistry", "a chemical reaction", "the limiting reagent", "it runs out first and caps how much product forms", "the excess reagent", False),
            ("biology and ecology", "plant growth in a field", "the scarcest nutrient", "it caps the yield no matter how much else is plentiful", "abundant sunlight", False),
            ("program behavior", "a processing pipeline", "the slowest stage", "it caps the end-to-end throughput", "a fast stage", False),
            ("finance and business operations", "a project schedule", "the critical path", "it sets the earliest finish date", "a task with slack", False),
            ("everyday planning", "a dinner-party timeline", "the longest-cooking dish", "it sets when dinner can be served", "a quick side salad", False),
            ("logic puzzles", "a chain of deductions", "the weakest inference", "it caps how certain the final conclusion is", "a rock-solid step", False),
            ("economics and markets", "a supply chain", "the tightest supplier", "it caps how much can be delivered", "a plentiful input", True),
            ("medicine-style diagnosis", "oxygen delivery to tissue", "the narrowest artery", "it caps the flow that can reach the tissue", "a wide vein", True),
         ]),
    dict(name="a buffer absorbs shocks to stabilize a variable",
         role="acts as the buffer that absorbs shocks to keep the variable steady",
         diff=3, insts=[
            ("chemistry", "a buffered solution", "the weak acid-base pair", "it absorbs added acid or base to keep pH steady", "the strong acid", False),
            ("finance and business operations", "a household budget", "the emergency fund", "it absorbs income shocks to keep spending steady", "the mortgage", False),
            ("biology and ecology", "a wetland by a river", "the wetland's storage", "it absorbs flood surges to keep downstream levels steady", "a bridge", False),
            ("engineering and physical systems", "a power grid", "the battery bank", "it absorbs demand spikes to keep voltage steady", "a transformer", False),
            ("program behavior", "a video stream", "the playback buffer", "it absorbs network dips to keep playback smooth", "the codec", False),
            ("medicine-style diagnosis", "the bloodstream's pH", "the bicarbonate system", "it absorbs metabolic acid to keep pH steady", "a red blood cell", True),
            ("economics and markets", "a commodity market", "the strategic reserve", "it absorbs supply shocks to keep prices steady", "an exchange", True),
         ]),
    dict(name="a gatekeeper selectively admits some and blocks the rest",
         role="acts as the gatekeeper that selectively admits some and blocks the rest",
         diff=3, insts=[
            ("biology and ecology", "a living cell", "the cell membrane", "it lets select ions in and keeps others out", "the nucleus", False),
            ("engineering and physical systems", "a private network", "the firewall", "it admits allowed traffic and blocks the rest", "a server", False),
            ("law and regulation", "a national border", "customs and visa control", "it admits eligible travelers and turns others away", "the airport", False),
            ("medicine-style diagnosis", "the brain's blood supply", "the blood-brain barrier", "it admits select molecules and blocks most others", "a neuron", False),
            ("narrative and discourse", "an editorial desk", "the editor", "it admits publishable pieces and cuts the rest", "the newsroom", False),
            ("program behavior", "a web API", "the authentication layer", "it admits valid requests and rejects the rest", "the database", True),
            ("social situations", "an exclusive club", "the door policy", "it admits members and turns others away", "the bar", True),
         ]),
    dict(name="a whole built up from many repeated small parts",
         role="is the whole that its basic repeated parts compose",
         diff=3, insts=[
            ("formal grammars and symbol systems", "language", "a sentence", "words compose it", "a letter", False),
            ("chemistry", "matter", "a molecule", "atoms compose it", "an electron", False),
            ("engineering and physical systems", "masonry", "a wall", "bricks compose it", "a window", False),
            ("program behavior", "a software system", "a program", "functions compose it", "a variable", False),
            ("social situations", "a population", "a community", "individual people compose it", "a rule", False),
            ("biology and ecology", "a body's structure", "a tissue", "cells compose it", "an organ", True),
            ("narrative and discourse", "a story", "a chapter", "scenes compose it", "a title", True),
         ]),
    dict(name="compounding: output is fed back to enlarge the base that produces it",
         role="is the reinvested output that feeds back to enlarge the base",
         diff=3, insts=[
            ("finance and business operations", "a savings account", "the reinvested interest", "it is added to the balance so future interest grows", "the account fee", False),
            ("program behavior", "a viral post", "each reshare", "it exposes more people who reshare in turn", "a single like", False),
            ("chemistry", "a nuclear chain reaction", "each emitted neutron", "it triggers a further reaction", "the container wall", False),
            ("economics and markets", "a growing firm", "reinvested profit", "it expands capacity so output rises", "a temporary grant", False),
            ("biology and ecology", "a growing population", "each new cohort", "it matures into breeders so growth accelerates", "a passing predator", True),
            ("science", "an avalanche", "each dislodged mass", "it dislodges still more snow", "a distant peak", True),
         ]),
    dict(name="a small input controls a much larger output (amplification)",
         role="is the small control that governs a much larger output",
         diff=3, insts=[
            ("engineering and physical systems", "a transistor circuit", "the base current", "a tiny current controls a large one", "the load resistor", False),
            ("economics and markets", "a leveraged trade", "the margin deposit", "a small stake controls a large position", "the brokerage", False),
            ("biology and ecology", "an enzyme reaction", "the enzyme", "a trace amount drives a large conversion", "the solvent", False),
            ("engineering and physical systems", "a lever", "the effort at the long arm", "a small force moves a large load", "the fulcrum block", False),
            ("social situations", "a rumor's spread", "the first influential sharer", "one well-placed voice moves a large crowd", "the venue", False),
            ("program behavior", "a feature flag", "the config toggle", "one setting switches behavior for all users", "a log line", True),
            ("medicine-style diagnosis", "a hormonal signal", "the releasing hormone", "a minute dose triggers a large downstream response", "a blood vessel", True),
         ]),
    dict(name="one trigger sets off a self-propagating cascade",
         role="is the initial trigger that sets off the whole cascade",
         diff=3, insts=[
            ("finance and business operations", "a bank run", "the first mass withdrawal", "it spooks others into withdrawing too", "the vault", False),
            ("biology and ecology", "a trophic collapse", "the loss of the keystone species", "it topples dependent species in turn", "a rock pool", False),
            ("engineering and physical systems", "a power-grid blackout", "the first overloaded line tripping", "it shifts load and trips the next", "a substation fence", False),
            ("science", "a chain reaction", "the first fission event", "it releases neutrons that split more nuclei", "the reactor casing", False),
            ("program behavior", "a cascading outage", "the first overloaded service failing", "its retries overwhelm the next service", "a dashboard", False),
            ("medicine-style diagnosis", "an allergic cascade", "the first mast-cell release", "it recruits more cells to release in turn", "the skin surface", True),
            ("social situations", "a stampede", "the first panicked runner", "their motion triggers the crowd to bolt", "the exit sign", True),
         ]),
    dict(name="a threshold must be crossed before any effect occurs",
         role="is the threshold that must be crossed before the effect switches on",
         diff=3, insts=[
            ("biology and ecology", "a firing neuron", "the action-potential threshold", "no spike until the voltage crosses it", "the axon sheath", False),
            ("chemistry", "a reaction needing activation", "the activation energy", "no reaction until the energy barrier is crossed", "the beaker", False),
            ("economics and markets", "a progressive tax bracket", "the bracket cutoff", "the higher rate applies only past it", "the tax form", False),
            ("engineering and physical systems", "a static-friction start", "the breakaway force", "the object stays put until force exceeds it", "the floor", False),
            ("law and regulation", "a legal filing requirement", "the reporting threshold", "no duty to file until the amount exceeds it", "the office", False),
            ("medicine-style diagnosis", "a drug's effect", "the minimum effective dose", "no clinical effect until the dose crosses it", "the pill bottle", True),
            ("social situations", "a protest turning into a movement", "the critical mass of participants", "little happens until the count crosses it", "the plaza", True),
         ]),
    dict(name="a catalyst speeds a process without being consumed by it",
         role="is the catalyst that speeds the process without being used up",
         diff=3, insts=[
            ("chemistry", "a catalyzed reaction", "the catalyst", "it speeds the reaction and is left unchanged", "a reactant", False),
            ("economics and markets", "regional commerce", "the shared infrastructure", "it enables far more trade without being traded", "the goods", False),
            ("biology and ecology", "a metabolic step", "the enzyme", "it accelerates the step and is recovered intact", "the substrate", False),
            ("social situations", "a productive meeting", "the skilled facilitator", "they speed agreement without being party to it", "the agenda", False),
            ("program behavior", "a fast build", "the compiler cache", "it speeds rebuilds and is not part of the output", "the source file", False),
            ("finance and business operations", "a closed deal", "the broker", "they speed the transaction without owning the asset", "the contract", True),
            ("science", "faster nucleation", "the seed crystal", "it speeds crystallization and stays a tiny fraction", "the solution", True),
         ]),

    dict(name="signal separated from background noise",
         role="is the meaningful signal that must be distinguished from background noise",
         diff=3, insts=[
            ("incident and root-cause analysis", "a monitoring dashboard", "the sustained error-rate spike", "it indicates the outage amid routine jitter", "a random graph wiggle", False),
            ("medicine-style diagnosis", "a clinical exam", "the focal neurologic deficit", "it points to the lesion amid vague complaints", "a mild headache", False),
            ("science", "a telescope observation", "the repeated transit dip", "it reveals a planet amid sensor noise", "a cosmic-ray hit", False),
            ("finance and business operations", "fraud review", "the repeated mismatched shipping address", "it marks suspicious activity amid ordinary purchases", "a large but typical order", False),
            ("biology and ecology", "bird monitoring", "the species-specific call", "it identifies the bird amid wind sounds", "rustling leaves", False),
            ("law and regulation", "evidence review", "the authenticated timestamp", "it proves timing amid irrelevant documents", "a cover-sheet logo", True),
         ]),
    dict(name="redundancy provides a backup when the primary fails",
         role="is the backup that preserves function when the primary fails",
         diff=3, insts=[
            ("engineering and physical systems", "an aircraft control system", "the redundant actuator", "it moves the surface if the main actuator fails", "the paint", False),
            ("program behavior", "a storage cluster", "the replica", "it serves data if the primary node fails", "the dashboard", False),
            ("biology and ecology", "paired kidneys", "the second kidney", "it maintains filtration if one kidney fails", "the ureter", False),
            ("finance and business operations", "payment processing", "the alternate processor", "it takes transactions if the main processor is down", "the receipt template", False),
            ("everyday planning", "travel planning", "the backup route", "it still reaches the destination if the main road closes", "the suitcase", False),
            ("incident and root-cause analysis", "disaster recovery", "the standby region", "it carries traffic if the primary region fails", "the status page", True),
         ]),
    dict(name="division of labor assigns specialized roles",
         role="is the specialized unit assigned to one part of the work",
         diff=3, insts=[
            ("biology and ecology", "a bee colony", "the forager bee", "it gathers food while others perform different tasks", "the hive wall", False),
            ("program behavior", "a compiler", "the parser", "it specializes in syntax analysis before later stages", "the progress bar", False),
            ("finance and business operations", "an operations team", "the accounts-payable clerk", "it specializes in paying vendors", "the office printer", False),
            ("engineering and physical systems", "an assembly line", "the welding station", "it specializes in joining parts", "the warehouse sign", False),
            ("social situations", "a project group", "the note-taker", "they specialize in recording decisions", "the meeting snacks", False),
            ("law and regulation", "a trial team", "the evidence clerk", "they specialize in managing exhibits", "the courtroom bench", True),
         ]),
    dict(name="scarcity pressure changes allocation",
         role="is the scarcity pressure that pushes allocation or price upward",
         diff=3, insts=[
            ("economics and markets", "a housing market", "limited available homes", "it pushes prices upward when buyers compete", "fresh paint", False),
            ("finance and business operations", "warehouse inventory", "low stock", "it pushes reorder priority upward", "a barcode label", False),
            ("medicine-style diagnosis", "emergency triage", "few available ICU beds", "it raises the admission threshold", "the waiting-room chairs", False),
            ("everyday planning", "restaurant reservations", "few open tables", "it pushes booking urgency upward", "the menu design", False),
            ("incident and root-cause analysis", "on-call staffing", "too few responders", "it pushes queue time upward", "the chat avatar", False),
            ("negotiation and interpersonal strategy", "salary bargaining", "competing offers", "it raises the candidate's bargaining power", "the conference room", True),
         ]),
    dict(name="inertia resists a change in state",
         role="is the resistance that keeps the system in its current state",
         diff=3, insts=[
            ("engineering and physical systems", "a moving cart", "the cart's mass", "it resists changes in velocity", "the handle paint", False),
            ("economics and markets", "consumer habits", "switching costs", "they keep buyers with the old supplier", "a billboard", False),
            ("program behavior", "a legacy codebase", "backward compatibility", "it resists changing interfaces", "a comment", False),
            ("social situations", "a committee routine", "habitual procedure", "it keeps the group following the old process", "the room number", False),
            ("biology and ecology", "an ecosystem state", "established species interactions", "they resist a shift to a new state", "a trail marker", False),
            ("law and regulation", "case law", "precedent", "it resists sudden doctrinal change", "the courthouse steps", True),
         ]),
    dict(name="phase transition after a boundary is crossed",
         role="is the boundary where gradual change produces a new state",
         diff=3, insts=[
            ("chemistry", "water boiling", "the boiling point", "crossing it changes liquid water to vapor", "the beaker label", False),
            ("biology and ecology", "a lake eutrophication shift", "the nutrient tipping point", "crossing it flips the lake to algal dominance", "a dock", False),
            ("economics and markets", "a bank run", "the confidence threshold", "crossing it changes waiting depositors into withdrawing depositors", "the teller window", False),
            ("program behavior", "a load balancer", "the saturation point", "crossing it changes stable service into queue growth", "a log banner", False),
            ("social situations", "a crowd mood", "the critical mass", "crossing it changes scattered concern into collective action", "the plaza fountain", False),
            ("medicine-style diagnosis", "fever onset", "the hypothalamic setpoint change", "crossing it shifts the body into heat-conserving behavior", "a blanket", True),
         ]),
    dict(name="error correction uses redundancy to restore the intended message",
         role="is the redundant check that detects or corrects the error",
         diff=3, insts=[
            ("formal grammars and symbol systems", "a parity-coded message", "the parity bit", "it reveals a single-bit error", "the envelope", False),
            ("program behavior", "a network packet", "the checksum", "it detects corruption in transit", "the port number", False),
            ("biology and ecology", "DNA replication", "proofreading polymerase", "it corrects mispaired bases", "the sugar backbone", False),
            ("finance and business operations", "bookkeeping", "the reconciliation", "it catches mismatched ledger entries", "the office chair", False),
            ("law and regulation", "appellate review", "the appeal", "it can correct legal error in a judgment", "the courtroom seal", False),
            ("science", "a replicated experiment", "the independent replication", "it catches a spurious result", "the lab coat", True),
         ]),
    dict(name="incentive shaping changes behavior by changing payoffs",
         role="is the reward or penalty that steers behavior toward the desired action",
         diff=3, insts=[
            ("economics and markets", "a carbon policy", "the emissions tax", "it makes emitting more costly", "the smokestack", False),
            ("social situations", "classroom participation", "the participation credit", "it rewards speaking up", "the whiteboard", False),
            ("program behavior", "reinforcement learning", "the reward function", "it steers the agent's choices", "the training log", False),
            ("finance and business operations", "sales compensation", "the commission", "it rewards completed sales", "the desk phone", False),
            ("ethics", "professional conduct", "the conflict-of-interest penalty", "it discourages biased decisions", "the office badge", False),
            ("law and regulation", "traffic enforcement", "the speeding fine", "it discourages unsafe speed", "the road sign", True),
         ]),
    dict(name="slack absorbs variation before a bottleneck binds",
         role="is the slack capacity that absorbs variation before output is capped",
         diff=3, insts=[
            ("finance and business operations", "a project schedule", "float time", "it absorbs delay before the finish date moves", "the project logo", False),
            ("program behavior", "a message queue", "unused queue capacity", "it absorbs bursts before producers block", "a log line", False),
            ("engineering and physical systems", "a bridge design", "the safety margin", "it absorbs extra load before failure", "the paint color", False),
            ("everyday planning", "an itinerary", "free time between appointments", "it absorbs traffic delay before the next appointment is missed", "the coffee cup", False),
            ("medicine-style diagnosis", "lung function", "respiratory reserve", "it absorbs exertion before oxygenation fails", "the stethoscope", False),
            ("incident and root-cause analysis", "server capacity", "headroom", "it absorbs demand spikes before saturation", "the dashboard title", True),
         ]),
    dict(name="hierarchy passes inherited properties downward",
         role="is the parent category whose properties are inherited by children",
         diff=3, insts=[
            ("program behavior", "an object-oriented class tree", "the superclass", "its methods are inherited by subclasses", "an instance field", False),
            ("biology and ecology", "taxonomy", "the genus", "its traits help classify species below it", "a habitat note", False),
            ("law and regulation", "legal authority", "the higher court precedent", "it binds lower courts", "the courthouse map", False),
            ("formal grammars and symbol systems", "a grammar", "the nonterminal category", "its production options generate lower forms", "a punctuation mark", False),
            ("finance and business operations", "account hierarchy", "the parent account", "its category contains subsidiary accounts", "a receipt", False),
            ("logic puzzles", "a category rule", "the broad class", "membership in it transfers constraints to members", "a token color", True),
         ]),
    dict(name="host-parasite relation extracts benefit while imposing cost",
         role="is the parasite-like dependent that benefits while costing the host",
         diff=3, insts=[
            ("biology and ecology", "a tick on a deer", "the tick", "it feeds while costing the deer blood and irritation", "the grass", False),
            ("program behavior", "malware on a computer", "the cryptominer", "it uses compute while slowing the host", "the wallpaper", False),
            ("economics and markets", "rent seeking", "the toll collector", "it extracts value without adding matching output", "the road paint", False),
            ("social situations", "a group project", "the free rider", "they take credit while others do the work", "the slide template", False),
            ("finance and business operations", "a predatory fee scheme", "the hidden fee", "it extracts revenue while burdening the customer", "the brand color", False),
            ("medicine-style diagnosis", "intestinal infection", "the tapeworm", "it draws nutrients while harming the patient", "the clinic door", True),
         ]),
    dict(name="encode then decode preserves content across representation",
         role="is the decoder that recovers the original content from the representation",
         diff=3, insts=[
            ("formal grammars and symbol systems", "Morse communication", "the Morse reader", "it recovers letters from dots and dashes", "the telegraph key", False),
            ("program behavior", "compressed data", "the decompressor", "it recovers bytes from the compressed stream", "the filename", False),
            ("biology and ecology", "gene expression", "the ribosome", "it reads codons to build the protein", "the cell wall", False),
            ("law and regulation", "coded statute references", "the legal citation parser", "it recovers the cited authority from the shorthand", "the binder clip", False),
            ("narrative and discourse", "symbolic allegory", "the interpretive key", "it recovers the intended meaning from symbols", "the page number", False),
            ("engineering and physical systems", "radio reception", "the demodulator", "it recovers the message from the carrier", "the antenna mast", True),
         ]),
    dict(name="map is a simplified representation of territory",
         role="is the representation that guides action without being the thing itself",
         diff=3, insts=[
            ("everyday planning", "city navigation", "the street map", "it represents roads without being the city", "a traffic cone", False),
            ("science", "a climate model", "the simulation", "it represents climate processes without being the planet", "a thermometer", False),
            ("finance and business operations", "a budget", "the spreadsheet forecast", "it represents expected cash flows without being the cash", "a receipt", False),
            ("program behavior", "database design", "the schema diagram", "it represents tables without being the database", "a server fan", False),
            ("law and regulation", "statutory summary", "the compliance checklist", "it represents duties without being the statute", "a filing cabinet", False),
            ("medicine-style diagnosis", "an anatomical chart", "the chart", "it represents the body without being the patient", "the exam table", True),
         ]),
    dict(name="delayed feedback can overshoot and oscillate",
         role="is the feedback delay that causes correction to arrive late",
         diff=3, insts=[
            ("economics and markets", "inventory ordering", "shipping lead time", "it delays replenishment so orders overshoot demand", "the shelf label", False),
            ("engineering and physical systems", "a shower temperature adjustment", "pipe delay", "it delays hot water so the user overcorrects", "the tile", False),
            ("program behavior", "autoscaling", "metric lag", "it delays scaling decisions so capacity oscillates", "a log line", False),
            ("biology and ecology", "predator-prey cycles", "reproductive lag", "it delays population response and permits oscillation", "a rock", False),
            ("finance and business operations", "quarterly planning", "reporting lag", "it delays corrections until after conditions changed", "the report cover", False),
            ("medicine-style diagnosis", "insulin dosing", "absorption delay", "it delays effect and can cause overcorrection", "the syringe cap", True),
         ]),
    dict(name="diminishing returns shrink each added unit's benefit",
         role="is the saturation limit that makes each additional input less useful",
         diff=3, insts=[
            ("economics and markets", "labor on a fixed field", "fixed land", "it limits how much each extra worker adds", "the farm gate", False),
            ("finance and business operations", "advertising spend", "audience saturation", "it makes later ads reach fewer new buyers", "the invoice", False),
            ("medicine-style diagnosis", "drug dosing", "receptor saturation", "it makes extra dose add less effect", "the pill bottle", False),
            ("program behavior", "parallel speedup", "the serial portion", "it limits the gain from adding processors", "the keyboard", False),
            ("everyday planning", "studying for an exam", "mental fatigue", "it makes each extra hour less productive", "the notebook cover", False),
            ("biology and ecology", "plant fertilization", "another limiting nutrient", "it makes extra nitrogen add little growth", "the fence", True),
         ]),
    dict(name="common-pool overuse depletes a shared resource",
         role="is the shared resource depleted by individually rational use",
         diff=3, insts=[
            ("ethics", "a commons dilemma", "the shared pasture", "each herder benefits from adding animals while the pasture degrades", "the fence post", False),
            ("economics and markets", "open-access fishing", "the fish stock", "each boat catches more while the stock declines", "the dock", False),
            ("program behavior", "unthrottled API access", "the shared rate limit", "each client benefits from more calls while service degrades", "a request header", False),
            ("biology and ecology", "groundwater pumping", "the aquifer", "each well draws water while the water table falls", "the pump handle", False),
            ("social situations", "shared kitchen cleanup", "the clean kitchen", "each person saves effort by not cleaning while the kitchen deteriorates", "a coffee mug", False),
            ("law and regulation", "pollution control", "the airshed", "each emitter benefits while air quality worsens", "a smokestack label", True),
         ]),
    dict(name="principal-agent misalignment creates hidden-action risk",
         role="is the agent whose incentives can diverge from the principal's goal",
         diff=3, insts=[
            ("finance and business operations", "sales management", "the commissioned salesperson", "they may maximize commission rather than long-term fit", "the brochure", False),
            ("law and regulation", "corporate governance", "the manager", "they may pursue private benefits over shareholder value", "the letterhead", False),
            ("medicine-style diagnosis", "delegated care", "the contractor clinic", "it may cut corners if monitoring is weak", "the waiting-room poster", False),
            ("program behavior", "automated bidding", "the bidding bot", "it may optimize the proxy metric rather than business value", "the server rack", False),
            ("social situations", "hiring a house sitter", "the house sitter", "they may exert less care when unobserved", "the key ring", False),
            ("ethics", "stewardship", "the entrusted steward", "they may use delegated power for personal gain", "the seal", True),
         ]),
    dict(name="triage prioritizes scarce attention by urgency and severity",
         role="is the triage rule that orders cases by urgency rather than arrival order",
         diff=3, insts=[
            ("medicine-style diagnosis", "emergency department intake", "the triage protocol", "it sends the most urgent patients first", "the magazine rack", False),
            ("incident and root-cause analysis", "alert response", "the severity policy", "it handles customer-impacting pages before minor alerts", "the emoji reaction", False),
            ("law and regulation", "court docket management", "the emergency motion rule", "it moves urgent requests ahead of routine filings", "the filing stamp", False),
            ("finance and business operations", "support queue", "the priority rubric", "it handles high-impact tickets first", "the office plant", False),
            ("everyday planning", "errand planning", "the deadline-based ordering", "it handles time-critical errands first", "the shopping bag", False),
            ("program behavior", "operating-system scheduling", "the priority scheduler", "it runs high-priority tasks before background work", "the desktop wallpaper", True),
         ]),
    dict(name="boundary object coordinates groups with different internal views",
         role="is the shared artifact that lets groups coordinate without identical perspectives",
         diff=3, insts=[
            ("engineering and physical systems", "construction coordination", "the blueprint", "it lets architects and builders coordinate", "the hard hat", False),
            ("program behavior", "API integration", "the interface contract", "it lets teams build separately against the same expectations", "the office chair", False),
            ("medicine-style diagnosis", "care handoff", "the discharge summary", "it lets clinicians coordinate across departments", "the hallway sign", False),
            ("law and regulation", "settlement talks", "the term sheet", "it lets parties negotiate from a shared outline", "the conference table", False),
            ("science", "interdisciplinary project", "the shared dataset", "it lets teams analyze from different methods", "the lab badge", False),
            ("social situations", "community planning", "the neighborhood map", "it lets residents and officials discuss the same area", "the folding chair", True),
         ]),
    dict(name="keystone element has disproportionate system influence",
         role="is the keystone element whose removal disproportionately changes the system",
         diff=3, insts=[
            ("biology and ecology", "a kelp forest", "the sea otter", "it controls urchins and preserves kelp", "a pebble", False),
            ("engineering and physical systems", "an arch", "the keystone block", "it locks the arch into compression", "a decorative carving", False),
            ("program behavior", "a distributed service", "the identity provider", "it enables many services to authenticate users", "a CSS file", False),
            ("finance and business operations", "a production line", "the sole certified operator", "they keep the specialized step running", "the break-room poster", False),
            ("law and regulation", "a regulatory scheme", "the enabling statute", "it gives authority to many rules", "the docket cover", False),
            ("narrative and discourse", "a mystery plot", "the hidden motive", "it explains many otherwise separate clues", "the chapter title", True),
         ]),
    dict(name="commitment device prevents later temptation from changing the plan",
         role="is the commitment device that makes reversal costly or impossible",
         diff=3, insts=[
            ("economics and markets", "saving money", "the locked savings account", "it makes impulsive spending harder", "the bank logo", False),
            ("everyday planning", "exercise planning", "the prepaid class", "it makes skipping costly", "the water bottle", False),
            ("negotiation and interpersonal strategy", "public bargaining stance", "the public pledge", "it makes backing down costly", "the microphone stand", False),
            ("program behavior", "database migration", "the immutable migration log", "it prevents silent reversal of applied changes", "the terminal color", False),
            ("ethics", "conflict avoidance", "the blind trust", "it prevents later self-dealing choices", "the office door", False),
            ("law and regulation", "plea agreement", "the signed stipulation", "it locks in terms unless conditions fail", "the courthouse flag", True),
         ]),
    dict(name="negative externality imposes a cost on outsiders",
         role="is the unpriced harm imposed on parties outside the transaction",
         diff=3, insts=[
            ("economics and markets", "factory production", "the air pollution", "it harms neighbors who are not part of the sale", "the invoice", False),
            ("ethics", "loud late-night music", "the lost sleep", "it burdens neighbors who did not choose the activity", "the speaker brand", False),
            ("engineering and physical systems", "road traffic", "the congestion delay", "it imposes time costs on other drivers", "the dashboard", False),
            ("biology and ecology", "fertilizer runoff", "the algal bloom", "it harms downstream ecosystems", "the field sign", False),
            ("law and regulation", "nuisance case", "the smoke intrusion", "it burdens adjacent property users", "the fence", False),
            ("social situations", "shared workspace", "the distracting noise", "it imposes attention costs on coworkers", "the desk lamp", True),
         ]),
    dict(name="a motif repeats with variation to create coherence",
         role="is the recurring motif that ties separate moments together while changing with context",
         diff=3, insts=[
            ("narrative and discourse", "a novel", "the repeated image of the river", "it returns in different scenes to connect change and memory", "the chapter number", False),
            ("social situations", "a family gathering", "the recurring toast", "it returns each year while meaning shifts with events", "the tablecloth", False),
            ("everyday planning", "a workshop series", "the opening reflection prompt", "it recurs each session to link learning over time", "the room sign", False),
            ("ethics", "an organizational culture", "the repeated fairness question", "it recurs in decisions to keep values coherent", "the meeting agenda", False),
            ("science", "a research program", "the recurring control experiment", "it anchors different studies to a shared comparison", "the lab calendar", False),
            ("law and regulation", "a line of cases", "the repeated reasonableness test", "it recurs across facts to connect judgments", "the filing stamp", True),
         ]),
    dict(name="negative space makes the absent thing perceptible",
         role="is the deliberate absence that shapes attention toward what matters",
         diff=3, insts=[
            ("narrative and discourse", "a short story", "the unsaid apology", "its absence makes the broken relationship visible", "the font", False),
            ("social situations", "a conversation", "the respectful pause", "it leaves space for the other person to speak", "the coffee cup", False),
            ("everyday planning", "a quiet room", "the uncluttered wall", "it lets attention rest on the speaker", "the door hinge", False),
            ("ethics", "informed consent", "the option not to answer", "it makes voluntary choice visible", "the clipboard", False),
            ("program behavior", "interface design", "the empty margin", "it makes the main action easier to find", "the server", False),
            ("medicine-style diagnosis", "a patient interview", "the clinician's silence", "it invites details the checklist might miss", "the exam light", True),
         ]),
    dict(name="a frame changes how the same facts are interpreted",
         role="is the framing context that changes interpretation without changing the underlying facts",
         diff=3, insts=[
            ("narrative and discourse", "a first-person confession", "the narrator's remorse", "it frames events as regret rather than boasting", "the page number", False),
            ("law and regulation", "a legal brief", "the public-safety frame", "it casts the same act as prevention rather than punishment", "the binder", False),
            ("medicine-style diagnosis", "a diagnosis conversation", "the recovery frame", "it casts treatment as regaining function rather than mere restriction", "the stethoscope", False),
            ("ethics", "a resource decision", "the fairness frame", "it casts allocation as equal respect rather than favoritism", "the spreadsheet", False),
            ("social situations", "feedback", "the growth frame", "it casts criticism as support rather than rejection", "the chair", False),
            ("economics and markets", "a price change", "the scarcity frame", "it casts higher price as rationing rather than greed", "the price tag", True),
         ]),
    dict(name="a foil reveals a character or choice by contrast",
         role="is the contrasting counterpart that reveals the target's defining quality",
         diff=3, insts=[
            ("narrative and discourse", "a novel", "the cautious friend", "their caution reveals the hero's impulsiveness", "the street name", False),
            ("social situations", "a team", "the meticulous planner", "their order reveals another member's spontaneity", "the snack bowl", False),
            ("ethics", "a moral dilemma", "the easy self-serving option", "it reveals the costliness of the principled choice", "the office lamp", False),
            ("law and regulation", "case comparison", "the distinguishable precedent", "it reveals which fact actually matters", "the courthouse map", False),
            ("science", "an experiment", "the control condition", "it reveals the treatment's effect by contrast", "the lab coat", False),
            ("finance and business operations", "product strategy", "the no-frills competitor", "it reveals the premium product's service value", "the invoice", True),
         ]),
    dict(name="constraint becomes a creative catalyst rather than only a limit",
         role="is the constraint that focuses invention by ruling out easy options",
         diff=3, insts=[
            ("narrative and discourse", "a sonnet", "the fixed form", "it focuses expression by limiting line and rhyme choices", "the ink color", False),
            ("everyday planning", "a small apartment", "limited space", "it focuses design toward multipurpose furniture", "the mailbox", False),
            ("finance and business operations", "a startup", "a tight budget", "it focuses the team on the highest-value feature", "the office plant", False),
            ("ethics", "privacy-preserving research", "the consent boundary", "it focuses inquiry on respectful methods", "the lab badge", False),
            ("engineering and physical systems", "lightweight design", "the weight limit", "it focuses invention toward efficient structure", "the paint", False),
            ("social situations", "a short toast", "the time limit", "it focuses the speaker on the essential gratitude", "the microphone stand", True),
         ]),
    dict(name="attunement adjusts response to another's state",
         role="is the attuned response that changes because the other party's state is noticed",
         diff=3, insts=[
            ("social situations", "comforting a friend", "the softened tone", "it responds to the friend's distress", "the couch cushion", False),
            ("medicine-style diagnosis", "bedside care", "the pain-adjusted exam", "it changes because the patient's discomfort is noticed", "the wall clock", False),
            ("ethics", "respectful consent", "the check-in question", "it responds to signs of hesitation", "the form header", False),
            ("everyday planning", "teaching a child", "the slower explanation", "it changes because confusion is noticed", "the pencil case", False),
            ("negotiation and interpersonal strategy", "mediation", "the reframed proposal", "it responds to the other side's underlying concern", "the conference table", False),
            ("program behavior", "adaptive interface", "the simplified view", "it responds to signs of user overload", "the server fan", True),
         ]),
    dict(name="repair names harm and rebuilds trust through action",
         role="is the repair action that acknowledges harm and makes renewed trust possible",
         diff=3, insts=[
            ("social situations", "a friendship after a hurtful remark", "the specific apology", "it names the harm and opens repair", "the cafe chair", False),
            ("ethics", "institutional accountability", "the public correction", "it acknowledges error and commits to change", "the podium", False),
            ("law and regulation", "restorative justice", "the restitution agreement", "it recognizes harm and specifies repair", "the docket number", False),
            ("finance and business operations", "customer service", "the refund plus explanation", "it acknowledges the failure and restores confidence", "the receipt ink", False),
            ("medicine-style diagnosis", "medical disclosure", "the error disclosure conversation", "it names the mistake and explains next steps", "the clinic hallway", False),
            ("narrative and discourse", "a reconciliation scene", "the offered amends", "it converts regret into visible action", "the chapter title", True),
         ]),
    dict(name="perspective shift reveals previously hidden meaning",
         role="is the shifted viewpoint that makes a hidden relation visible",
         diff=3, insts=[
            ("narrative and discourse", "a retold scene", "the second narrator", "their viewpoint reveals why the first account was incomplete", "the page edge", False),
            ("social situations", "a disagreement", "the other person's account", "it reveals the need behind the complaint", "the coffee spoon", False),
            ("ethics", "policy review", "the affected person's testimony", "it reveals burdens the designer did not notice", "the hearing room sign", False),
            ("medicine-style diagnosis", "patient history", "the caregiver's observation", "it reveals symptoms missed in the brief exam", "the clinic poster", False),
            ("law and regulation", "case investigation", "the witness angle", "it reveals a fact hidden from the main camera", "the exhibit sticker", False),
            ("science", "microscopy", "the higher magnification", "it reveals structure invisible at lower scale", "the lab stool", True),
         ]),
    dict(name="an underdog uses asymmetry to offset a stronger opponent",
         role="is the asymmetric advantage that lets the weaker side compete against a stronger opponent",
         diff=4, insts=[
            ("narrative and discourse", "David facing Goliath", "the sling", "it converts distance and precision into leverage against brute strength", "Goliath's armor", False),
            ("finance and business operations", "a startup against an incumbent", "rapid iteration", "it lets the smaller firm learn faster than the large firm can react", "the incumbent's headquarters", False),
            ("negotiation and interpersonal strategy", "a small supplier bargaining with a large buyer", "a scarce specialty capability", "it gives the supplier leverage despite size imbalance", "the meeting room", False),
            ("law and regulation", "public-interest litigation", "a narrow test case", "it lets a small group challenge a broad policy through focused facts", "the courthouse steps", False),
            ("incident and root-cause analysis", "a lean response team", "a precise rollback", "it reverses damage faster than adding many responders", "the status-page banner", False),
            ("social situations", "a quiet student in a debate", "the well-chosen question", "it shifts the room despite louder voices", "the lectern", True),
         ]),
    dict(name="hubris causes a powerful actor to ignore a small warning",
         role="is the ignored warning that exposes overconfidence before failure",
         diff=4, insts=[
            ("narrative and discourse", "a king before a downfall", "the ignored prophecy", "it signals danger that pride refuses to hear", "the crown", False),
            ("finance and business operations", "a dominant product team", "the niche user's complaint", "it signals a shift the team dismisses as unimportant", "the launch party", False),
            ("incident and root-cause analysis", "a service before outage", "the low-volume error spike", "it signals fragility before the major failure", "the dashboard theme", False),
            ("law and regulation", "an agency rulemaking", "the minority report", "it flags a flaw the majority overlooks", "the letterhead", False),
            ("medicine-style diagnosis", "a reassuring initial diagnosis", "the inconsistent symptom", "it warns that the simple explanation may be wrong", "the exam table", False),
            ("ethics", "a confident institution", "the dissenting voice", "it warns that power is blinding the group to harm", "the podium", True),
         ]),
    dict(name="a catalyst character triggers change without being the main beneficiary",
         role="is the catalyst actor that provokes transformation without owning the outcome",
         diff=3, insts=[
            ("narrative and discourse", "a coming-of-age story", "the visiting stranger", "they unsettle the town and force the protagonist to choose", "the town sign", False),
            ("finance and business operations", "a turnaround case", "the external consultant", "they reveal the stuck pattern but do not own the company", "the slide template", False),
            ("social situations", "a family conversation", "the honest guest", "they name the tension and make avoidance harder", "the dinner plate", False),
            ("law and regulation", "policy reform", "the investigative report", "it triggers hearings without itself making law", "the archive box", False),
            ("medicine-style diagnosis", "behavior change", "the candid nurse", "they prompt the patient to ask for help", "the clinic badge", False),
            ("negotiation and interpersonal strategy", "stalled talks", "the mediator's reframing question", "it changes the discussion without taking a side", "the water pitcher", True),
         ]),
    dict(name="a scapegoat absorbs blame that belongs to a deeper system",
         role="is the scapegoat that receives blame while the structural cause remains",
         diff=4, insts=[
            ("narrative and discourse", "a village fable", "the accused outsider", "they receive blame for a fear caused by the village's own choices", "the village well", False),
            ("finance and business operations", "a missed quarterly target", "the front-line manager", "they are blamed while the incentive system remains unchanged", "the conference badge", False),
            ("incident and root-cause analysis", "a failed deploy", "the on-call engineer", "they are blamed while the release process remains brittle", "the alert sound", False),
            ("law and regulation", "public scandal", "the low-level official", "they absorb blame while the policy design remains intact", "the press podium", False),
            ("social situations", "a group project", "the quiet member", "they are blamed while coordination failures go unnamed", "the shared document", False),
            ("ethics", "institutional harm", "the individual rule-breaker", "they are blamed while the enabling culture persists", "the office seal", True),
         ]),
    dict(name="a bridge figure translates between groups that do not trust each other",
         role="is the trusted bridge figure that carries meaning across a divide",
         diff=3, insts=[
            ("narrative and discourse", "two rival clans", "the bilingual messenger", "they carry intent between groups that suspect each other", "the clan banner", False),
            ("negotiation and interpersonal strategy", "labor talks", "the respected shop steward", "they translate management proposals into worker concerns and back", "the bargaining table", False),
            ("finance and business operations", "technical sales", "the solutions engineer", "they translate product capability into customer value", "the demo laptop", False),
            ("medicine-style diagnosis", "care coordination", "the patient navigator", "they translate clinical instructions into practical next steps", "the waiting-room poster", False),
            ("law and regulation", "community consent", "the local liaison", "they translate legal process into community concerns", "the hearing sign", False),
            ("social situations", "a blended family", "the trusted relative", "they carry reassurance between people who are wary", "the family photo", True),
         ]),
    dict(name="a poisoned gift creates dependency while appearing beneficial",
         role="is the attractive offer that carries a hidden dependency or cost",
         diff=4, insts=[
            ("narrative and discourse", "a fairy bargain", "the enchanted gift", "it solves an immediate problem while binding the hero to a price", "the castle gate", False),
            ("finance and business operations", "vendor strategy", "the steep initial discount", "it lowers entry cost while creating lock-in", "the invoice footer", False),
            ("economics and markets", "resource diplomacy", "the subsidized loan", "it funds a project while increasing dependence on the lender", "the ribbon cutting", False),
            ("program behavior", "software integration", "the proprietary SDK", "it speeds launch while tying the product to one platform", "the code comment", False),
            ("social situations", "a favor with strings", "the unsolicited expensive favor", "it appears generous while creating obligation", "the gift wrap", False),
            ("ethics", "sponsored research", "the restricted grant", "it supports work while constraining what may be reported", "the grant logo", True),
         ]),
    dict(name="a kingmaker changes the outcome without becoming the ruler",
         role="is the pivotal third party whose support determines which side prevails",
         diff=4, insts=[
            ("narrative and discourse", "a succession tale", "the neutral general", "their allegiance decides the throne without making them monarch", "the banquet hall", False),
            ("law and regulation", "a coalition legislature", "the small pivotal party", "its votes decide which bill can pass", "the chamber clock", False),
            ("finance and business operations", "a platform ecosystem", "the major complementor", "its support decides which standard gains adoption", "the trade-show booth", False),
            ("negotiation and interpersonal strategy", "multi-party talks", "the holdout stakeholder", "their consent determines whether a deal closes", "the agenda packet", False),
            ("social situations", "a committee decision", "the undecided member", "their preference decides between two camps", "the whiteboard", False),
            ("economics and markets", "standard-setting", "the anchor customer", "their purchase decision tips the market toward one supplier", "the purchase order", True),
         ]),
    dict(name="a decoy redirects attention away from the decisive move",
         role="is the decoy that attracts attention while the real action happens elsewhere",
         diff=4, insts=[
            ("narrative and discourse", "a heist story", "the noisy street argument", "it draws guards away while the vault is entered", "the getaway car", False),
            ("finance and business operations", "competitive launch", "the teaser feature", "it draws rivals' attention while the core platform is built", "the press badge", False),
            ("incident and root-cause analysis", "an attack response", "the obvious phishing email", "it draws analysts while the credential abuse continues", "the ticket title", False),
            ("law and regulation", "investigation strategy", "the minor procedural dispute", "it absorbs attention while the central evidence is developed", "the binder tab", False),
            ("negotiation and interpersonal strategy", "bargaining", "the expendable demand", "it attracts concessions while the real priority is protected", "the coffee carafe", False),
            ("social situations", "a surprise party", "the fake errand", "it redirects attention while guests gather", "the wrapping paper", True),
         ]),
    dict(name="a formal transformation applies the same operator despite different symbols",
         role="is the operator that transforms the source representation into the target representation",
         diff=4, insts=[
            ("formal grammars and symbol systems", "a letter-string analogy", "advance the final symbol", "it maps abc to abd while preserving the prefix", "the font style", False),
            ("mathematics", "a function table", "add one to the last coordinate", "it changes only the final coordinate by the same rule", "the table border", False),
            ("program behavior", "a state update", "increment the tail field", "it changes the final field while leaving earlier fields unchanged", "the log prefix", False),
            ("logic puzzles", "a symbol matrix", "advance the rightmost token", "it completes the row by applying the same local operation", "the grid color", False),
            ("formal grammars and symbol systems", "a cipher exercise", "shift the last glyph", "it changes the final glyph while preserving the rest", "the page margin", False),
            ("algorithms and program analysis", "a tuple rewrite", "successor on the last component", "it applies the operation to one selected component", "the variable name", True),
         ]),
    dict(name="a matrix cell is determined by row and column relations, not by appearance",
         role="is the row-column rule that determines the missing cell",
         diff=4, insts=[
            ("logic puzzles", "a Raven-style grid", "combine row features then subtract the overlap", "it fixes the missing symbol by a relation across the row", "a darker outline", False),
            ("mathematics", "a number table", "row sum conservation", "it fixes the missing entry from the other cells", "the cell shading", False),
            ("formal grammars and symbol systems", "a glyph matrix", "column-wise rotation", "it fixes the missing glyph by the column operation", "the glyph size", False),
            ("program behavior", "a truth-table grid", "exclusive-or composition", "it determines the output cell from two inputs", "the column label", False),
            ("science", "an experimental design table", "factor interaction", "it predicts the combined-condition result", "the lab notebook line", False),
            ("finance and business operations", "a scenario table", "base case plus adjustment", "it determines the missing forecast cell", "the spreadsheet color", True),
         ]),
    dict(name="robust connectivity arises from alternate paths across unlike substrates",
         role="is the alternate-path design that preserves connectivity when one route fails",
         diff=4, insts=[
            ("engineering and physical systems", "a data-center backbone", "the redundant spine links", "they keep packets moving if one link fails", "the rack label", False),
            ("biology and ecology", "ion movement through a membrane network", "multiple percolation channels", "they preserve passage if one pore is blocked", "the lipid label", False),
            ("science", "a porous rock", "connected pore pathways", "they let fluid percolate around a blocked pore", "the sample tray", False),
            ("program behavior", "a service mesh", "alternate service routes", "they keep requests flowing around a failed instance", "the log color", False),
            ("economics and markets", "a trading network", "substitute trading partners", "they keep exchange possible if one partner exits", "the price board", False),
            ("social situations", "mutual-aid communication", "multiple trusted contacts", "they keep information moving if one contact is unavailable", "the group logo", True),
         ]),
    dict(name="the same gradient law drives flow in distant STEM and social systems",
         role="is the gradient that drives movement from high potential to low potential",
         diff=4, insts=[
            ("science", "heat transfer", "the temperature difference", "it drives heat from warmer to cooler regions", "the thermometer casing", False),
            ("chemistry", "diffusion", "the concentration gradient", "it drives molecules from high concentration to low concentration", "the beaker", False),
            ("engineering and physical systems", "an electrical circuit", "the voltage difference", "it drives current through the circuit", "the wire insulation", False),
            ("economics and markets", "cash allocation", "the return difference", "it pulls capital toward higher return until the gap narrows", "the account logo", False),
            ("biology and ecology", "osmosis", "the water-potential gradient", "it drives water across the membrane", "the cell wall", False),
            ("everyday planning", "traffic route choice", "the travel-time difference", "it diverts drivers toward the faster route", "the road sign", True),
         ]),
    dict(name="recursion embeds the same relation inside itself at a smaller scale",
         role="is the self-similar rule that repeats inside its own output",
         diff=4, insts=[
            ("formal grammars and symbol systems", "a nested sentence grammar", "the recursive noun phrase rule", "it lets a phrase contain another phrase of the same kind", "the punctuation mark", False),
            ("program behavior", "a recursive function", "the self-call", "it solves a smaller version of the same problem", "the stack trace color", False),
            ("biology and ecology", "branching growth", "the repeated branching rule", "it creates smaller branches with the same pattern", "the bark texture", False),
            ("mathematics", "a fractal construction", "the self-similar replacement rule", "it inserts a scaled copy of the whole pattern", "the axis label", False),
            ("narrative and discourse", "a story within a story", "the embedded tale", "it repeats the outer conflict at a smaller scale", "the chapter title", False),
            ("algorithms and program analysis", "divide-and-conquer", "the recursive subproblem", "it repeats the original task on a smaller input", "the variable name", True),
         ]),
    dict(name="emergent order appears when many local interactions create a global pattern",
         role="is the local interaction rule whose repetition produces the global pattern",
         diff=4, insts=[
            ("biology and ecology", "an ant trail", "pheromone following", "many simple choices create a stable trail", "a leaf", False),
            ("economics and markets", "a market price", "individual bids and asks", "many local trades create a public price", "the trading-floor clock", False),
            ("program behavior", "a cellular automaton", "the neighbor update rule", "many local updates create global structure", "the screen border", False),
            ("social situations", "a crowd norm", "small acts of imitation", "many local choices create a shared expectation", "the venue sign", False),
            ("science", "crystal formation", "local molecular attachment", "many local attachments create a lattice", "the vial cap", False),
            ("narrative and discourse", "a rumor plot", "person-to-person retelling", "many local retellings create the public story", "the poster", True),
         ]),
    dict(name="a game tree separates hidden information from optimal action",
         role="is the information constraint that limits which strategy can be justified",
         diff=4, insts=[
            ("logic puzzles", "a card-placement game", "the hidden card", "it prevents certainty and requires contingent planning", "the table felt", False),
            ("negotiation and interpersonal strategy", "a bargaining game", "the private reservation value", "it limits what each side can infer from offers", "the coffee cup", False),
            ("finance and business operations", "competitive bidding", "unknown rival costs", "they force bids to account for hidden information", "the bid form", False),
            ("program behavior", "adversarial search", "the opponent's private state", "it requires planning over possible branches", "the debug label", False),
            ("law and regulation", "settlement strategy", "undisclosed evidence strength", "it shapes offers under uncertainty", "the docket tab", False),
            ("social situations", "a trust-building choice", "the other person's unspoken preference", "it requires a cautious action rather than a guaranteed one", "the room light", True),
         ]),
    dict(name="a constraint graph propagates obligations through dependencies",
         role="is the dependency edge that transmits a constraint from one node to another",
         diff=4, insts=[
            ("mathematics", "a graph coloring problem", "the adjacency edge", "it transmits the not-same-color constraint", "the node label", False),
            ("program behavior", "a build system", "the dependency link", "it forces prerequisites to complete first", "the terminal prompt", False),
            ("finance and business operations", "a project plan", "the predecessor task", "it constrains when the dependent task may start", "the slide footer", False),
            ("law and regulation", "a compliance chain", "the delegated duty", "it passes an obligation to the next actor", "the form number", False),
            ("biology and ecology", "a food web", "the feeding link", "it transmits population pressure between species", "the field notebook", False),
            ("engineering and physical systems", "a load path", "the structural joint", "it transmits force to the next member", "the paint mark", True),
         ]),
]

def _iso_srcs(train_insts, target, picks):
    """Two sources from DIFFERENT domains than the target (and each other)."""
    td = target[0]
    cands = [s for s in picks if s[0] != td and s is not target and s in train_insts]
    chosen, doms = [], set()
    for s in cands:
        if s[0] not in doms:
            chosen.append(s); doms.add(s[0])
        if len(chosen) == 2:
            return chosen
    return None

def _mk_iso(st, target, srcs, split, use_distractor, fmt="rolemap"):
    name, role, diff = st["name"], st["role"], st["diff"]
    opposing = st.get("opposing", False)
    (td, tsys, tfill, tctx, tdis, _) = target
    src_txt = "; ".join(f"in {s[1]}, {s[2]} {role} ({s[3]})" for s in srcs)
    ans_head = tfill.split("(")[0].replace("the ", "").strip().lower()
    if ans_head and ans_head in src_txt.lower():   # leakage guard
        return None
    if opposing:
        first_force = tctx
        prob = (f"The same structure -- {name} -- appears in many systems: "
                + "; ".join(f"in {s[1]}, {s[3]}, and {s[2]} balances it" for s in srcs)
                + f". Now consider {tsys} ({td}): {first_force}. "
                f"By the same structure, what opposing influence balances it?")
        steps = [
            (f"Shared structure: {name}. The role to map is the opposing influence that restores balance.", "valid"),
            ("; ".join(f"in {s[1]} ({s[0]}) it is {s[2]}" for s in srcs) + " -- different domains, same balancing role.", "valid"),
            (f"Map onto {tsys}: given that {first_force}, the influence that balances it is {tfill}.", "valid"),
            (f"Check: sources ({', '.join(s[0] for s in srcs)}) and target ({td}) share no vocabulary, so the balancing role -- not surface similarity -- yields {tfill}.", "valid"),
        ]
        final = f"{cap(tfill)} -- it is the opposing influence that balances {first_force.split(' pushes')[0].split(' builds')[0]} in {tsys}."
    elif fmt == "proportional":
        prob = (f"{cap(srcs[0][2])} is to {srcs[0][1]} as {srcs[1][2]} is to {srcs[1][1]} -- in each, it {role}. "
                f"By the same relation, what is to {tsys} ({td})?")
        if use_distractor and tdis:
            prob += f" (Note: {tdis} is present too, but decide by role, not by surface prominence.)"
        steps = [
            (f"Read the relation across the two source pairs: in {srcs[0][1]} it is {srcs[0][2]}, in {srcs[1][1]} it is {srcs[1][2]} -- in each, the element that {role}.", "valid"),
            (f"That is one shared structure ({name}) in two unrelated domains ({srcs[0][0]}, {srcs[1][0]}), so it is the relation -- not any surface feature -- that must transfer.", "valid"),
            (f"Carry it to {tsys}: the element that {role} there is {tfill}.", "valid"),
        ]
        if use_distractor and tdis:
            steps.append((f"Reject the surface distractor: {tdis} is present in {tsys} but does not fill this role.", "valid"))
        steps.append((f"Check: target domain ({td}) differs from both sources and shares no vocabulary with the answer, so only the structural role yields {tfill}.", "valid"))
        final = f"{cap(tfill)} -- to {tsys} as {srcs[0][2]} is to {srcs[0][1]}."
    else:
        prob = (f"The same relational structure -- {name} -- shows up across unrelated systems: {src_txt}. "
                f"Now consider {tsys} ({td}). By the same structure, which element {role}?")
        if use_distractor and tdis:
            prob += f" (Note: {tdis} is also present, but decide by role, not by surface prominence.)"
        steps = [
            (f"Shared relational structure: {name}. The role to map across systems is: it {role}.", "valid"),
            ("In " + srcs[0][1] + f", that role is filled by {srcs[0][2]}; in {srcs[1][1]}, by {srcs[1][2]} -- different domains, one role.", "valid"),
            (f"Map the structure onto {tsys}: the element that {role} there is {tfill}.", "valid"),
        ]
        if use_distractor and tdis:
            steps.append((f"Reject the surface distractor: {tdis} is present in {tsys} but does not fill this role, so its presence does not make it the answer.", "valid"))
        steps.append((f"Check: the sources ({srcs[0][0]}, {srcs[1][0]}) and the target ({td}) share no vocabulary, so only the structural role -- not surface similarity -- yields {tfill}.", "valid"))
        final = f"{cap(tfill)} -- it plays the same role in {tsys} that {srcs[0][2]} plays in {srcs[0][1]}."
    d = diff + (1 if (use_distractor and tdis) else 0)
    return dict(domain=td, problem=prob, steps=steps, final=final, difficulty=d,
                vm="process_check", split=split,
                vd=f"Cross-domain role map ({name}): answer fills the target role in a domain ({td}) different from the sources; leakage-guarded so the answer is not present in the source text.")

# --------------------------------------------------------------------------
# The seed-driven sampling engine.
#
# Eval is a STABLE benchmark: a fixed set of held-out items in the plain
# register, enumerated deterministically, so it is identical every season and
# saturates (dedup adds nothing on later seasons). Train is sampled from the
# huge combinatorial space (parametric numeric traps are unbounded; iso has
# structure x target x source-pair x format; simple has relation x target x
# phrasing) with a random register per item, so each --seed yields fresh,
# non-overlapping train items. Volume is set by --per-type (`need`).
# --------------------------------------------------------------------------
def _analogical_eval(seen):
    """Deterministic held-out benchmark (plain register)."""
    out = []
    def add(it):
        if it and it["problem"] not in seen:
            seen.add(it["problem"]); out.append(it)
    # simple: eval answer pools (and whole eval-only relations)
    for rt in ANA_REL:
        relphrase, left, right, dom, diff, eval_only, pairs = rt
        _, eval_pairs = _simple_pools(rt)
        for ti, (t0, ans) in enumerate(eval_pairs):
            others = [(a, b) for (a, b) in eval_pairs if a != t0][:3]
            if len(others) < 3:
                continue
            add(_mk_simple(relphrase, left, right, dom, diff, t0, ans, others, Q_ANA_EVAL[0], "eval"))
    # traps: eval skins / a fixed param set
    for sk in T1_SKINS[6:]:
        for bi in (0, 1):
            add(_mk_t1(sk, bi, "eval"))
    for sk in T2_SKINS[4:]:
        for nums in [((60, 24, 90), (4, 2, 9)), ((100, 30, 80), (5, 2, 8))]:
            add(_mk_t2(sk, nums[0], nums[1], "eval"))
    for sk in T3_SKINS[4:]:
        add(_mk_t3(sk, "eval"))
    for spec in [(_sq, "each number maps to its square", _tpl, "tripling", [3, 5, 7], 8, "mathematics"),
                 (_cube, "each number maps to its cube", (lambda n: 7 * n - 6), "the rule 7n-6", [1, 4, 6], 5, "science")]:
        add(_mk_t4(spec, "eval"))
    # iso: held-out target instances, canonical role-map, sources from train insts
    for st in STRUCTURES:
        train_insts = [i for i in st["insts"] if not i[5]]
        evalt = [i for i in st["insts"] if i[5]]
        for t in evalt:
            srcs = _iso_srcs(train_insts, t, train_insts)
            if srcs:
                add(_mk_iso(st, t, srcs, "eval", False, "rolemap"))
    return out

def _shash(s):
    """Small stable string hash (Python's hash() is salted per process). Used to
    pick a phrasing/register DETERMINISTICALLY per answer-identity so a given
    analogy always renders the same way -> it dedups across seasons and its
    answer is never re-emitted under a different surface."""
    h = 2166136261
    for c in s:
        h = ((h ^ ord(c)) * 16777619) & 0xFFFFFFFF
    return h

# clean parametric numeric analogy (no distractor) -- answer-diverse volume at
# lower difficulty, to balance the difficulty-5 traps.
_NUM_OPS = [(_sq, "each maps to its square", 2), (_cube, "each maps to its cube", 3),
            (_dbl, "each maps to double itself", 2), (_tpl, "each maps to triple itself", 2),
            (_tri, "each maps to the running total 1..n", 3),
            ((lambda n: n * n + 1), "each maps to (its square + 1)", 3),
            ((lambda n: n * (n - 1)), "each maps to n*(n-1)", 3)]
def _mk_num(rng, split):
    op, label, diff = rng.choice(_NUM_OPS)
    xs = sorted(rng.sample(range(2, 14), 3)); tx = rng.choice([x for x in range(2, 16) if x not in xs])
    dom = rng.choice(T4_DOMAINS)
    demo = "; ".join(f"{x} -> {op(x)}" for x in xs)
    steps = [
        (f"Read the relation shared by every pair: {label}. Confirm on the demos: {demo}.", "valid"),
        (f"The pairs differ in their numbers but share this one relation, so it -- not any single pair -- is what transfers.", "valid"),
        (f"Apply it to {tx}: the answer is {op(tx)}.", "valid"),
        (f"Check: {op(tx)} follows the relation that holds across all demonstrations.", "valid"),
    ]
    prob = f"Every pair follows one relation ({label}): {demo}. By the same relation, what completes {tx} -> ?"
    return dict(domain=dom, problem=prob, steps=steps, final=f"{op(tx)}.", difficulty=diff,
                vm="process_check", split=split,
                vd=f"Parametric numeric analogy verified: op({tx})={op(tx)} under '{label}'.")

_SCALE_LAWS = [("square", 2, "doubling the input multiplies the output by four"),
               ("cube", 3, "doubling the input multiplies the output by eight"),
               ("inverse-square", 2, "halving the distance multiplies the intensity by four")]
_SCALE_DOMAINS = ["engineering and physical systems", "science", "economics and markets",
                  "biology and ecology", "chemistry"]
def _mk_scale(rng, split):
    """Parametric difficulty-4: transfer a power law across domains; the linear
    reading is the distractor."""
    lawname, p, gloss = rng.choice(_SCALE_LAWS)
    k = rng.randint(2, 6)
    dom = rng.choice(_SCALE_DOMAINS)
    ans = k ** p
    prob = (f"Reference system: the output follows {_art(lawname)} law of the input -- {gloss}, "
            f"so the output scales with the input raised to the power {p}. "
            f"An analogous system in {dom} obeys the same {lawname} law; its input is scaled by a factor of {k}. "
            f"By the same law, the output is scaled by what factor? (A linear reading -- factor {k} -- is the distractor.)")
    steps = [
        (f"Relation to transfer: output scales as (input)^{p} (the {lawname} law), not linearly.", "valid"),
        (f"Reject the surface distractor: a linear reading would give factor {k}, but the law is a power of {p}.", "valid"),
        (f"Apply the law: the input factor {k} raised to the power {p} is {k}^{p} = {ans}.", "valid"),
        (f"Check: under {_art(lawname)} law an input factor of {k} yields an output factor of {k}**{p} = {ans}; the linear value {k} is wrong.", "valid"),
    ]
    return dict(domain=dom, problem=prob, steps=steps, final=f"A factor of {ans}.", difficulty=4,
                vm="process_check", split=split,
                vd=f"Power-law transfer verified: {k}^{p} = {ans}; linear distractor {k} rejected.")

def build_analogical(need, rng, exclude):
    """Two-layer engine.

    Breadth layer (bounded, deterministic): every distinct answer-identity from
    the banks -- relations, cross-domain structures, trap skins -- rendered ONCE
    with a phrasing/register fixed by a stable hash of its identity. Stable
    across seasons, so it dedups and no answer is re-emitted under a new surface.

    Volume layer (unbounded, parametric): numeric analogies, binding-constraint
    and competing-relation traps with rng-drawn, self-verified numbers -- every
    item has a genuinely different answer, so volume scales to tens of thousands
    across seasons without repeating answers. `need` (=--per-type) sets train
    volume; `rng` (=--seed) drives seasonal freshness; `exclude` dedups vs disk.
    """
    out, seen = [], set(exclude)
    # stable eval benchmark (identical every season; saturates via dedup)
    for it in _analogical_eval(seen):
        out.append(it)
    made = 0

    def emit(it):
        nonlocal made
        if it and it["problem"] not in seen:
            seen.add(it["problem"]); out.append(it); made += 1
            return True
        return False

    # ---- breadth layer (deterministic per identity) ----
    bounded = []
    for rt in ANA_REL:
        if rt[5]:
            continue
        relphrase, left, right, dom, diff, _, _ = rt
        tp = _simple_pools(rt)[0]
        for t0, ans in tp:
            others = [(a, b) for (a, b) in tp if a != t0][:3]
            if len(others) < 3:
                continue
            key = "S|" + relphrase + "|" + str(t0)
            tmpl = Q_ANA_TRAIN[_shash(key) % len(Q_ANA_TRAIN)]
            it = _mk_simple(relphrase, left, right, dom, diff, t0, ans, others, tmpl, "train")
            it["problem"] = REGISTERS[_shash("R" + key) % len(REGISTERS)](it["problem"])
            bounded.append(it)
    for st in STRUCTURES:
        ti = [i for i in st["insts"] if not i[5]]
        if len(ti) < 3:
            continue
        for vi, t in enumerate(ti):
            for variant in (0, 1):     # <=2 framings per (structure,target)
                order = ti[variant:] + ti[:variant]
                srcs = _iso_srcs(ti, t, order)
                if not srcs:
                    continue
                fmt = "rolemap" if (st.get("opposing") or variant == 0) else "proportional"
                ud = bool(t[4]) and variant == 1
                it = _mk_iso(st, t, srcs, "train", ud, fmt)
                if not it:
                    continue
                key = "I|" + st["name"] + "|" + t[1] + "|" + str(variant)
                it["problem"] = REGISTERS[_shash(key) % len(REGISTERS)](it["problem"])
                bounded.append(it)
    for sk in T1_SKINS[:6]:
        for bi in (0, 1):
            it = _mk_t1(sk, bi, "train")
            it["problem"] = REGISTERS[_shash("T1|" + sk[0] + str(bi)) % len(REGISTERS)](it["problem"])
            bounded.append(it)
    for sk in T3_SKINS[:4]:
        it = _mk_t3(sk, "train")
        it["problem"] = REGISTERS[_shash("T3|" + sk[1] + sk[3]) % len(REGISTERS)](it["problem"])
        bounded.append(it)
    rng.shuffle(bounded)   # order variety only; content is identity-stable
    for it in bounded:
        if made >= need:
            break
        emit(it)

    # ---- volume layer (parametric, unbounded) ----
    t2_train = T2_SKINS[:4]
    # weighted mix spreads difficulty across the volume tail:
    # num = d2-3, scale = d4, t2/t4 = d5.
    vol = (["num"] * 4 + ["scale"] * 3 + ["t4"] * 2 + ["t2"] * 2)
    tries, cap = 0, need * 80 + 5000
    while made < need and tries < cap:
        tries += 1
        pick = rng.choice(vol)
        if pick == "num":
            it = _mk_num(rng, "train")
        elif pick == "scale":
            it = _mk_scale(rng, "train")
        elif pick == "t2":
            nums = _rand_t2_nums(rng)
            it = _mk_t2(rng.choice(t2_train), nums[0], nums[1], "train") if nums else None
        else:
            spec = _rand_t4(rng)
            it = _mk_t4(spec, "train") if spec else None
        if it:
            it["problem"] = _register(rng, it["problem"])
            emit(it)
    return out

# Moral-ethical: each MODE asks a genuinely DIFFERENT question about the dilemma
# and produces a genuinely different analysis (not one trace re-skinned across
# stems), so each (scenario, mode) is a distinct reasoning task. Verification is
# rubric_judge and NO mode forces a verdict (the type rewards reasoning quality,
# not a fixed conclusion).
def _mor_both(fa, fb):
    return ([(f"Frame the tension: {fa} versus {fb}.", "valid"),
             (f"Strongest case for the first side: it rests on {fa}, and on that principle the choice is justified.", "valid"),
             (f"Strongest case for the second side: it rests on {fb}, which a pure appeal to the first ignores.", "valid"),
             ("Each case is internally coherent; they differ on which principle is prior.", "valid")],
            f"Both sides are defensible: one case rests on {fa}, the other on {fb}. Which is stronger turns on the priority given to each principle, not on a forced verdict.")
def _mor_conflict(fa, fb):
    return ([(f"Principle at stake on one side: {fa}.", "valid"),
             (f"Principle at stake on the other: {fb}.", "valid"),
             (f"They conflict because honoring {fa} here requires setting aside {fb}, and vice versa.", "valid"),
             ("The conflict is genuine: no option satisfies both principles fully.", "valid")],
            f"The competing principles are {fa} and {fb}; they conflict because each can be honored here only at the other's expense.")
def _mor_frameworks(fa, fb):
    return ([(f"Consequentialist reading: weighing outcomes tends to support {fa}.", "valid"),
             (f"Deontological reading: some duties resist an outcome count, supporting {fb}.", "valid"),
             (f"Virtue/care reading: asks which choice expresses good character and preserves relationships, which can cut either way.", "valid"),
             ("The frameworks disagree, so the verdict depends on which framework is authoritative.", "valid")],
            f"Under consequentialism the case leans to {fa}; under a duty- or rights-based view it leans to {fb}; a virtue reading is not decisive. The frameworks diverge.")
def _mor_reason(fa, fb):
    return ([(f"Name the tension: {fa} versus {fb}.", "valid"),
             ("Weigh the outcomes and the duties on each side without assuming a winner.", "valid"),
             ("Consistency check: the chosen principle must be acceptable applied generally and to oneself.", "valid"),
             ("Acknowledge the residual: any choice leaves a real moral cost, which the reasoning must name.", "valid")],
            f"Reason it by making the trade-off between {fa} and {fb} explicit and adopting a principle one could accept applied generally; whichever is chosen leaves a residual cost to acknowledge.")
def _mor_rebut(fa, fb):
    return ([(f"Argue one side: the choice grounded in {fa} is justified because the stakes it protects are weighty.", "valid"),
             (f"Strongest rebuttal: {fb} shows that this reasoning proves too much and licenses conclusions we would reject elsewhere.", "valid"),
             ("The rebuttal does not simply reassert the other side; it attacks the first argument's principle.", "valid"),
             ("A reply would have to limit the principle so it does not overreach.", "valid")],
            f"The case for the first side rests on {fa}; the strongest rebuttal is that {fb} exposes it as overreaching, so the argument needs a principled limit to survive.")
def _mor_stakeholders(fa, fb):
    return ([("Identify the stakeholders: the decision-maker, those directly affected, and third parties.", "valid"),
             (f"The party favored by {fa} is owed the protection that principle names.", "valid"),
             (f"The party favored by {fb} is owed what that principle names, which the first can override.", "valid"),
             ("Naming what each is owed shows the decision distributes burdens, not just benefits.", "valid")],
            f"Each stakeholder is owed something: those served by {fa} and those served by {fb}. The dilemma is how to distribute the unavoidable burden between them.")
def _mor_diverge(fa, fb):
    return ([(f"A consequentialist counts total welfare and tends to endorse {fa}.", "valid"),
             (f"A rights-based view asks what is owed regardless of the tally, and tends to endorse {fb}.", "valid"),
             ("They diverge exactly where maximizing the aggregate would violate an individual claim.", "valid"),
             ("Neither view is obviously wrong, so the divergence is the crux.", "valid")],
            f"They diverge where maximizing aggregate welfare ({fa}) would override an individual claim ({fb}); that trade-off is the crux.")
def _mor_principle(fa, fb):
    return ([(f"To justify the first choice you must accept a principle like: {fa} may take priority even at the cost named by {fb}.", "valid"),
             (f"To justify the second you must accept: {fb} constrains action even when {fa} would gain by overriding it.", "valid"),
             ("Test each principle by universalizing it: would you accept it applied to yourself and in other cases?", "valid"),
             ("Each choice commits you to a general principle, so the decision is really a choice between principles.", "valid")],
            f"Each option commits you to a general principle -- either that {fa} may override, or that {fb} constrains -- and the choice is defensible only if you accept the principle it requires, universally.")
MORAL_MODES = [
    ("Lay out the strongest case for each side.", 3, _mor_both),
    ("What competing ethical principles are at stake, and how do they conflict?", 3, _mor_conflict),
    ("Analyze this dilemma from at least two ethical frameworks.", 4, _mor_frameworks),
    ("How should this be reasoned about?", 3, _mor_reason),
    ("Argue the case for one side, then give the strongest rebuttal.", 4, _mor_rebut),
    ("Identify the stakeholders and what each is owed.", 2, _mor_stakeholders),
    ("Where would a consequentialist and a rights-based view diverge here?", 4, _mor_diverge),
    ("What principle would you have to accept to justify each choice?", 5, _mor_principle),
]
def _mk_moral(scenario, fa, fb, mode, split):
    stem, diff, fn = mode
    steps, final = fn(fa, fb)
    return dict(domain="ethics", problem=scenario + " " + stem, steps=steps, final=final,
        difficulty=diff, vm="rubric_judge", split=split,
        vd="Rubric scored perspective coverage, reasoning quality, and internal consistency; no predetermined verdict rewarded.")

def _moral_eval(seen):
    """Held-out eval: reserved scenarios (last 16) across all modes."""
    out = []
    def add(it):
        if it and it["problem"] not in seen:
            seen.add(it["problem"]); out.append(it)
    for scenario, fa, fb in MORAL_BANK[-16:]:
        for mode in MORAL_MODES:
            add(_mk_moral(scenario, fa, fb, mode, "eval"))
    return out

def build_moral(need, rng, exclude):
    out, seen = [], set(exclude)
    for it in _moral_eval(seen):
        out.append(it)
    made = 0
    for scenario, fa, fb in MORAL_BANK[:-16]:      # train scenarios (reserved held out)
        for mode in MORAL_MODES:
            if made >= need:
                return out
            it = _mk_moral(scenario, fa, fb, mode, "train")
            if it["problem"] in seen:
                continue
            seen.add(it["problem"]); out.append(it); made += 1
    return out

BUILDERS = {"deductive": build_deductive, "inductive": build_inductive, "probabilistic": build_probabilistic,
            "counterfactual": build_counterfactual, "causal": build_causal, "metacognitive": build_metacognitive,
            "abductive": build_abductive, "analogical": build_analogical, "moral-ethical": build_moral}
NEG_BUILDERS = {"deductive": neg_deductive, "inductive": neg_inductive, "probabilistic": neg_probabilistic,
                "counterfactual": neg_counterfactual, "causal": neg_causal}
# HONESTY (SCHEMA/RULES + the BAR): every type here is produced by THIS SCRIPT,
# so generation_method is "procedural" for all of them -- labeling any of them
# "human_expert"/"multi_agent" would misrepresent how the data is really made.
# The verifiable types are additionally checked by running their verifier in
# Python (only passing items are emitted); the rest carry their verification
# method by construction and await an independent Solver/judge pass, which the
# provenance.source string states plainly (never claims hand-authoring).
GEN_METHOD = {t: "procedural" for t in RT}
SOURCE = {t: ("procedural-gen, verified" if t in VERIFIABLE
              else "procedural-gen from authored banks" if t in {"abductive", "moral-ethical"}
              else "procedural-gen") for t in RT}

# --------------------------------------------------------------------------
# Word / relation / scenario BANKS  (# BANK: extend to scale a type)
# --------------------------------------------------------------------------
NAMES = ["Priya","Marcus","Lena","Diego","Aisha","Tomas","Nadia","Ravi","Mei","Omar","Sofia","Jonas","Yuki","Kwame","Ines","Bardia",
         "Anika","Bjorn","Chidi","Dalia","Ewan","Fatima","Goran","Hana","Idris","Juno","Keiko","Lucas","Mira","Nils","Oksana","Pablo",
         "Rania","Selim","Tariq","Uma","Vikram","Wen","Ximena","Yara","Zane","Amara","Bao","Camille","Darius","Eitan","Freya","Gita"]
WORDS = ["cat","door","lamp","river","stone","cloud","piano","glass","tiger","melon","robot","north","amber","field","otter","zebra","quartz","violet","harbor","cactus","ember","willow","comet","ledger",
         "anchor","basil","cobalt","dune","echo","falcon","garnet","hazel","ivory","jade","kelp","lantern","marble","nectar","onyx","pebble","raven","saffron","thistle","umber","vellum","walnut","yonder","zephyr",
         "beacon","cinder","drift","fjord","gable","hollow","islet","juniper","kernel","lichen","mesa","nimbus","orchard","prairie","reef","spruce","tundra","vortex"]

CAUSAL_DRIVERS = {  # BANK: extend to scale causal (each driver adds C(len,2) confounder pairs)
    "hot weather": (["ice cream sales","swimming-pool visits","air-conditioner use","cold-drink sales","sunscreen sales","beach attendance","fan sales","popsicle sales"], "economics and markets"),
    "cold weather": (["heater use","hot-chocolate sales","soup sales","firewood sales","scarf purchases","ice-rink attendance","space-heater sales","hot-tea sales"], "economics and markets"),
    "heavy rainfall": (["umbrella sales","raincoat purchases","indoor-cinema attendance","taxi demand","boot sales","gutter-cleaning calls"], "science"),
    "a traffic surge": (["page latency","error-log volume","support tickets","checkout failures","server CPU load","cache misses"], "incident and root-cause analysis"),
    "a public holiday": (["retail foot traffic","restaurant bookings","parking occupancy","toll-road volume","cinema tickets","ride-share demand"], "economics and markets"),
    "pollen season": (["antihistamine sales","tissue sales","allergy-clinic visits","eye-drop sales","air-purifier sales"], "medicine-style diagnosis"),
    "a heat wave": (["heatstroke ER visits","bottled-water sales","AC-repair calls","pool-chemical sales","dehydration cases"], "medicine-style diagnosis"),
    "exam season": (["campus coffee sales","library occupancy","energy-drink sales","printing-shop demand","late-night food orders"], "economics and markets"),
    "a drought": (["wildfire-risk alerts","irrigation demand","crop-insurance claims","dust-storm reports","reservoir-refill costs","hay prices"], "science"),
    "an economic downturn": (["loan defaults","pawnshop traffic","discount-store sales","bankruptcy filings","unemployment claims","gold demand"], "economics and markets"),
    "a flu outbreak": (["pharmacy visits","school absences","clinic wait times","tissue sales","sick-leave requests","thermometer sales"], "medicine-style diagnosis"),
    "a major product launch": (["support-ticket volume","server CPU load","social mentions","return requests","checkout errors"], "incident and root-cause analysis"),
    "a cold snap": (["pipe-burst calls","road-salt use","blanket sales","frostbite ER visits","power demand"], "engineering and physical systems"),
    "rising fuel prices": (["freight surcharges","carpool sign-ups","staycation bookings","e-bike sales","transit ridership"], "economics and markets"),
    "wildfire smoke": (["air-purifier sales","asthma-clinic visits","mask sales","indoor-gym attendance","flight delays"], "medicine-style diagnosis"),
    "a construction boom": (["cement demand","crane-rental rates","hardware-store sales","permit filings","dumpster rentals"], "economics and markets"),
    "spring bloom": (["beehive activity","pollen counts","nursery sales","lawn-mower use","allergy-clinic visits"], "biology and ecology"),
    "a regional festival": (["hotel occupancy","rideshare demand","street-vendor sales","parking fines","late-night transit use"], "economics and markets"),
}
CAUSAL_MED = [
    ("rain","car skids","a wet road surface","science"),("infection","sweating","a fever","medicine-style diagnosis"),
    ("a price cut","higher revenue","increased units sold","economics and markets"),("exercise","weight loss","a calorie deficit","medicine-style diagnosis"),
    ("more study time","higher test scores","better mastery","science"),("a marketing campaign","more sign-ups","increased site visits","economics and markets"),
    ("smoking","lung damage","accumulated tar","medicine-style diagnosis"),("a software update","fewer crashes","a fixed memory leak","incident and root-cause analysis"),
    ("heavy rain","a flooded basement","a rising water table","incident and root-cause analysis"),("a sugary diet","tooth decay","acid from oral bacteria","medicine-style diagnosis"),
    ("a wage rise","more spending","higher disposable income","economics and markets"),("deforestation","soil erosion","loss of root structure","science"),
    ("a fever","a rapid pulse","raised metabolic demand","medicine-style diagnosis"),("adding a database index","faster page loads","quicker query response","program behavior"),
    ("overfishing","a seabird decline","the collapse of their prey stock","biology and ecology"),("a minimum-wage rise","higher menu prices","increased labor cost","economics and markets"),
    ("a drought","more wildfires","drier vegetation","science"),("caffeine intake","disrupted sleep","delayed melatonin release","medicine-style diagnosis"),
    ("a factory retooling","fewer defects","tighter process tolerances","incident and root-cause analysis"),("a currency devaluation","rising exports","cheaper goods abroad","economics and markets"),
    ("regular watering","a taller plant","sustained cell turgor and growth","biology and ecology"),("a firmware patch","longer battery life","a fixed wake-lock bug","mechanical / systems troubleshooting"),
    ("higher altitude","faster breathing","lower oxygen partial pressure","science"),("a tariff","lower import volume","higher landed cost","economics and markets"),
]
CAUSAL_DIRECT = [
    ("pressing the switch","the light turning on","program behavior"),("adding fertilizer","faster plant growth","science"),
    ("raising the price","fewer units sold","economics and markets"),("taking the antibiotic","the infection clearing","medicine-style diagnosis"),
    ("tightening the valve","the leak stopping","mechanical / systems troubleshooting"),("increasing study hours","a higher exam score","science"),
    ("adding an index","faster query response","program behavior"),("lowering the thermostat","a colder room","engineering and physical systems"),
    ("cutting interest rates","more borrowing","economics and markets"),("applying the brakes","the car slowing","engineering and physical systems"),
    ("adding a catalyst","a faster reaction","chemistry"),("watering the seedling","its germination","biology and ecology"),
    ("closing the relay","the pump starting","program behavior"),("raising the dose","a stronger response","medicine-style diagnosis"),
    ("tightening the bolt","less vibration","mechanical / systems troubleshooting"),("increasing the voltage","a brighter bulb","engineering and physical systems"),
    ("adding cache","faster responses","program behavior"),("raising the tariff","fewer imports","economics and markets"),
    ("heating the gas","higher pressure","chemistry"),("pruning the branch","denser regrowth","biology and ecology"),
]
# ABD_BANK: (domain, subject, [ (distinguishing-signature, planted-cause, [(alt, why-ruled-out), ...]) ])
# BANK: extend to scale abductive.
ABD_BANK = [
 ("mechanical / systems troubleshooting","a portable speaker with no sound",[("paired via Bluetooth; other apps play on the phone","the speaker's amplifier or driver has failed",[("phone muted","other apps play"),("wrong source","it is paired")]),("plays over aux but not Bluetooth","the Bluetooth module has failed",[("dead speaker","wired audio works"),("dead battery","it runs wired")])]),
 ("medicine-style diagnosis","an adult with a sudden rash",[("hours after a new detergent, itchy, on clothed skin","contact dermatitis from the detergent",[("infection","no fever"),("food allergy","it maps to clothed skin")]),("with fever, spreading, after a tick-heavy hike","a tick-borne illness",[("detergent","there is fever"),("heat rash","hike exposure")])]),
 ("incident and root-cause analysis","a dashboard going blank",[("blank in one browser; another shows data","a client-side script/extension issue",[("backend down","another browser works"),("data loss","data shows elsewhere")]),("blank for all after a metrics-agent upgrade","the upgraded agent stopped reporting",[("browser issue","all clients blank"),("network","the app loads")])]),
 ("mechanical / systems troubleshooting","a garage door that will not close",[("reverses before closing; sensor light blinks","a misaligned safety sensor",[("dead motor","it moves"),("broken spring","it travels far")]),("hums but does not move","a stripped drive gear",[("sensor block","no movement"),("power loss","motor hums")])]),
 ("science","a home barometer reading oddly",[("drifts up as the room heats","a temperature-sensitive gauge error",[("real weather","tracks the room"),("altitude","the room is fixed")]),("drops sharply before a storm","a genuine falling-pressure signal",[("gauge fault","aligns with the storm"),("heat","it dropped")])]),
 ("medicine-style diagnosis","a toddler with a limp",[("after a playground fall, tender ankle","a minor sprain",[("infection","clear trauma"),("illness","localized")]),("with fever, no injury, refusing weight","a joint infection needing care",[("sprain","fever, no trauma"),("growing pains","refusal is a red flag")])]),
 ("incident and root-cause analysis","search returning no results",[("empty only for items added today","the index has not re-indexed recent data",[("outage","old items found"),("bad query","common terms fail for new items")]),("empty for all since a config change","the search service is misconfigured",[("indexing lag","old items fail too"),("permissions","all users affected")])]),
 ("mechanical / systems troubleshooting","a coffee machine not brewing",[("powers on, no water flows","a clogged line or failed pump",[("no power","lights on"),("empty tank","tank is full")]),("brews weak and watery","under-extraction from a worn grinder",[("no water","water flows"),("power","it brews")])]),
 ("science","a thermometer disagreeing with others",[("reads high only in direct sun","solar heating of the sensor",[("real heat","shade matches"),("calibration","agrees in shade")]),("reads 2 degrees off everywhere","a calibration offset",[("sun","offset is constant"),("battery","reading is stable")])]),
 ("incident and root-cause analysis","a mobile app showing stale data",[("stale until force-refresh","an over-aggressive local cache",[("server down","refresh fixes it"),("auth","data appears")]),("stale for all despite refresh","the backend feed is not updating",[("local cache","refresh fails"),("client bug","all users see it")])]),
 ("mechanical / systems troubleshooting","a vacuum robot missing spots",[("misses the same corner each run","a mapping blind spot there",[("weak suction","elsewhere is clean"),("battery","it finishes")]),("stops mid-run with a full bin","bin-full sensor halting early",[("mapping","it stops"),("battery","bin is full")])]),
 ("medicine-style diagnosis","a person dizzy on standing",[("only when rising quickly, resolves fast","a brief blood-pressure drop",[("inner-ear","posture-timed"),("dehydration","resolves fast")]),("spinning triggered by head turns","an inner-ear positional cause",[("blood pressure","head-position triggered"),("anxiety","spinning is specific")])]),
 ("incident and root-cause analysis","emails arriving hours late",[("late only to one domain","that domain's server is slow/greylisting",[("our outage","others on time"),("content","one domain")]),("late for all since a queue change","the mail queue is backed up",[("recipient side","all domains late"),("spam filter","they arrive")])]),
 ("science","seeds not sprouting",[("weeks in cold, wet soil","cold soil preventing germination",[("old seed","fresh also failed cold"),("no water","soil is wet")]),("no sprouts despite warmth, old packet","low seed viability",[("cold","it is warm"),("overwatering","moisture is fine")])]),
 ("mechanical / systems troubleshooting","a TV with no picture",[("sound plays, screen black","a backlight/panel failure",[("no signal","audio plays"),("wrong input","sound matches")]),("black screen, no sound, standby light on","standby or wrong input",[("panel failure","no sound too"),("power","standby light on")])]),
 ("medicine-style diagnosis","a cat over-grooming",[("one bald patch during flea season","flea allergy",[("stress","seasonal, localized"),("boredom","physical signs")]),("broadly after a house move","stress from the change",[("fleas","followed the move"),("infection","behavior-linked")])]),
 ("mechanical / systems troubleshooting","a faucet with low pressure",[("low only at the hot tap","sediment in the water heater/hot line",[("main supply","cold is strong"),("faucet","cold works here")]),("low at every tap","a main supply/regulator issue",[("hot line","cold also low"),("one aerator","all taps")])]),
 ("incident and root-cause analysis","a report with missing rows",[("missing only for one region","a filter/join dropping that region",[("data loss","others complete"),("outage","report runs")]),("missing since a timezone change","a date-window shift excluding boundary rows",[("region filter","all regions lost rows"),("permissions","began with config")])]),
 ("mechanical / systems troubleshooting","a laptop touchpad misbehaving",[("cursor jumps while typing","palm contact triggering it",[("driver","typing-correlated"),("hardware","fine when not typing")]),("no response after a spill","liquid damage",[("palm rejection","no response"),("driver","followed the spill")])]),
 ("science","a well turning cloudy",[("cloudy after heavy rain, clears in days","surface runoff after rain",[("contamination","tracks rainfall"),("pump","clears naturally")]),("cloudy with sulfur smell, no rain link","dissolved minerals/bacteria in the aquifer",[("runoff","no rain link"),("pipes","smell points to source")])]),
 ("incident and root-cause analysis","a service with rising memory",[("climbs until a restart resets it","a memory leak",[("traffic","climbs when idle"),("config","only restart clears it")]),("spikes only under heavy upload","large in-memory buffers",[("leak","drops after load"),("restart","none needed")])]),
 ("mechanical / systems troubleshooting","a dishwasher leaving dishes dirty",[("dirty with standing water","a clogged filter/drain",[("no power","it runs"),("detergent","not draining")]),("gritty film on glasses","hard-water scaling",[("drain","it drains"),("power","cycle completes")])]),
 ("incident and root-cause analysis","a checkout button not working",[("no response for mobile users","a mobile touch-handler bug",[("outage","desktop works"),("payment down","the button fails")]),("clicks but payment never confirms","the gateway integration failing",[("button bug","click registers"),("mobile only","all platforms fail")])]),
 ("medicine-style diagnosis","a person with a persistent cough",[("dry, began with a new BP medication","a medication side effect",[("infection","no fever"),("allergy","started with meds")]),("productive, with fever after a cold","a chest infection",[("medication","there is fever"),("allergy","followed infection")])]),
 ("mechanical / systems troubleshooting","a lawn mower that will not start",[("after sitting all winter","stale fuel/gummed carburetor",[("battery","pull-start"),("blade","engine, not blade")]),("starts then dies in seconds","a clogged fuel line",[("stale fuel","it does start"),("plug","it fires")])]),
 ("incident and root-cause analysis","a CI build suddenly failing",[("since a dependency bump","a broken dependency version",[("flaky test","fails consistently"),("runner","it is a build error")]),("intermittently on one test","a flaky timing-dependent test",[("dependency","others pass"),("infra","one test flaps")])]),
 ("engineering and physical systems","lights flickering in a room",[("only when the AC switches on","AC inrush current dropping the voltage",[("bulbs","would flicker regardless"),("outage","tied to the AC")]),("one fixture constantly, others steady","a loose connection there",[("circuit","only one fixture"),("AC load","constant, not load-linked")])]),
 ("medicine-style diagnosis","a person feeling unusually tired",[("pale skin, craving ice","iron-deficiency anemia",[("poor sleep","sleep is normal"),("overwork","pallor and craving")]),("thirsty, urinating often","elevated blood sugar",[("anemia","thirst dominates"),("stress","metabolic signs")])]),
 ("mechanical / systems troubleshooting","a car pulling to one side",[("after hitting a pothole","a knocked wheel alignment",[("tire pressure","followed the impact"),("brakes","pulls while cruising")]),("with fast tire wear","uneven alignment/suspension",[("pothole","gradual"),("camber","persists on flat roads")])]),
 ("science","aquarium plants dying",[("melting leaves after setup","transition shock to new water",[("light","eases as they adapt"),("fish","began at setup")]),("yellowing with glass algae","excess light and nutrients",[("transition","worsens with light"),("temperature","algae tracks light")])]),
 ("mechanical / systems troubleshooting","a wall clock keeping bad time",[("slow and worsening over weeks","a dying battery",[("magnet","steady slowdown"),("humidity","no moisture")]),("stopped after a knock","a jarred movement",[("battery","followed the knock"),("dust","abrupt onset")])]),
 ("medicine-style diagnosis","an office worker with wrist pain",[("worse after long typing","repetitive strain",[("injury","no trauma"),("arthritis","activity-linked")]),("with thumb-side numbness","nerve compression at the wrist",[("strain","numbness pattern"),("cold","positional")])]),
 ("incident and root-cause analysis","a backup failing nightly",[("only when the dataset is large","it exceeds a time or space limit",[("network","small runs pass"),("permissions","it starts")]),("since a path change","a broken target path",[("size","small runs fail too"),("disk","errors on the path")])]),
 ("science","a solar light not turning on",[("dim after cloudy days","insufficient charge from low sun",[("dead LED","works after sun"),("switch","charge-linked")]),("never lights even after sun","a failed cell or sensor",[("clouds","sunny days fail too"),("dirt","panel is clean")])]),
 ("mechanical / systems troubleshooting","a shower running cold fast",[("warm briefly then cold in a big household","an undersized tank",[("broken heater","does warm first"),("valve","volume-linked")]),("cold at one shower only","a failed mixing valve there",[("tank","others stay warm"),("no gas","other taps hot")])]),
 ("medicine-style diagnosis","a plant with spotted leaves",[("dark spreading spots in humid, crowded conditions","a fungal infection",[("underwatering","soil is moist"),("sunburn","spreads between leaves")]),("pale stippling with fine webbing","spider mites",[("fungus","webbing present"),("nutrient","pest pattern")])]),
 ("incident and root-cause analysis","an API responding slowly",[("only for one endpoint","a costly query or missing index there",[("outage","others fast"),("client","server timing high")]),("across all endpoints under load","capacity saturation",[("one query","all slow"),("bug","tracks load")])]),
 ("mechanical / systems troubleshooting","a car pulling to one side",[("after hitting a pothole","a knocked alignment",[("pressure","followed impact"),("brakes","pulls cruising")]),("with fast tire wear","alignment or suspension wear",[("pothole","gradual"),("camber","persists on flat")])]),
 ("incident and root-cause analysis","push notifications not arriving",[("only on one platform","that platform's push service or cert lapsed",[("app down","other platform works"),("network","in-app works")]),("for all since a token change","stale device tokens",[("platform","all platforms fail"),("content","nothing sends")])]),
 ("medicine-style diagnosis","a dog scratching constantly",[("worse in warm months at the tail base","flea allergy",[("dry skin","seasonal, localized"),("boredom","physical signs")]),("with a new food and gut upset","a food sensitivity",[("fleas","followed the diet"),("stress","gut signs too")])]),
 ("mechanical / systems troubleshooting","a washing machine not draining",[("fills and washes but water stays","a clogged pump or filter",[("no power","runs the cycle"),("door lock","cycle proceeds")]),("will not start, door light blinks","a faulty door interlock",[("clog","never starts"),("no water","will not begin")])]),
 ("engineering and physical systems","a smart thermostat misbehaving",[("house overheats while the display reads target","a stuck relay keeping heat on",[("sensor drift","display correct"),("battery","actively heating")]),("never reaches setpoint on cold days","undersized heating or heat loss",[("stuck relay","under-heating"),("wiring","heats, not enough")])]),
 ("medicine-style diagnosis","a dog that stopped eating",[("drooling and pawing at its mouth","a dental problem or oral object",[("illness","mouth-focused"),("stress","drooling local")]),("drinking heavily and lethargic","a systemic illness",[("dental","systemic signs"),("picky","lethargy indicates illness")])]),
 ("mechanical / systems troubleshooting","a phone battery draining fast",[("since a recent app install","the new app runs in the background",[("aged battery","changed with install"),("charger","drain not charging")]),("hot while idle","a stuck background process",[("cold","it is hot"),("brightness","drains idle")])]),
 ("science","a lake fish die-off",[("after a hot still week with algae scum","oxygen depletion from a bloom",[("spill","followed heat/algae"),("disease","many species at once")]),("just downstream of a new pipe","a pollutant from the pipe",[("bloom","localized to the pipe"),("temperature","location points to discharge")])]),
 ("mechanical / systems troubleshooting","a printer with bad output",[("regular horizontal streaks","a dirty or failing drum",[("low ink","periodic, not faded"),("driver","mechanical pattern")]),("faint washed-out pages","low toner",[("drum","uniform fade"),("jam","prints faintly")])]),
 ("incident and root-cause analysis","an app crashing on launch",[("only after the latest update","a bug in the new release",[("storage","crashed post-update"),("network","fails before any request")]),("only on older phones","the update exceeds old-device limits",[("all-device bug","new phones fine"),("network","crashes offline too")])]),
 ("medicine-style diagnosis","a baby crying more than usual",[("pulling at one ear with mild fever","an ear infection",[("hunger","ear-focus, fever"),("teething","fever + ear-pull specific")]),("arching back after feeds","reflux discomfort",[("ear","feed-related"),("colic","post-feed timing")])]),
 ("mechanical / systems troubleshooting","a lawn mower issues",[("dies seconds after starting","a clogged fuel line starving it",[("stale fuel","does start"),("plug","it fires")]),("smokes heavily on start","burning oil from a worn seal or overfill",[("fuel line","it runs"),("battery","pull-start")])]),
 ("incident and root-cause analysis","a database running out of space",[("steady fall after debug logging was on","verbose logs filling the disk",[("data growth","began with log change"),("backup","logs are growing")]),("sudden overnight drop","a large import or runaway job",[("gradual","sudden drop"),("logs","single event")])]),
 ("mechanical / systems troubleshooting","a vacuum losing suction",[("weak with a full bin","a full bin or clogged filter",[("motor","still runs"),("hose","weak not absent")]),("no suction at the head, airflow at the hose","a blockage in the head",[("bin","air moves"),("motor","spins")])]),
 ("science","a greenhouse wilting midday",[("wilts midday on hot days despite wet soil","heat stress outpacing uptake",[("underwatering","soil wet"),("disease","recovers by evening")]),("wilts with dry soil","simple water shortage",[("heat","soil dry"),("pests","no damage")])]),
 ("incident and root-cause analysis","logins rejecting valid passwords",[("only for recent password changes","a caching or sync delay",[("outage","old passwords work"),("network","most succeed")]),("for all since a library update","a bug in the auth library",[("bad passwords","universal"),("network","code error")])]),
 ("medicine-style diagnosis","a runner with shin pain",[("worse after ramping mileage","overuse from the ramp-up",[("shoes","tracks mileage"),("diet","load change")]),("only on hard pavement","impact stress from the surface",[("overuse","surface-specific"),("elsewhere","localized to impact")])]),
 ("mechanical / systems troubleshooting","a gas stove burner not lighting",[("clicks but will not ignite, others work","a clogged igniter port there",[("no gas","others light"),("module","the click is present")]),("no burners light, no click","a power or igniter-module failure",[("one port","nothing clicks"),("no gas","clicking dead too")])]),
 ("incident and root-cause analysis","a reconciliation mismatch",[("equal to a fixed fee per transaction","an unaccounted processing fee",[("fraud","consistent fee"),("timing","scales with count")]),("only for cross-currency payments","an exchange-rate or rounding gap",[("fees","only FX"),("duplicates","close, not doubled")])]),
 ("mechanical / systems troubleshooting","a door that will not close",[("sticks only in humid weather","wood swelling with moisture",[("hinge","tracks humidity"),("frame","eases when dry")]),("scrapes the floor year-round","loose or worn hinges",[("humidity","constant"),("swelling","scrapes at bottom")])]),
 ("science","a well turning salty",[("saltier after heavy pumping near the coast","saltwater intrusion from over-pumping",[("rain","tied to pumping"),("pipes","chemistry points to source")]),("saltier after road de-icing season","runoff of road salt",[("intrusion","seasonal"),("aquifer","tracks de-icing")])]),
 ("incident and root-cause analysis","a dashboard with wrong numbers",[("off since a timezone config change","a timezone mismatch shifting the window",[("data loss","shifted not missing"),("query bug","began with config")]),("one metric doubled overnight","double-counting from a duplicated source",[("growth","implausibly exact"),("timezone","one metric")])]),
 ("mechanical / systems troubleshooting","a wifi connection dropping",[("only when the microwave runs","2.4GHz interference",[("ISP","tracks the microwave"),("router","works otherwise")]),("for far devices only","weak signal at range",[("interference","distance-linked"),("account","near devices fine")])]),
 ("medicine-style diagnosis","a person feeling faint in heat",[("after standing in a hot queue","heat and blood pooling",[("cardiac","clear heat trigger"),("infection","no fever")]),("with heavy sweating and cramps","heat exhaustion from fluid loss",[("faint alone","cramps present"),("virus","exertional/heat context")])]),
 ("finance and business operations","a monthly budget overrun",[("only in the travel line after a policy change","looser travel-approval rules",[("fraud","tracks the policy"),("price rises","confined to one line")]),("across all lines since a vendor switch","higher unit costs from the new vendor",[("one-off","it persists"),("volume","costs rose, not counts")])]),
 ("program behavior","a function returning wrong output",[("only for empty input","an unhandled empty-case branch",[("all inputs","only empty fails"),("type error","valid inputs work")]),("only for very large input","an overflow or precision limit",[("logic bug","small inputs pass"),("memory fault","values wrong, no crash")])]),
 ("program behavior","a web page loading slowly",[("only the first visit per session","a cold cache warming up",[("server load","later loads are fast"),("network","repeat visits are quick")]),("slow for everyone right after a deploy","a regression in the new build",[("cache","all users are slow"),("traffic","tied to the deploy")])]),
 ("engineering and physical systems","a bridge strain sensor drifting",[("drifting with the daily temperature cycle","thermal expansion of the mount",[("real load","it tracks temperature"),("fault","cyclic, not random")]),("a sudden step then steady","a knock that reseated the sensor",[("thermal","the change was abrupt"),("load","a step, not a ramp")])]),
 ("chemistry","a reaction not proceeding",[("the mixture stays cold and unchanged","a missing catalyst",[("wrong reagent","components are correct"),("temperature","added heat still stalls it")]),("it bubbles then stops early","a limiting reagent exhausted",[("catalyst","it did start"),("contamination","the start was clean")])]),
 ("science","a pendulum clock keeping poor time",[("running slow on hot days","a rod lengthened by heat",[("amplitude","temperature-linked"),("pivot","gradual with heat")]),("suddenly erratic after a bump","a loosened pivot",[("heat","the onset was abrupt"),("air current","tied to the bump")])]),
 ("biology and ecology","a beehive losing workers",[("a sudden collapse after nearby spraying","pesticide exposure",[("mites","tied to the spraying"),("queen loss","brood is present")]),("a gradual decline with mite specks","a mite infestation",[("pesticide","no spray event"),("poor forage","the specks are visible")])]),
 ("medicine-style diagnosis","a patient with recurring headaches",[("every morning, easing by noon, with loud snoring","disrupted sleep from apnea",[("tumor","clear time-of-day pattern"),("dehydration","morning-specific")]),("with jaw clicking and worse when chewing","a jaw-joint disorder",[("migraine","chewing-linked"),("sinus","localized to the jaw")])]),
 ("incident and root-cause analysis","orders failing to ship",[("only from one warehouse","a local system or stock issue there",[("carrier","other sites ship"),("payment","paid orders are stuck")]),("for all sites since an address-format change","a validation rejecting the new format",[("one site","all sites affected"),("carrier","it fails before handoff")])]),
 ("mechanical / systems troubleshooting","a refrigerator not cooling",[("the compressor runs constantly, warm inside","a refrigerant leak",[("no power","it runs"),("door seal","it runs nonstop")]),("it clicks periodically but never starts","a failed start relay",[("refrigerant","it never runs"),("thermostat","the clicks are start attempts")])]),
 ("social situations","a friend suddenly distant",[("only since you cancelled plans twice","hurt over the cancellations",[("just busy","timing matches the cancellations"),("unrelated","tied to your actions")]),("withdrawn from everyone after a job loss","personal stress unrelated to you",[("you specifically","it is group-wide"),("anger at you","stress, not blame")])]),
 ("social situations","a team's morale dropping",[("since a well-liked lead left","the loss of trusted leadership",[("pay","tied to the departure"),("workload","began with the exit")]),("only in one sub-team after a reorg","friction from the new structure",[("company-wide","confined to one team"),("season","tied to the reorg")])]),
 ("narrative and discourse","a novel's abrupt tone shift",[("the prose style changing midway and never returning","a co-author or ghostwriter took over",[("intended device","no in-story trigger"),("editing","persistent, not local")]),("only in the flashback chapters","a deliberate device marking the past",[("author change","confined to flashbacks"),("error","a consistent pattern")])]),
 ("narrative and discourse","a witness account that will not add up",[("consistent times but an impossible route","an honest mistake about location",[("lying","other details check out"),("coercion","no sign of pressure")]),("every detail conveniently exonerating them","a rehearsed, self-serving story",[("the plain truth","it is too tidy"),("memory","selective in one direction")])]),
 ("science","a rain gauge reading too low",[("consistently under a nearby station","an obstructed or sheltered site",[("a real dry spell","only this gauge differs"),("calibration","siting explains it")]),("suddenly zero after a storm","a clogged funnel",[("no rain","the station logged rain"),("siting","the onset was abrupt")])]),
 ("medicine-style diagnosis","a child with a recurring stomachache",[("on school mornings, gone on weekends","stress or school avoidance",[("infection","weekend-free"),("diet","calendar-linked")]),("after dairy meals with bloating","lactose intolerance",[("stress","food-timed"),("appendicitis","it recurs benignly")])]),
 ("incident and root-cause analysis","a metrics dashboard flatlining",[("one metric flat since an agent update","that metric's collector broke",[("a real drop","other metrics are normal"),("outage","the app is healthy")]),("all metrics flat overnight","the ingestion pipeline stalled",[("one collector","everything is flat"),("display bug","no data is arriving")])]),
 ("mechanical / systems troubleshooting","a drone drifting in flight",[("always drifting the same direction","a miscalibrated compass or trim",[("wind","consistent direction indoors"),("motor","steady, not erratic")]),("drifting only in gusts","a normal response to wind",[("calibration","gust-linked"),("sensor","an external cause")])]),
 ("engineering and physical systems","a solar array underperforming",[("output dipping only when one panel is shaded","shading dragging down the string",[("inverter","shade-linked"),("wiring","it tracks the sun")]),("a gradual decline over months","dust or soiling on the panels",[("shading","slow and uniform"),("inverter","lower output, no faults")])]),
 ("chemistry","a solution changing color unexpectedly",[("only after air exposure","oxidation of a component",[("contamination","air-linked"),("temperature","exposure-timed")]),("only under the lab lights","a light-sensitive photoreaction",[("air","light-linked"),("heat","it tracks illumination")])]),
 ("biology and ecology","a fish tank clouding overnight",[("days after a fresh setup","a bacterial bloom while cycling",[("overfeeding","setup-timed"),("algae","it is not green")]),("a green tint under strong light","an algae bloom",[("bacteria","green and light-linked"),("waste","it tracks the light")])]),
 ("finance and business operations","a sudden drop in daily sales",[("only online since a checkout change","a broken checkout flow",[("demand","stores are steady"),("season","tied to the change")]),("across all channels after a price rise","reduced demand at the new price",[("checkout","in-store is also down"),("supply","price-timed")])]),
 ("program behavior","intermittent test failures",[("only when tests run in parallel","a shared-state race condition",[("logic bug","it passes serially"),("environment","order-dependent")]),("only on the CI machine","an environment or timezone difference",[("flakiness","it is consistent on CI"),("code","it passes locally")])]),
 ("medicine-style diagnosis","an athlete's dropping performance",[("with breathlessness and pale gums","anemia",[("overtraining","the physical signs"),("diet alone","pallor points to iron")]),("only in heat with heavy sweat","dehydration and heat strain",[("anemia","heat-specific"),("illness","no fever")])]),
 ("incident and root-cause analysis","a spike in cart abandonment",[("only on mobile after a redesign","a broken mobile checkout step",[("price","desktop is steady"),("season","tied to the redesign")]),("across devices at one step","a surprise fee added at that step",[("mobile bug","all devices affected"),("outage","the site works")])]),
 ("narrative and discourse","an essay that loses coherence",[("in one section dense with jargon","padding over a weak argument",[("the topic","it is localized"),("style","only where evidence is thin")]),("throughout after a strong opening","preparation that ran out",[("one section","the decline is global"),("editing","it degrades with length")])]),
 ("social situations","a punctual colleague arriving late",[("only since a change on their commute line","a new transit constraint",[("motivation","external timing"),("illness","tied to the schedule")]),("with fatigue and a short temper","personal stress at home",[("the commute","behavioral signs"),("laziness","distress, not choice")])]),
 ("mechanical / systems troubleshooting","a car that stalls at idle",[("only when cold, fine once warm","a faulty cold-idle control",[("fuel","fine when warm"),("battery","it starts")]),("stalling with the AC on","an electrical load the idle cannot hold",[("cold-idle","load-linked"),("fuel","AC-specific")])]),
 ("engineering and physical systems","a motor overheating",[("only under heavy load","cooling undersized for the load",[("bearing","load-linked"),("power","normal when light")]),("hot even at idle","a failing bearing adding friction",[("load","hot at idle too"),("cooling","present at no load")])]),
 ("chemistry","a pH meter reading off",[("drifting after long use between calibrations","electrode drift needing recalibration",[("real change","it tracks time since calibration"),("temperature","gradual, not stepped")]),("wildly wrong right after storage","a dried-out or fouled electrode",[("drift","the error is large and sudden"),("sample","other samples read wrong too")])]),
]
# MORAL_BANK: (scenario, framework-A, framework-B).  BANK: extend to scale moral-ethical.
MORAL_BANK = [
 ("A self-driving car must choose between risking its passenger and risking two pedestrians.","minimizing total harm","the duty to one's own passenger and the pedestrians' agency"),
 ("A hospital has one ventilator and two patients of equal need, one younger with a better prognosis.","maximizing expected life-years","the fairness of a prior claim and equal moral worth"),
 ("A journalist can publish leaked documents exposing wrongdoing that also endanger an informant.","the public interest in exposing wrongdoing","the duty of care to a person at risk"),
 ("An employee finds the firm pollutes within legal limits but against its public promises.","honesty and public accountability","loyalty, due process, and the livelihoods at stake"),
 ("A parent can give one child tutoring the others in the class cannot afford.","the special obligation to one's child","fairness and equal opportunity for others"),
 ("A rescue team can save five strangers or one close friend, not both.","the impartial value of saving more","the moral weight of special relationships"),
 ("Police ask a shopkeeper for customer footage without a warrant to catch a likely thief.","helping prevent and solve a crime","customers' privacy and due process"),
 ("A scientist can publish now to claim priority or wait to be sure the result holds.","honesty and reliability of the record","the good timely results could do"),
 ("A doctor can respect a competent patient's refusal of life-saving treatment.","the patient's autonomy","the duty to preserve life"),
 ("A city can build needed housing fast by overriding a slow community-consultation process.","the urgent good of more housing","fair process and community consent"),
 ("A friend asks you to lie on their behalf to get a job they may not be qualified for.","loyalty to a friend","honesty to the employer and fairness to applicants"),
 ("A product team can ship a feature that boosts engagement but nudges compulsive use.","business goals and user demand","the duty not to exploit users' psychology"),
 ("A witness's truthful testimony would convict a guilty person but harm an innocent family.","the duty to tell the truth and serve justice","the foreseeable harm to innocents"),
 ("A manager can lay off one person now to likely save ten other jobs, or risk all eleven.","protecting the greater number","the duty to the individual singled out"),
 ("A soldier is ordered to carry out a lawful command they believe is unjust.","duty to lawful authority","individual conscience and the rights of those affected"),
 ("A startup can exaggerate its metrics to secure funding that would save the company.","saving jobs and the venture","honesty to investors and fair markets"),
 ("A developer finds a security flaw; disclosing warns users but also aids attackers pre-fix.","users' right to know the risk","preventing harm by not arming attackers"),
 ("A therapist learns a client may pose a vague future risk to a third party.","the duty to warn and protect others","client confidentiality and trust"),
 ("An engineer is pressured to certify a product safe before testing is complete.","meeting the deadline and cost pressure","the duty to protect users from unproven risk"),
 ("A company can automate roles, cutting costs but displacing loyal workers.","efficiency and competitiveness","responsibility to long-serving employees"),
 ("A parent must decide on a risky but potentially life-changing surgery for their child.","the chance of a much better life","avoiding exposing the child to serious risk"),
 ("A landlord can evict a long-term tenant who fell behind due to illness.","the owner's property rights","compassion for a tenant in hardship"),
 ("A charity can accept a large gift from a donor with a tainted reputation.","the good the funds would do","integrity and endorsing the donor"),
 ("A government can use mass surveillance that would prevent some attacks.","protecting citizens from violence","civil liberties and privacy"),
 ("A teacher can inflate a hardworking but failing student's grade to keep a scholarship.","compassion for the student's future","the integrity of grades and fairness"),
 ("A country can stockpile a scarce vaccine for its citizens first.","duty to one's population","global fairness in a shared crisis"),
 ("A nurse can bend a rigid visiting rule to let a dying patient see family after hours.","compassion and wellbeing","fairness of consistent rules and trust"),
 ("A reporter can protect a source who lied to them, or expose the lie.","the promise of confidentiality","honesty and accountability"),
 ("A firm can use a legal tax loophole that shifts burden onto ordinary taxpayers.","its duty to shareholders and legality","fairness to the wider public"),
 ("A coach can play an injured star who wants to play in a decisive game.","the athlete's autonomy and the team's goal","protecting the athlete's long-term health"),
 ("A store can raise prices sharply on scarce supplies during a local emergency.","letting prices ration scarce goods","fairness and not exploiting people in crisis"),
 ("A voter chooses between a candidate who shares their values but will lose and a winnable one they like less.","expressing genuine convictions","the consequences of who governs"),
 ("A biographer can reveal a dead public figure's private failings that reshape their legacy.","the public's interest in the full truth","respect for the dead and family"),
 ("A doctor can prescribe a placebo that often helps without telling the patient.","the patient's likely improvement","the right to informed, honest care"),
 ("A parent can push a talented child toward a demanding career the child is unsure about.","giving the child their best prospects","the child's autonomy and present wellbeing"),
 ("A city can price roads to cut congestion, burdening low-income drivers.","efficient, cleaner traffic","fairness to those least able to pay"),
 ("A researcher can pursue dual-use work that could cure disease or enable harm.","the potential medical benefit","the risk of catastrophic misuse"),
 ("A manager can keep a toxic top performer whose results carry the team.","the results the team depends on","a healthy, respectful workplace"),
 ("A family can place an aging parent in care against the parent's wish to stay home.","the parent's safety and care","the parent's autonomy and dignity"),
 ("A platform can leave up lawful but hateful speech or remove it.","free expression and open debate","protecting targeted users from harm"),
 ("A pilot can divert to save a sick passenger, delaying two hundred others.","the acute need of one person","the aggregate burden on many"),
 ("A mayor can name and shame tax delinquents to boost compliance.","the public benefit of compliance","the privacy of those named"),
 ("A coder can quietly log keystrokes to fix a rare bug faster.","solving a real problem","users' consent and privacy"),
 ("A parent can forbid a teen from a risky sport they love.","protecting the child","the teen's autonomy"),
 ("A nurse can prioritize a cooperative patient over a hostile one of equal need.","smoother, safer care","equal treatment regardless of behavior"),
 ("A teacher can teach to the test to lift scores that fund the school.","securing needed resources","genuine education"),
 ("A journalist can protect a source who misled them once.","source confidentiality","honesty to readers"),
 ("A landlord can install cameras in common areas for safety.","tenant safety","tenants' privacy"),
 ("A manager can promote a loyal but weaker candidate over a stronger newcomer.","loyalty and morale","merit and fairness"),
 ("A doctor can allocate a scarce drug by lottery.","fairness through equal chance","directing it to those who benefit most"),
 ("A city can ban downtown cars to cut pollution, hurting small shops.","cleaner air for all","affected livelihoods"),
 ("A friend can tell a hard truth that may end the friendship.","honesty and long-term good","the relationship and kindness"),
 ("A company can keep a profitable product a few misuse harmfully.","serving the many","protecting the vulnerable few"),
 ("A soldier can share intel that saves allies but exposes an informant.","protecting allied lives","the duty to the informant"),
 ("A researcher can reuse anonymized patient data without re-consent.","research that helps many","control over one's data"),
 ("A parent can spend savings on one child's rare treatment.","saving a child's life","fairness to the other children"),
 ("A referee can award a soft penalty that is fair on balance.","overall fairness","strict rule-following"),
 ("A startup can build on a rival's public data.","fair competition and user benefit","respect for a rival's effort"),
 ("A voter can back a policy helping their region at the nation's cost.","loyalty to community","the common good"),
 ("A teacher can excuse a plagiarizing refugee student adjusting to a new system.","compassion for hardship","academic integrity"),
 ("A firm can pay legal local wages far below home standards.","providing lawful jobs","a living wage"),
 ("A bystander can post a viral clip that ruins someone over rudeness.","calling out bad behavior","proportionality and dignity"),
 ("A scientist can downplay uncertainty to spur beneficial action.","motivating life-saving action","honesty about what is known"),
 ("A parent can track a young child's location always.","the child's safety","trust and independence"),
 ("A manager can warn a friend before public layoffs.","loyalty to a friend","fairness to all employees"),
 ("A country can deport a long-settled undocumented family per law.","the rule of law","compassion and rootedness"),
 ("A designer can add friction that reduces impulsive purchases.","protecting users from regret","respecting free choice"),
 ("A witness can soften testimony to spare a remorseful youth.","mercy and a second chance","truthful testimony"),
 ("A hospital can offer an unproven therapy to a dying willing patient.","hope and autonomy","evidence-based care"),
 ("A company can offshore support, cutting costs but lowering quality.","lower prices","service quality and jobs"),
 ("A parent can open a teen's diary after signs of distress.","protecting the child","privacy and trust"),
 ("A charity can spend on overhead that makes it far more effective.","greater long-run impact","donors' expectations"),
 ("A driver can speed to rush an injured friend to hospital.","the urgent need","the risk to others"),
 ("A teacher can separate a disruptive child for the class's sake.","the whole class's learning","the child's inclusion"),
 ("A firm can settle a valid claim cheaply by exploiting ignorance.","the firm's interest","honesty and fair dealing"),
 ("A city can use facial recognition to find missing persons.","public safety","surveillance and liberties"),
 ("A friend can keep a secret that might prevent a bad marriage.","respecting a confidence","preventing harm"),
 ("A manager can reject a qualified applicant likely to soon leave.","stability and cost","fairness to the candidate"),
 ("A parent can let a child quit a hard instrument.","the child's happiness","perseverance"),
 ("A platform can shadow-limit lawful misinformation silently.","reducing harm","transparency and speech"),
 ("A nonprofit can use graphic suffering images to raise funds.","raising aid","the dignity of those depicted"),
 ("A worker can report a colleague's small ongoing theft.","honesty and the employer","loyalty and proportionality"),
 ("A teacher can give a lenient deadline only to a struggling student.","meeting real need","equal treatment"),
 ("A company can use an aggressive but legal offshore tax structure.","duty to shareholders","fair contribution"),
 ("A parent can share an embarrassing but heartwarming story of a child online.","sharing joy","the child's consent"),
 ("A rescuer can prioritize the youngest in disaster triage.","maximizing years saved","equal worth"),
 ("A journalist can withhold a true story to avoid inflaming tensions.","preventing violence","the right to know"),
 ("A firm can automate a beloved craft role to compete.","survival and lower prices","the dignity of skilled work"),
 ("A doctor can soften bad news at a family's request.","compassion and their wishes","the patient's right to know"),
 ("A city can clear an informal settlement for infrastructure.","the public benefit","residents' homes and rights"),
 ("A friend can lend money they doubt will be repaid to someone in crisis.","compassion","prudence and security"),
 ("A company can sell anonymized data to keep a service free.","a free service for all","data ownership"),
 ("A parent can press vaccination on a hesitant adult child.","family and public health","an adult's autonomy"),
 ("A teacher can report a student's confession of home abuse.","protecting the child","trust and confidentiality"),
 ("A manager can deny remote work fairly but hard on a caregiver.","consistent policy","individual compassion"),
 ("A scientist can patent a life-saving invention to fund research.","funding breakthroughs","affordable access now"),
 ("A voter can support painful austerity for long-term stability.","long-term stability","hardship on the vulnerable"),
 ("A hospital can let students practice on poorly-informed willing patients.","training doctors","informed consent"),
 ("A firm can drop support for an old product many still use.","focusing on the future","duty to existing customers"),
 ("A parent can choose a safer distant school over a local one with friends.","safety and prospects","social ties and voice"),
 ("A community can require volunteer service for benefits.","shared responsibility","freedom and burden on the struggling"),
 ("A doctor can prescribe a cheaper drug with more side effects.","affordability and sustainability","the patient's comfort"),
 ("A manager can keep a toxic top performer who carries results.","results the team needs","a healthy workplace"),
 ("A city can lure a big employer with tax breaks at public cost.","jobs and growth","fair use of public money"),
 ("A firm can use persuasive design to keep forgetful subscribers.","revenue and service","honesty and user intent"),
 ("A doctor can allocate a transplant by need and success odds.","maximizing benefit","equal claim"),
 ("A government can censor a violent extremist site.","preventing incitement","free speech and overreach"),
 ("A parent can require a teen's social-media passwords.","oversight and safety","growing autonomy"),
 ("A company can relocate to a low-tax state, defunding its old community.","financial survival","loyalty to the community"),
 ("A rescuer can risk their life for a stranger against orders.","the endangered life","protocol and safety"),
 ("A doctor can perform a procedure a patient wants but does not need.","autonomy","avoiding needless risk"),
 ("A regulator can approve a drug faster with thinner evidence in a crisis.","speed that saves lives","safety and rigor"),
 ("A teacher can lower standards to keep struggling students enrolled.","access and encouragement","the value of the credential"),
 ("A hospital can publish surgeon-level success rates that may deter them from hard cases.","patient transparency","fair incentives to treat the sickest"),
 ("An editor can run a correct but unverifiable tip from an anonymous source.","the public's right to timely truth","verification and accountability"),
 ("A city can install speed cameras that fine mostly low-income commuters.","fewer road deaths","fairness to those least able to pay"),
 ("A manager can read an employee's work chat logs after a leak.","protecting the company","worker privacy and trust"),
 ("A doctor can enroll a dying patient in a trial that mostly helps future patients.","advancing treatment for many","the patient's own best interest"),
 ("A charity can spend on lobbying that could unlock far larger public funding.","greater long-run impact","donor intent for direct aid"),
 ("A startup can keep a dark-pattern cancel flow that boosts retention.","revenue that sustains the product","honest, easy user choice"),
 ("A scientist can share raw data that could be misread to cause public panic.","open science and transparency","preventing foreseeable harm"),
 ("A parent can vaccinate a child over the other parent's objection.","the child's and public health","shared parental authority"),
 ("A judge can impose a lighter sentence to spare a defendant's dependent children.","mercy and the children's welfare","equal treatment under law"),
 ("A firm can relocate a polluting plant to a poorer region with weaker rules.","jobs and lower costs","environmental justice"),
 ("A coach can bench a star to enforce a team rule before a crucial match.","consistent discipline","the team's chance to win"),
 ("A nurse can override a doctor's order she believes will harm a patient.","patient safety","the chain of clinical authority"),
 ("A landlord can rent to the highest bidder over a long-waiting local family.","efficient use of property","community and fairness"),
 ("A developer can ship an accessibility-poor app on time or delay for inclusion.","meeting commitments to most users","equal access for disabled users"),
 ("A government can release a prisoner early to ease overcrowding.","humane conditions and cost","public safety and the sentence served"),
 ("A biographer can honor a subject's request to omit a formative scandal.","the subject's dignity and wishes","a truthful record"),
 ("A teacher can let a gifted student skip ahead, straining the class's cohesion.","the student's potential","the group's shared progress"),
 ("A company can use a customer's data to prevent a likely fraud against them.","protecting the customer","consent and data limits"),
 ("A pilot can fly through marginal weather to avoid stranding passengers.","meeting travelers' needs","an ample safety margin"),
 ("A regulator can grandfather an unsafe-by-modern-standards but widely used product.","stability and avoiding disruption","holding all products to current safety"),
 ("A parent can donate a child's college fund to save many lives abroad.","the greater good for strangers","a specific promise to one's child"),
 ("A manager can quietly counteroffer to keep a leaving employee, unsettling peers' pay.","retaining key talent","pay fairness across the team"),
 ("A city can use eminent domain for a park that serves thousands.","broad public benefit","the rights of displaced owners"),
 ("A doctor can respect a teenager's wish to keep a diagnosis from strict parents.","the young patient's trust","the parents' role and rights"),
 ("A firm can automate content moderation, cutting jobs but reducing worker trauma.","protecting workers from harm","the livelihoods lost"),
 ("A scientist can accept industry funding that speeds vital research but risks bias.","faster progress","the integrity of the findings"),
 ("A teacher can report suspected but unproven cheating that could end a scholarship.","academic integrity","the cost of a false accusation"),
 ("A platform can down-rank a truthful post that is fueling a dangerous panic.","reducing real-world harm","not suppressing true speech"),
 ("A hospital can prioritize vaccinating staff over higher-risk patients.","keeping the system running","protecting the most vulnerable first"),
 ("A parent can let a child face a natural consequence that will hurt but teach.","long-term growth","preventing present harm"),
 ("A firm can honor a mistaken low price it advertised to thousands.","honesty and goodwill","fairness to the business and its staff"),
 ("A mayor can divert flood defenses to protect a dense downtown over scattered farms.","protecting the most people","not sacrificing a minority's homes"),
 ("A researcher can withhold a method that is beneficial but easily weaponized.","preventing misuse","open access that also helps defenders"),
 ("A worker can unionize quietly, risking a struggling employer that treats them well.","collective bargaining rights","loyalty to a fair employer in hardship"),
 ("A doctor can give a scarce bed to a patient more likely to recover.","maximizing lives saved","equal claim regardless of prognosis"),
 ("A parent can enroll a shy child in intensive therapy against the child's protest.","the child's future wellbeing","the child's present autonomy"),
 ("A company can keep selling a safe product in a market that misuses it culturally.","respecting a lawful market","responsibility for foreseeable misuse"),
 ("A teacher can spend limited time on the few failing students or the many average ones.","lifting those most at risk","the greatest total gain"),
 ("A city can name a whistleblower in records requests as the law seems to require.","legal transparency","protecting someone who exposed wrongdoing"),
 # --- bank extension (2026-07): additional distinct dilemmas across settings ---
 ("An AI lab can release a capable open model that helps researchers but also lowers the bar for misuse.","open access and scientific progress","preventing foreseeable large-scale harm"),
 ("A hospital can use a triage algorithm that is more accurate overall but slightly worse for a minority group.","maximizing lives saved across everyone","equal quality of care for every group"),
 ("A software team can log detailed user sessions to fix bugs faster without asking each user.","shipping a more reliable product","informed consent over personal data"),
 ("A city can deploy predictive policing that lowers crime but concentrates stops in poorer areas.","reducing overall victimization","fairness and freedom from disproportionate scrutiny"),
 ("A founder can tell employees the company is fine to prevent panic while quietly seeking a buyer.","protecting jobs by avoiding a run","honesty owed to the people who depend on the firm"),
 ("A translator can soften a dying patient's blunt words to comfort the family.","kindness to grieving relatives","faithfulness to what the patient actually said"),
 ("A researcher can exclude an outlier that would weaken a result they believe is genuinely correct.","advancing a likely-true finding","the integrity of reporting all the data"),
 ("A teacher can quietly give a struggling student extra time on tests without telling the class.","meeting a real individual need","transparency and equal rules for all"),
 ("A journalist can pay a source for information that would expose serious corruption.","exposing wrongdoing in the public interest","the integrity risk of paid testimony"),
 ("A game studio can add loot boxes that fund the game but resemble gambling for minors.","funding continued development","protecting young players from exploitative design"),
 ("A doctor can honor a patient's wish to stop dialysis knowing it will end their life.","respecting autonomy over one's own body","the professional duty to preserve life"),
 ("A charity can photograph beneficiaries in distress to raise more funds for them.","raising aid that materially helps","the dignity and consent of those depicted"),
 ("A regulator can approve a cheaper generic drug slightly less consistent than the brand.","broadening access through lower cost","uniform quality assurance for every patient"),
 ("A manager can assign the best projects to a rising star, starving steady performers of growth.","maximizing the team's output","fair development opportunities for all"),
 ("A city can use eminent domain to route a rail line through a historic neighborhood.","transit that serves the whole region","the rights and roots of displaced residents"),
 ("A scientist can accept a fast-tracked review that speeds publication but skips replication.","getting useful results to the field sooner","the reliability that replication protects"),
 ("A parent can veto a teenager's gender-affirming request pending more time.","cautious protection of a minor","respect for the young person's identity and voice"),
 ("A bank can deny a loan using a model that is accurate but opaque to the applicant.","sound, data-driven lending","the applicant's right to an explanation"),
 ("A newsroom can publish a politician's leaked medical records relevant to fitness for office.","the public's interest in a leader's capacity","the individual's medical privacy"),
 ("A company can keep manufacturing in a region despite lax safety to preserve local jobs.","the livelihoods that depend on the plant","worker safety to a higher standard"),
 ("A teacher can report a colleague's outdated but not yet harmful teaching methods.","students' long-term learning","loyalty and proportionality toward a peer"),
 ("A platform can auto-translate hate speech to reach moderators, exposing staff to more of it.","catching harmful content faster","protecting moderators from psychological harm"),
 ("A city can fluoridate water for public dental health over the objection of some residents.","population-level health benefit","individual consent to medical treatment"),
 ("A startup can use a competitor's leaked pricing to win a critical deal.","the survival advantage it confers","fair dealing and respect for others' confidential work"),
 ("A hospital can let a family withhold a terminal diagnosis from an elderly patient.","cultural respect and family wishes","the patient's right to know their condition"),
 ("A developer can ship an addictive streak feature that boosts learning-app retention.","keeping learners engaged and progressing","not exploiting compulsion loops"),
 ("A government can release anonymized census microdata useful to researchers but re-identifiable.","the public value of open data","citizens' protection from re-identification"),
 ("A coach can quietly rest a young athlete against the parents' win-now demands.","the athlete's long-term health","respecting the family's authority and goals"),
 ("A firm can offer a lower wage to a desperate applicant who would accept it.","a lawful, mutually agreed deal","fairness to someone with weak bargaining power"),
 ("A doctor can prescribe off-label for a condition with no approved treatment.","a real chance to help a suffering patient","staying within tested, approved uses"),
 ("A city can install gunshot-detection sensors that also capture ambient conversation.","faster response to violence","residents' privacy from ambient surveillance"),
 ("A teacher can let a talented but rule-breaking student compete despite a code violation.","rewarding genuine achievement","consistent enforcement of shared rules"),
 ("A company can quietly patch a security hole without disclosing it was ever exploited.","avoiding panic and reputational harm","affected users' right to know they were exposed"),
 ("A parent can enroll a child in a clinical trial that mainly benefits future patients.","contributing to cures that help many","the child's own best interest and limited consent"),
 ("A relief agency can negotiate with an armed group to reach starving civilians.","getting aid to people who will otherwise die","not legitimizing or funding violent actors"),
 ("A manager can use a personality test that screens out some qualified neurodivergent applicants.","a cheaper, standardized hiring filter","equal opportunity regardless of neurotype"),
 ("A city can price water higher in drought to force conservation, straining poor households.","protecting a scarce shared resource","affordability of a basic necessity"),
 ("A scientist can publish a method to edit heritable genes that could cure or be abused.","opening a path to end genetic disease","guarding against irreversible misuse"),
 ("A company can offer a refund only to customers who complain, keeping quiet ones' money.","a lawful policy that rewards diligence","honest fairness to every affected customer"),
 ("A doctor can break confidentiality to warn a partner of a serious infectious risk.","preventing foreseeable harm to a third party","the patient's confidentiality and trust"),
 ("A teacher can grade anonymously, losing context that would help a struggling student.","impartial, bias-free grading","responsiveness to individual circumstances"),
 ("A firm can automate a warehouse, raising safety and cutting many stable jobs.","fewer injuries and lower prices","the livelihoods of long-serving workers"),
 ("A city can grant a homeless encampment a sanctioned site near unwilling residents.","shelter and dignity for the unhoused","the concerns of nearby homeowners"),
 ("A researcher can share code that reproduces results but reveals a collaborator's unpublished idea.","open, reproducible science","credit and consent owed to a collaborator"),
 ("A parent can let a mature 15-year-old take a gap job abroad against school advice.","the teen's growing autonomy and initiative","protection and the value of finishing school"),
 ("A platform can down-rank sensational but lawful content to improve discourse.","healthier public conversation","neutral treatment of lawful speech"),
 ("A doctor can accept a patient's refusal of a blood transfusion on religious grounds.","respect for deeply held belief and autonomy","the duty to prevent an avoidable death"),
 ("A company can keep a legacy product alive for a few dependent hospitals at a loss.","duty to critical existing users","responsible use of shareholders' capital"),
 ("A city can ticket jaywalking to cut pedestrian deaths, burdening poorer neighborhoods more.","fewer traffic fatalities","equitable, non-punitive enforcement"),
 ("A journalist can honor an embargo that delays a safety-relevant story.","trust that keeps sources talking","the public's timely access to safety information"),
 ("A manager can hire a relative who is genuinely the best candidate.","choosing the strongest applicant","the appearance and risk of favoritism"),
 ("A teacher can use an AI detector that sometimes falsely flags honest students.","deterring and catching cheating","protecting the innocent from false accusations"),
 ("A company can sell an aging product to a firm that will discontinue support.","a good return for shareholders","continuity for customers who rely on it"),
 ("A city can require energy retrofits that cut emissions but raise rents.","climate benefit for everyone","affordability for current tenants"),
 ("A doctor can enroll only English speakers in a trial to simplify consent.","cleaner, faster study logistics","equitable access to research for all groups"),
 ("A parent can share a child's medical journey online to build a support network.","community and solidarity in hardship","the child's future privacy and consent"),
 ("A firm can use dynamic pricing that charges loyal customers more than new ones.","revenue that funds the service","fairness to committed customers"),
 ("A regulator can let a struggling bank hide losses briefly to avoid a panic.","financial stability for depositors","transparency owed to markets and the public"),
 ("A teacher can let students use calculators, easing frustration but weakening fluency.","reducing anxiety and access barriers","building durable underlying skills"),
 ("A city can offer tax breaks to keep a major employer, at the cost of school funding.","preserving jobs and the tax base","adequate resources for public education"),
 ("A researcher can withhold negative trial results a sponsor dislikes.","continued funding for the lab","honest, complete reporting to science"),
 ("A doctor can give a frightened patient a smaller true risk figure to secure consent.","obtaining consent for a beneficial procedure","full and honest disclosure of risk"),
 ("A company can require arbitration clauses that quietly limit customers' legal options.","predictable, lower dispute costs","customers' access to the courts"),
 ("A parent can push a shy child into public performance to build confidence.","fostering growth and resilience","the child's present comfort and consent"),
 ("A city can adopt facial recognition to speed transit but track riders' movements.","convenience and efficiency for millions","riders' freedom from routine tracking"),
 ("A teacher can spend a windfall grant on the top students most likely to excel.","maximizing measurable achievement","fairness to those with the greatest need"),
 ("A firm can meet a diversity target by lowering a bar for one role.","broadening representation and opportunity","uniform standards and fairness to all applicants"),
 ("A doctor can prioritize a compliant patient over a hostile one with the same need.","smoother, safer clinical care","equal treatment regardless of behavior"),
 ("A journalist can quote an off-record remark that reveals a major public danger.","warning the public of real risk","the promise implied by going off the record"),
 ("A company can keep a profitable ad model that tracks children's behavior.","revenue that keeps a free service running","special protection of children's data"),
 ("A city can clear tents before winter into shelters some residents refuse.","preventing exposure deaths","respecting the autonomy of those who refuse"),
 ("A researcher can use a captured dataset scraped without clear consent.","a valuable resource for public-good research","the consent and rights of the data subjects"),
 ("A manager can deny a raise that is fair but would break the team's pay bands.","internal consistency and equity","rewarding one person's genuine contribution"),
 ("A doctor can recommend a costly test with a small chance of catching a serious disease.","catching a rare but grave condition early","avoiding overtreatment and wasted resources"),
 ("A teacher can bar a disruptive but curious student from a field trip.","the group's safety and learning","the individual student's inclusion and growth"),
 ("A firm can green-light a feature that helps most users but excludes those on old devices.","progress for the majority","not stranding users who cannot upgrade"),
 ("A city can subsidize electric cars, aiding the middle class more than the poor.","accelerating emissions cuts","equitable distribution of public benefit"),
 ("A parent can accept a scholarship that requires the child to move far from family.","the child's expanded opportunity","family closeness and support"),
 ("A scientist can co-author with a powerful figure who contributed little, easing publication.","smoothing the path for good work","honest attribution of credit"),
 ("A company can retain data indefinitely in case it is useful later.","future analytical and safety value","data minimization and users' expectations"),
 ("A doctor can defer to a hospital cost-control rule that delays a helpful scan.","sustainable, system-wide stewardship","the individual patient's timely care"),
 ("A teacher can post exemplary student work publicly to inspire others.","celebrating and motivating achievement","the student's consent and privacy"),
 ("A city can allow a noisy night market that boosts the economy near homes.","local livelihoods and vibrancy","residents' rest and quiet enjoyment"),
 ("A firm can keep an underperforming employee who is a sole earner supporting a family.","compassion for real hardship","fairness to the team and the business"),
 ("A researcher can run a deception study that yields insight but misleads participants.","knowledge that can help many","honesty and respect toward participants"),
 ("A parent can limit a child's screen time strictly against the child's strong wishes.","the child's development and wellbeing","the child's autonomy and trust"),
 ("A company can comply with a foreign government's data request to keep operating there.","serving millions of users in that market","protecting individual users from state overreach"),
 ("A doctor can allocate the last ICU bed to a younger patient over an older one of equal need.","expected life-years saved","the equal moral worth of each patient"),
 ("A teacher can excuse a star athlete's absence that a regular student would be marked for.","supporting a valued representative of the school","consistent rules applied to everyone"),
 ("A city can permit a data center that brings jobs but strains the local water supply.","economic growth and employment","sustainable use of a shared resource"),
 ("A firm can quietly A/B test a price increase on unwitting customers.","learning what the market will bear","honest treatment of experimental subjects"),
 ("A scientist can accept a defense contract that funds basic research with military uses.","resources for open scientific advance","complicity in potential military harm"),
 ("A parent can refuse a school's request to medicate a restless child.","caution about medicating a young child","the child's ability to learn and belong"),
 ("A company can offshore data to a cheaper jurisdiction with weaker privacy law.","lower costs that keep the service affordable","stronger protection of user data"),
 ("A doctor can tell a white lie that a placebo is a strong medicine to help a patient.","the real relief the placebo may bring","the patient's right to truthful care"),
 ("A city can require businesses to report undocumented workers to enforce labor law.","upholding lawful employment rules","the safety and rootedness of vulnerable workers"),
 ("A teacher can let a grieving student skip a major exam with no makeup penalty.","compassion in a time of loss","fairness to classmates held to the schedule"),
 ("A firm can adopt a returns policy that curbs abuse but hurts honest edge cases.","preventing costly fraud","fair treatment of genuine customers"),
 ("A researcher can publish a security flaw in medical devices already in patients.","warning hospitals and patients to act","not handing attackers a ready exploit"),
 ("A parent can decline a risky experimental surgery their child's doctors recommend.","avoiding exposing the child to grave risk","following expert medical judgment"),
 ("A company can use unpaid interns for real work that it cannot otherwise afford.","offering experience and a foot in the door","fair pay for genuine labor"),
 ("A city can ration a scarce vaccine by lottery rather than by risk.","equal chance and simplicity","directing doses where they save the most lives"),
 ("A doctor can spend extra unbilled time with one patient, delaying a full waiting room.","attentive care for a patient in need","fairness to everyone else waiting"),
 ("A teacher can teach a controversial but accurate topic parents object to.","students' access to truthful knowledge","respect for families' values and role"),
 ("A firm can lay off staff by seniority, protecting veterans but not performance.","loyalty to long-serving employees","retaining those who contribute most now"),
 ("A city can allow short-term rentals that aid owners but shrink long-term housing.","property owners' freedom and income","affordable housing for residents"),
 ("A scientist can slow a promising line of work over a small but serious safety doubt.","caution against a low-probability catastrophe","the many who could benefit from progress"),
 ("A parent can insist a capable adult child repay a loan during the child's hardship.","fairness and the value of keeping promises","compassion for a struggling family member"),
 ("A company can use a supplier with lower prices but a poor human-rights record.","lower costs that benefit customers","not profiting from others' mistreatment"),
 ("A doctor can withhold a grim prognosis a patient explicitly said they do not want.","respecting the patient's stated wishes","the value of honest, complete information"),
 ("A city can build a shelter quickly by waiving normal accessibility standards.","housing people before winter arrives","equal access for disabled residents"),
 ("A teacher can allow AI writing tools that help some but blur authorship.","access and support for diverse learners","developing and assessing genuine skill"),
 ("A firm can settle a meritless nuisance suit cheaply rather than fight on principle.","saving money and management time","not rewarding or inviting bad-faith claims"),
 ("A researcher can anonymize and reuse interview data for a new unrelated study.","extracting more value from hard-won data","the scope of consent participants gave"),
 ("A parent can move the family for a better job, uprooting a child's friendships.","the family's economic security","the child's stability and social ties"),
 ("A company can require weekend on-call that strains staff to guarantee uptime.","reliable service customers depend on","employees' rest and personal lives"),
]

# --------------------------------------------------------------------------
# Corpus IO
# --------------------------------------------------------------------------
def load_corpus(data_dir):
    existing = {t: set() for t in RT}
    # Global next-id per type. New records are numbered above the highest id
    # seen anywhere in the corpus - train, eval, raw, AND negatives - so an
    # appended batch can never collide with a pre-gate raw candidate id (raw
    # batches continue the same numeric run) or any other file.
    max_id = {t: 999 for t in RT}
    counts = {t: {"train": 0, "eval": 0, "neg": 0} for t in RT}
    for f in glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True):
        loc = "eval" if ".eval." in f else "train" if ".train." in f else "neg" if os.sep+"negatives"+os.sep in f else "other"
        for line in open(f):
            line = line.strip()
            if not line: continue
            r = json.loads(line); t = r["reasoning_type"]; existing[t].add(r["problem"])
            if loc in counts[t]: counts[t][loc] += 1
            m = re.match(r'^[a-z]{3}-(\d{6})(?:-neg)?$', r["id"])
            if m:
                max_id[t] = max(max_id[t], int(m.group(1)))
    return existing, max_id, counts

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser(description="Generate a reasoning-data batch (default 500/type, dry run).")
    ap.add_argument("--per-type", type=int, default=500, help="new records to add per reasoning type (a '500-shot' batch)")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed; change per season for fresh items")
    ap.add_argument("--root", default=default_root, help="repo root containing DOMAINS.md and data/")
    ap.add_argument("--train-frac", type=float, default=0.8, help="fraction of new records that go to train")
    ap.add_argument("--neg-frac", type=float, default=0.25, help="paired negatives per verifiable type, as fraction of its new train records")
    ap.add_argument("--created", default=datetime.date.today().isoformat(), help="ISO date stamped on provenance.created")
    ap.add_argument("--write", action="store_true", help="actually append to data/ (default: dry run)")
    ap.add_argument("--report-only", action="store_true", help="print current corpus stats and exit")
    ap.add_argument("--only", default="", help="comma-separated reasoning types to (re)generate; default all")
    ap.add_argument("--replace", action="store_true",
                    help="overwrite the affected types' curated files and renumber ids from 1000, "
                         "instead of appending. Use with --only for a clean regeneration of one type.")
    args = ap.parse_args()

    data_dir = os.path.join(args.root, "data")
    canon = load_canonical_domains(args.root)
    existing, max_id, counts = load_corpus(data_dir)

    if args.report_only:
        print("Current corpus:")
        for t in RT:
            c = counts[t]; print(f"  {t:15s} train={c['train']:4d} eval={c['eval']:4d} neg={c['neg']:4d}")
        return

    only = [t.strip() for t in args.only.split(",") if t.strip()] or list(RT)
    bad = [t for t in only if t not in RT]
    if bad: raise SystemExit(f"--only has unknown types: {bad}")

    rng = random.Random(args.seed)
    new_files = {}          # rel path -> list of records
    replace_files = set()   # rel paths to overwrite rather than append
    report = []
    for t in only:
        # In replace mode regenerate the whole type: don't dedup against the
        # copy already on disk (we are overwriting it) and renumber from 1000.
        excl = set() if args.replace else existing[t]
        pool = BUILDERS[t](args.per_type, rng, excl)
        # Items may carry an explicit structural 'split' (analogical does, to keep
        # train/eval answer pools disjoint). Otherwise fall back to a fraction slice.
        tagged = any("split" in it for it in pool)
        produced = len(pool) if tagged else min(len(pool), args.per_type)
        chosen = pool if tagged else pool[:produced]
        if tagged:
            n_train = sum(1 for it in chosen if it.get("split") == "train")
            n_eval = len(chosen) - n_train
        else:
            n_eval = int(round(produced * (1 - args.train_frac)))
            n_train = produced - n_eval
        gid = 1000 if args.replace else max_id[t] + 1  # rising counter above existing ids
        pre = PREFIX[t]; new_train = []
        doms = set()
        for i, it in enumerate(chosen):
            it["_created"] = args.created
            split = it.get("split") or ("train" if i < n_train else "eval")
            num = gid + i
            rid = f"{pre}-{num:06d}"
            rec = make(rid, t, it["domain"], it["problem"], it["steps"], it["final"], True, it["difficulty"],
                       GEN_METHOD[t], it["vm"], True, it["vd"], SOURCE[t], None, args.created, confidence=it.get("confidence"))
            errs = validate(rec, "curated", canon)
            if errs: raise SystemExit(f"VALIDATION FAIL {rid}: {errs}")
            relpath = f"curated/{t}.{split}.jsonl"
            new_files.setdefault(relpath, []).append(rec)
            if args.replace: replace_files.add(relpath)
            doms.add(it["domain"])
            if split == "train": new_train.append((rid, it))
        made_neg = 0
        if t in VERIFIABLE:
            target_neg = int(round(len(new_train) * args.neg_frac))
            for rid, it in new_train:
                if made_neg >= target_neg: break
                it["_created"] = args.created
                nr = NEG_BUILDERS[t](rid, it)
                if nr is None: continue
                errs = validate(nr, "negatives", canon)
                if errs: raise SystemExit(f"VALIDATION FAIL {nr['id']}: {errs}")
                neg_rel = f"negatives/{t}.negatives.jsonl"
                new_files.setdefault(neg_rel, []).append(nr); made_neg += 1
                # In --replace mode the negatives file must be OVERWRITTEN too, or
                # stale negatives that point at now-renumbered positive ids linger.
                if args.replace: replace_files.add(neg_rel)
        report.append((t, args.per_type, produced, n_train, n_eval, made_neg, len(doms)))

    # report
    print(f"{'type':16s} {'req':>4s} {'made':>4s} {'train':>5s} {'eval':>4s} {'neg':>4s} {'domains':>7s}")
    short = []
    for t, req, made, tr, ev, ng, dm in report:
        flag = "" if made >= req else "  <-- SHORT (bank-limited; extend BANK to scale)"
        if made < req: short.append(t)
        print(f"{t:16s} {req:4d} {made:4d} {tr:5d} {ev:4d} {ng:4d} {dm:7d}{flag}")
    total = sum(len(v) for v in new_files.values())
    curated_added = sum(tr + ev for _, _, _, tr, ev, _, _ in report)
    neg_added = sum(ng for _, _, _, _, _, ng, _ in report)
    zero = [t for t, _, _, tr, ev, _, _ in report if tr + ev == 0]
    print(f"\nnew records this batch: {total} (seed={args.seed}, created={args.created})")
    print(f"  curated positives: +{curated_added}    paired negatives: +{neg_added}")
    if short:
        print("bank-limited types (produced < requested):", ", ".join(short))
    if zero:
        print("ADDED NOTHING to curated (every candidate already on disk). For a "
              "parametric type change --seed; for an authored type extend its BANK:",
              ", ".join(zero))

    if not args.write:
        print("\nDRY RUN - nothing written. Re-run with --write to append.")
        return
    for rel, recs in sorted(new_files.items()):
        path = os.path.join(data_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "w" if rel in replace_files else "a"
        with open(path, mode) as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if replace_files:
        print("REPLACED (overwritten):", ", ".join(sorted(replace_files)))
    print(f"\nWROTE {total} records to {data_dir}")

if __name__ == "__main__":
    main()
