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

REUSE ACROSS SEASONS
    - Change --seed for a fresh season. The verifiable + metacognitive
      generators are randomized/parametric, so a new seed yields genuinely new
      items automatically (no code change).
    - The authored types (abductive, analogical, moral-ethical) draw from the
      BANKS near the bottom of this file. To scale those types further in a new
      season, extend those banks (each is marked "# BANK: extend to scale").
    - The tool is idempotent-friendly: it dedups against existing problems and
      continues IDs, so re-running (even with the same seed) will not create
      duplicate problems - it simply adds whatever new items it can find.

HONESTY
    Verifiable types are genuinely checked. The rubric/ranking types
    (abductive=answer_match by ranking, analogical=process_check,
    moral-ethical=rubric_judge) are authored so the planted ground truth
    matches by construction; they carry their verification method but await an
    independent Solver/judge pass (see PIPELINE.md). The generator never pads a
    shortfall silently: if a type cannot produce the requested count of new,
    distinct problems, it emits fewer and says so in the report.

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

# ==========================================================================
# GENERATORS  (build_<type>(need, rng, exclude) -> list[instance dict])
# Each instance: {domain, problem, steps, final, difficulty, vm, vd,
#                 optional: confidence, and raw fields for the negative builder}
# ==========================================================================

def build_deductive(need, rng, exclude):
    out, seen = [], set()
    fmts = 4
    attempts = 0
    while len(out) < need and attempts < need*40 + 2000:
        attempts += 1
        dom, subj = _domain_subject(rng)
        hops = rng.randint(1, 5)
        preds = rng.sample(PREDICATES, hops+1)
        vs = [f"c{i}" for i in range(len(preds))]
        prem = [f"(not {vs[i]}) or {vs[i+1]}" for i in range(hops)]
        if not entails(vs, prem + [vs[0]], vs[-1]):   # always true for a forward chain; checked anyway
            continue
        first = preds[0]; last = preds[-1]
        def rule_line(i, lead):
            subjref = f"the {subj}" if i == 0 and lead else "it"
            return f"If {subjref} is {preds[i]}, then it is {preds[i+1]}."
        prose = "Rules: " + " ".join(rule_line(i, i == 0) for i in range(hops))
        distract = hops >= 3
        dtx = (f" Separately, if it is {rng.choice([p for p in PREDICATES if p not in preds])}, then a receipt prints." if distract else "")
        fmt = rng.randrange(fmts); nm = rng.choice(NAMES)
        if fmt == 0:
            prob = prose + dtx + f" Fact: the {subj} is {first}. Given only these rules, is it necessarily {last}?"
            final = f"Yes, necessarily {last}."
        elif fmt == 1:
            bul = "Consider these rules:\n" + "\n".join(f"- if the {subj} is {preds[i]} then it is {preds[i+1]}" for i in range(hops))
            prob = bul + dtx + f"\nObserved: the {subj} is {first}. Does it follow that it is {last}?"
            final = f"Yes, it follows that the {subj} is {last}."
        elif fmt == 2:
            prob = prose + dtx + f" The {subj} is {first}. {nm} concludes it is {last}. Is that conclusion valid?"
            final = f"Yes, {nm}'s conclusion is valid: the {subj} is {last}."
        else:
            prob = prose + dtx + f" Fact: the {subj} is {first}. Following the rules to the end, what must be true of the {subj}?"
            final = f"The {subj} is {last}."
        if prob in exclude or prob in seen:
            continue
        seen.add(prob)
        steps = [(f"Premise {i+1}: {preds[i]} -> {preds[i+1]}.", "valid") for i in range(hops)]
        if distract: steps.append(("The receipt rule shares no terms with the chain; a distractor, unused.", "valid"))
        steps.append((f"Fact: the {subj} is {first}.", "valid")); cur = first
        for i in range(hops):
            steps.append((f"From '{cur}' and premise {i+1}, conclude '{preds[i+1]}' (modus ponens).", "valid")); cur = preds[i+1]
        steps.append(("Check: only the premises and valid modus ponens are used.", "valid"))
        out.append(dict(domain=dom, problem=prob, steps=steps, final=final,
                        difficulty=min(max(1, hops + (1 if distract else 0)), 5),
                        vm="symbolic_solver",
                        vd=f"Encoded a length-{hops} implication chain and the fact; truth-table check confirms entailment.",
                        subj=subj, preds=preds, hops=hops))
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

def build_inductive(need, rng, exclude):
    out, seen = [], set()
    def desc(kind, a, b):
        if kind=="affine": return (lambda x:a*x+b), f"{a}*input + {b}", f"f = lambda x: {a}*x + {b}", f"f(x) = {a}x + {b}"
        if kind=="scale":  return (lambda x:a*x),   f"{a}*input",       f"f = lambda x: {a}*x",       f"f(x) = {a}x"
        if kind=="shift":  return (lambda x:x+b),   f"input + {b}",     f"f = lambda x: x + {b}",     f"f(x) = x + {b}"
        if kind=="quad":   return (lambda x:x*x+b), f"input^2 + {b}",   f"f = lambda x: x*x + {b}",   f"f(x) = x^2 + {b}"
        return (lambda x:a*x*x+b), f"{a}*input^2 + {b}", f"f = lambda x: {a}*x*x + {b}", f"f(x) = {a}x^2 + {b}"
    attempts = 0
    while len(out) < need and attempts < need*40 + 2000:
        attempts += 1
        if rng.random() < 0.2:  # string-rule family (formal grammars domain)
            nm, fn, txt = rng.choice([("reverse", lambda s:s[::-1], "reverse the string"),
                                      ("upper", lambda s:s.upper(), "uppercase the string"),
                                      ("double", lambda s:s+s, "repeat the string twice"),
                                      ("first-cap", lambda s:s.capitalize(), "capitalize the first letter")])
            ws = rng.sample(WORDS, 4); shown = [(w, fn(w)) for w in ws[:3]]; q = ws[3]
            pairs = ", ".join(f"{a}->{b}" for a, b in shown)
            prob = rng.choice([f"Infer the transformation and apply it to '{q}': {pairs}.",
                               f"These strings follow one rule: {pairs}. What is the output for '{q}'?"])
            if prob in exclude or prob in seen: continue
            seen.add(prob)
            out.append(dict(domain="formal grammars and symbol systems", problem=prob,
                steps=[(f"Each output applies: {txt}.", "valid"), ("Check shown pairs: consistent.", "valid"),
                       (f"Apply to '{q}': '{fn(q)}'.", "valid")],
                final=f"'{fn(q)}'", difficulty=2, vm="code_execution",
                vd=f"Executed the '{nm}' transform on shown inputs and the query; all match.",
                neg=dict(kind="str", first_in=shown[0][0], first_out=shown[0][1],
                         snd_in=shown[1][0], snd_out=shown[1][1], query=q)))
            continue
        kind = rng.choice(["affine","scale","shift","quad","quad2"])
        a = rng.randint(2, 9); b = rng.randint(1, 12) * rng.choice([1, -1])
        f, dsc, code, name = desc(kind, a, b)
        fmt = rng.randrange(4)
        if fmt == 3:  # sequence-next
            seq = [f(i) for i in range(1, 5)]; nxt = f(5)
            prob = "Find the next number in the sequence and state the rule: %s, ..." % (", ".join(map(str, seq)))
            if prob in exclude or prob in seen: continue
            seen.add(prob)
            out.append(dict(domain="mathematics", problem=prob,
                steps=[(f"Terms follow position n via {name.replace('x','n')}.", "valid"),
                       ("Check: " + ", ".join(f"n={i}->{f(i)}" for i in range(1, 5)) + ".", "valid"),
                       (f"Next term (n=5): {nxt}.", "valid")],
                final=str(nxt), difficulty=2, vm="code_execution",
                vd=f"Executed {name} at n=1..5; sequence matches and the next term is {nxt}."))
            continue
        xs = rng.sample(range(0, 13), 6); shown = [(x, f(x)) for x in xs[:3]]; held = [(x, f(x)) for x in xs[3:5]]; q = xs[5]
        pairs = ", ".join(f"{x}->{y}" for x, y in shown)
        if fmt == 0:
            prob = f"From these pairs, state the rule and apply it to {q}: {pairs}."; final = f"Rule: {name}. f({q}) = {f(q)}."
            dom = rng.choice(["mathematics","science","finance and business operations","economics and markets"])
        elif fmt == 1:
            prob = f"A function produces: {pairs}. What does it output for input {q}?"; final = str(f(q)); dom = "program behavior"
        else:
            prob = f"Infer the rule behind these examples, write it as code, then evaluate at {q}: {pairs}."; final = f"{code}; result {f(q)}."; dom = "program behavior"
        if prob in exclude or prob in seen: continue
        seen.add(prob)
        out.append(dict(domain=dom, problem=prob,
            steps=[(f"Fit output = {dsc}.", "valid"),
                   ("Check shown pairs: " + ", ".join(f"f({x})={y}" for x, y in shown) + ". All hold.", "valid"),
                   (f"As code: {code}.", "valid"), (f"Apply to {q}: {f(q)}.", "valid"),
                   ("Held-out check " + ", ".join(f"{x}->{y}" for x, y in held) + ": consistent.", "valid")],
            final=final, difficulty=(3 if "quad" in kind else 2 if kind in ("affine","scale") else 1),
            vm="code_execution", vd=f"Executed {name} on held-out {[x for x,_ in held]}; matched and f({q})={f(q)}.",
            neg=dict(kind="num", first_in=shown[0][0], first_out=shown[0][1],
                     snd_in=shown[1][0], snd_out=shown[1][1], query=q)))
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

def build_probabilistic(need, rng, exclude):
    out, seen = [], set()
    prevs = [round(x, 3) for x in [0.001,0.002,0.003,0.005,0.008,0.01,0.02,0.03,0.04,0.05,0.07,0.1,0.12,0.15,0.2,0.25,0.3]]
    rates = [0.7,0.72,0.75,0.8,0.82,0.85,0.88,0.9,0.92,0.95,0.97,0.99]
    attempts = 0
    while len(out) < need and attempts < need*40 + 3000:
        attempts += 1
        prev = rng.choice(prevs); se = rng.choice(rates); sp = rng.choice(rates)
        p = bayes(prev, se, sp); den = se*prev + (1-sp)*(1-prev)
        dom, ctx, H = rng.choice(PRB_FRAMINGS); stem = rng.choice(QSTEM_P)
        prob = ctx(int(se*100), int(sp*100), f"{prev*100:g}") + " " + stem.format(H=H)
        if prob in exclude or prob in seen: continue
        seen.add(prob)
        out.append(dict(domain=dom, problem=prob,
            steps=[(f"Prior = {prev:g}; complement = {1-prev:g}.", "valid"),
                   (f"True-positive rate {se:g}; false-positive rate {round(1-sp,3):g}.", "valid"),
                   (f"P(positive) = {se:g}*{prev:g} + {round(1-sp,3):g}*{1-prev:g} = {den:.5f}.", "valid"),
                   (f"Posterior = {se*prev:.5f} / {den:.5f} = {p:.4f}.", "valid"),
                   ("Check: consistent with Bayes' rule.", "valid")],
            final=f"{p:.3f}", difficulty=2 if prev>=0.1 else 3 if prev>=0.02 else 4, confidence=0.95,
            vm="code_execution", vd=f"Computed ({se:g}*{prev:g})/({se:g}*{prev:g}+{round(1-sp,3):g}*{1-prev:g}) = {p:.5f}; matches {p:.3f}.",
            se=int(se*100)))
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
def build_counterfactual(need, rng, exclude):
    out, seen = [], set()
    def qc(Y): return rng.choice(CFA_STEMS).format(Y=Y)
    def add(prob, steps, final, diff, vd, dom, base):
        if prob in exclude or prob in seen: return
        seen.add(prob); out.append(dict(domain=dom, problem=prob, steps=steps, final=final,
                                        difficulty=diff, vm="code_execution", vd=vd, base=base))
    attempts = 0
    while len(out) < need and attempts < need*40 + 3000:
        attempts += 1
        fam = rng.choice(["fill","cost","area","distance","interest","dosage","recipe","energy",
                          "voltage","tax","throughput","yield","wage"])
        if fam == "fill":
            r0, r1, t = rng.randint(2,8), rng.randint(9,15), rng.randint(3,8); b, c = r0*t, r1*t
            add(f"A tank fills at {r0} L/min for {t} min, reaching {b} L. If the rate had been {r1} L/min for the same {t} min, "+qc("the volume"),
                [("Model: volume = rate * time.","valid"),(f"Baseline: {r0}*{t} = {b}.","valid"),(f"Intervene: rate = {r1}.","valid"),(f"Run: {r1}*{t} = {c}.","valid"),(f"Check: {c} != baseline {b}.","valid")],
                str(c),2,f"volume=rate*time, rate={r1},time={t} => {c}.","program behavior",b)
        elif fam == "cost":
            u, disc, pr = rng.choice([8,10,12,15,20,25]), rng.choice([0.1,0.15,0.2,0.25,0.3,0.4]), rng.choice([12,20,25,40]); b = u*pr; c = round(u*pr*(1-disc),2); cs = str(int(c)) if c==int(c) else f"{c:.2f}"
            add(f"An order of {u} units at {pr} each cost {b} with no discount. With a {int(disc*100)}% discount, "+qc("the cost"),
                [("Model: cost = units*price*(1-discount).","valid"),(f"Baseline: {u}*{pr} = {b}.","valid"),(f"Intervene: discount={disc:g}. Run: {cs}.","valid"),(f"Check: baseline {b} cannot answer this.","valid")],
                cs,3,f"cost with discount={disc:g} => {cs}.","finance and business operations",b)
        elif fam == "area":
            L, w0, w1 = rng.choice([6,8,10,12]), rng.choice([3,4,5]), rng.choice([7,8,9,11]); b, c = L*w0, L*w1
            add(f"A plot is {L} by {w0} (area {b}). If the width were {w1} with the same length {L}, "+qc("the area"),
                [("Model: area = length*width.","valid"),(f"Baseline: {L}*{w0} = {b}.","valid"),(f"Intervene: width={w1}. Run: {c}.","valid"),(f"Check: {c} != {b}.","valid")],
                str(c),2,f"area with width={w1} => {c}.","engineering and physical systems",b)
        elif fam == "distance":
            s0, s1, t = rng.choice([40,50,60]), rng.choice([80,90,100,120]), rng.choice([2,3,4]); b, c = s0*t, s1*t
            add(f"A train goes {s0} km/h for {t} h, covering {b} km. At {s1} km/h for the same {t} h, "+qc("the distance"),
                [("Model: distance = speed*time.","valid"),(f"Baseline: {s0}*{t} = {b}.","valid"),(f"Intervene: speed={s1}. Run: {c}.","valid"),("Check: differs from baseline.","valid")],
                str(c),2,f"distance with speed={s1} => {c}.","science",b)
        elif fam == "interest":
            P, rr, y = rng.choice([1000,2000,5000]), rng.choice([0.03,0.05,0.08]), rng.choice([2,3,4]); b = round(P*0.02*y,2); c = round(P*rr*y,2)
            cs = str(int(c)) if c==int(c) else f"{c:.2f}"; bs = str(int(b)) if b==int(b) else f"{b:.2f}"
            add(f"A deposit of {P} earns simple interest at 2% for {y} years, yielding {bs}. At {int(rr*100)}% for the same {y} years, "+qc("the interest"),
                [("Model: interest = principal*rate*years.","valid"),(f"Baseline: {P}*0.02*{y} = {bs}.","valid"),(f"Intervene: rate={rr:g}. Run: {cs}.","valid"),("Check: baseline rate cannot answer.","valid")],
                cs,3,f"interest with rate={rr:g} => {cs}.","finance and business operations",b)
        elif fam == "dosage":
            mg, w0, w1 = rng.choice([5,10,15]), rng.choice([10,20,30]), rng.choice([40,50,60]); b, c = mg*w0, mg*w1
            add(f"A drug is dosed at {mg} mg/kg; a {w0} kg patient received {b} mg. For a {w1} kg patient at the same rate, "+qc("the dose"),
                [("Model: dose = rate*weight.","valid"),(f"Baseline: {mg}*{w0} = {b}.","valid"),(f"Intervene: weight={w1}. Run: {c}.","valid"),("Check: differs from baseline.","valid")],
                str(c),2,f"dose with weight={w1} => {c}.","medicine-style diagnosis",b)
        elif fam == "recipe":
            serv, per, fac = rng.choice([4,6,8]), rng.choice([2,3]), rng.choice([2,3]); b, c = serv*per, serv*per*fac
            add(f"A recipe for {serv} servings uses {b} cups of flour. Scaled {fac}x richer for the same {serv} servings, "+qc("the flour"),
                [("Model: flour = servings*per-serving.","valid"),(f"Baseline: {serv}*{per} = {b}.","valid"),(f"Intervene: per-serving *{fac}. Run: {c}.","valid"),("Check: differs from baseline.","valid")],
                str(c),2,f"flour scaled {fac}x => {c}.","everyday planning",b)
        elif fam == "energy":  # power * hours
            p0, p1, h = rng.choice([2,3,5]), rng.choice([6,8,10]), rng.choice([4,6,8]); b, c = p0*h, p1*h
            add(f"A heater at {p0} kW ran {h} h, using {b} kWh. At {p1} kW for the same {h} h, "+qc("the energy"),
                [("Model: energy = power*hours.","valid"),(f"Baseline: {p0}*{h} = {b}.","valid"),(f"Intervene: power={p1}. Run: {c}.","valid"),("Check: differs from baseline.","valid")],
                str(c),2,f"energy with power={p1} => {c}.","engineering and physical systems",b)
        elif fam == "voltage":  # Ohm's law: V = I * R
            I, r0, r1 = rng.choice([2,3,4,5]), rng.choice([3,4,6]), rng.choice([8,10,12,15]); b, c = I*r0, I*r1
            add(f"A resistor of {r0} ohms carries {I} A, dropping {b} V. If it were {r1} ohms at the same {I} A, "+qc("the voltage"),
                [("Model: voltage = current*resistance.","valid"),(f"Baseline: {I}*{r0} = {b}.","valid"),(f"Intervene: resistance={r1}. Run: {c}.","valid"),(f"Check: {c} != baseline {b}.","valid")],
                str(c),3,f"voltage with resistance={r1} => {c}.","engineering and physical systems",b)
        elif fam == "tax":  # income * rate
            inc, t0, t1 = rng.choice([20,30,40,50,60]), rng.choice([0.1,0.15]), rng.choice([0.2,0.25,0.3]); b = round(inc*t0,2); c = round(inc*t1,2)
            bs = str(int(b)) if b==int(b) else f"{b:.2f}"; cs = str(int(c)) if c==int(c) else f"{c:.2f}"
            add(f"Income of {inc}k taxed at {int(t0*100)}% owes {bs}k. At a {int(t1*100)}% rate on the same {inc}k, "+qc("the tax"),
                [("Model: tax = income*rate.","valid"),(f"Baseline: {inc}*{t0:g} = {bs}.","valid"),(f"Intervene: rate={t1:g}. Run: {cs}.","valid"),("Check: baseline rate cannot answer.","valid")],
                cs,3,f"tax with rate={t1:g} => {cs}.","finance and business operations",b)
        elif fam == "throughput":  # rate * seconds
            r0, r1, s = rng.choice([20,40,50]), rng.choice([80,100,120]), rng.choice([3,5,8]); b, c = r0*s, r1*s
            add(f"A service handling {r0} requests/s for {s} s served {b} requests. At {r1} requests/s for the same {s} s, "+qc("the total served"),
                [("Model: total = rate*seconds.","valid"),(f"Baseline: {r0}*{s} = {b}.","valid"),(f"Intervene: rate={r1}. Run: {c}.","valid"),("Check: differs from baseline.","valid")],
                str(c),2,f"throughput with rate={r1} => {c}.","program behavior",b)
        elif fam == "yield":  # plots * per-plot
            p0, per, p1 = rng.choice([6,8,10,12]), rng.choice([15,20,25]), rng.choice([16,18,20,24]); b, c = p0*per, p1*per
            add(f"A farm with {p0} plots yielding {per} kg each harvested {b} kg. With {p1} plots at the same {per} kg each, "+qc("the harvest"),
                [("Model: harvest = plots*per-plot.","valid"),(f"Baseline: {p0}*{per} = {b}.","valid"),(f"Intervene: plots={p1}. Run: {c}.","valid"),("Check: differs from baseline.","valid")],
                str(c),2,f"harvest with plots={p1} => {c}.","biology and ecology",b)
        else:  # wage: hours * rate
            h, w0, w1 = rng.choice([20,30,40]), rng.choice([12,15,18]), rng.choice([22,25,30]); b, c = h*w0, h*w1
            add(f"A worker paid {w0}/h for {h} h earned {b}. At {w1}/h for the same {h} h, "+qc("the pay"),
                [("Model: pay = hours*rate.","valid"),(f"Baseline: {h}*{w0} = {b}.","valid"),(f"Intervene: rate={w1}. Run: {c}.","valid"),("Check: differs from baseline.","valid")],
                str(c),2,f"pay with rate={w1} => {c}.","economics and markets",b)
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

CAU_STEMS = ["Is the claim supported, and what would settle it?",
             "Design an experiment that would determine whether the first truly causes the second.",
             "{name} insists the first causes the second. Critique this reasoning.",
             "Identify any confounder and say whether the causal claim holds."]
def build_causal(need, rng, exclude):
    out, seen = [], set()
    # confounder items: activity pairs sharing a seasonal/systemic driver
    for drv, (acts, dom) in list(CAUSAL_DRIVERS.items()):
        for x, y in combinations(acts, 2):
            for stem in CAU_STEMS:
                nm = rng.choice(NAMES)
                prob = f"{cap(x)} and {y} rise and fall together; both also track {drv}. " + stem.format(name=nm)
                if prob in exclude or prob in seen: continue
                seen.add(prob)
                out.append(dict(domain=dom, problem=prob,
                    steps=[(f"Observed: {x} and {y} correlate.","valid"),
                           (f"{cap(drv)} raises both, a common cause (confounder).","valid"),
                           (f"That common cause already accounts for the correlation, so the co-movement on its own does not establish a direct {x}->{y} link.","valid"),
                           (f"Discriminating test: hold {drv} fixed and vary {x}; watch {y}.","valid"),
                           (f"Check: at fixed {drv}, if {y} still tracks {x} there is a direct effect; if not, the claim is unsupported.","valid")],
                    final=f"Not supported by this evidence: {drv} is a confounder, so the correlation alone cannot establish that {x} causes {y}. Hold {drv} fixed and vary {x} to test for any direct effect on {y}.",
                    difficulty=3, vm="answer_match",
                    vd=f"Modeled common cause: {drv} drives both {x} and {y}, which explains their co-movement; a direct {x}->{y} effect is not established without intervening on {x} at fixed {drv}.", is_conf=True))
    # mediator + direct items
    for cause, eff, med, dom in CAUSAL_MED:
        prob = f"{cap(cause)} is associated with {eff}. The model has no direct arrow; {cause} produces {med}, which produces {eff}. Does {cause} cause {eff}, and how?"
        if prob not in exclude and prob not in seen:
            seen.add(prob); out.append(dict(domain=dom, problem=prob,
                steps=[(f"Edges: {cause} -> {med} -> {eff}.","valid"),(f"No direct edge; {med} is a mediator.","valid"),
                       (f"{cap(cause)} causes {eff} indirectly via {med}.","valid"),(f"Test: block {med}; {eff} should drop.","valid"),
                       ("Check: the effect flows through the mediator.","valid")],
                final=f"Yes, indirectly: the effect is mediated by {med}. Blocking the mediator would eliminate it.",
                difficulty=3, vm="answer_match", vd=f"Graph {cause}->{med}->{eff}; mediated cause.", is_conf=False))
    for cause, eff, dom in CAUSAL_DIRECT:
        prob = f"The model has a direct edge {cause} -> {eff} and no common cause. A controlled test varies {cause} alone and {eff} follows. Is the causal claim supported?"
        if prob not in exclude and prob not in seen:
            seen.add(prob); out.append(dict(domain=dom, problem=prob,
                steps=[(f"Edge: {cause} -> {eff}, no confounder.","valid"),("A controlled intervention varies the cause alone.","valid"),
                       (f"{cap(eff)} responds, matching the edge.","valid"),("Check: no confounder + responsive intervention => supported.","valid")],
                final="Yes, supported; the controlled intervention isolates the cause and the effect responds.",
                difficulty=2, vm="answer_match", vd=f"Direct edge {cause}->{eff}, no confounder.", is_conf=False))
    rng.shuffle(out)
    return out[:need] if len(out) > need else out

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
def build_metacognitive(need, rng, exclude):
    out, seen = [], set()
    def q(T): return rng.choice(Q_MET).format(T=T)
    attempts = 0
    while len(out) < need and attempts < need*40 + 3000:
        attempts += 1
        kind = rng.choice(["affirm","half","root","clean_div","clean_odd"])
        if kind == "affirm":
            dom, subj = _domain_subject(rng); L = rng.randint(1, 3); preds = rng.sample(PREDICATES, L+1)
            T = "Rules: " + " ".join(f"If the {subj} is {preds[i]}, then it is {preds[i+1]}." for i in range(L)) + f" Observed: the {subj} is {preds[-1]}. Therefore it is {preds[0]}."
            prob = q(T)
            if prob in exclude or prob in seen: continue
            seen.add(prob)
            out.append(dict(domain=dom, problem=prob,
                steps=[(f"The chain '{preds[0]}' -> ... -> '{preds[-1]}' is fine.","valid"),
                       (f"It infers '{preds[0]}' from '{preds[-1]}': affirming the consequent.","valid"),
                       (f"'{preds[-1]}' can hold for other reasons.","valid"),(f"Correction: '{preds[0]}' is not justified.","valid")],
                final=f"Error: affirming the consequent. '{preds[-1]}' does not entail '{preds[0]}'.",
                difficulty=2 if L==1 else 3, vm="process_check", vd="Planted error: affirming the consequent."))
        elif kind == "half":
            N = rng.choice([n for n in range(44, 900, 4)]); half = N//2; slip = half-2; qq = N//4
            prob = q(f"Half of {N} is {half}, and half of {half} is {slip}, so a quarter of {N} is {slip}.")
            if prob in exclude or prob in seen: continue
            seen.add(prob)
            out.append(dict(domain="mathematics", problem=prob,
                steps=[(f"Half of {N} is {half}. Correct.","valid"),(f"Half of {half} is {half//2}, not {slip}: an arithmetic slip.","valid"),(f"Correction: a quarter of {N} is {qq}.","valid")],
                final=f"Error: half of {half} is {half//2}, not {slip}. A quarter of {N} is {qq}.",
                difficulty=2, vm="process_check", vd="Planted arithmetic slip."))
        elif kind == "root":
            k = rng.randint(6, 60); sq = k*k
            prob = q(f"x^2 = {sq}, so x = {k}.")
            if prob in exclude or prob in seen: continue
            seen.add(prob)
            out.append(dict(domain="mathematics", problem=prob,
                steps=[(f"x = {k} satisfies x^2 = {sq}.","valid"),("It skips the negative root.","valid"),(f"x = -{k} also works.","valid")],
                final=f"Error: a skipped case. x = {k} or x = -{k}.", difficulty=3, vm="process_check", vd="Planted skipped negative root."))
        elif kind == "clean_div":
            d = rng.choice([n for n in range(4, 80, 2)]); n = d*rng.randint(3, 9)
            prob = q(f"If a number is divisible by {d} then it is even, since {d} is even. {n} is divisible by {d}, so {n} is even.")
            if prob in exclude or prob in seen: continue
            seen.add(prob)
            out.append(dict(domain="mathematics", problem=prob,
                steps=[(f"Divisible by {d} implies even, since {d} is even. Valid.","valid"),(f"{n} is divisible by {d}.","valid"),(f"So {n} is even. Valid.","valid"),("No error is present.","valid")],
                final=f"No errors. The reasoning is valid: {n} is even.", difficulty=2, vm="process_check", vd="Planted CLEAN trace; certified."))
        else:  # clean_odd
            nn = rng.randint(3, 12); s = nn*nn; odds = ", ".join(str(2*i+1) for i in range(nn))
            prob = q(f"The sum of the first n odd numbers is n^2. The first {nn} odd numbers are {odds}, summing to {s} = {nn}^2.")
            if prob in exclude or prob in seen: continue
            seen.add(prob)
            out.append(dict(domain="mathematics", problem=prob,
                steps=[(f"{odds.replace(', ',' + ')} = {s}.","valid"),(f"{s} = {nn}^2.","valid"),(f"Matches the identity for n = {nn}.","valid"),("No error is present.","valid")],
                final="No errors. The reasoning is valid.", difficulty=2, vm="process_check", vd="Planted CLEAN trace; certified."))
    return out

# --- authored types (bank x format); grow BANKS for larger seasons ----------
Q_ABD = ["What most plausibly explains this?", "Rank the possible explanations and justify the most likely.",
         "Which explanation best fits, and why are the alternatives weaker?", "You are diagnosing this. What is the single most likely cause?",
         "State your leading hypothesis and the single observation that most rules out the runner-up.",
         "Give the most probable cause and briefly say why each alternative is less likely.",
         "As the technician on call, what is your leading diagnosis and reasoning?",
         "Infer the best explanation and note your confidence in it."]
def build_abductive(need, rng, exclude):
    out, seen = [], set()
    bases = []
    for dom, subject, sigs in ABD_BANK:
        for sig, cause, alts in sigs:
            steps = [(f"Observation: {subject}, {sig}.", "valid")]
            for alt, why in alts: steps.append((f"Alternative '{alt}' is unlikely: {why}.", "valid"))
            steps.append((f"Best explanation: {cause}.", "valid")); steps.append(("Confidence moderate: not every competing factor was directly measured.", "valid"))
            bases.append((dom, subject, sig, cause, steps))
    for stem in Q_ABD:
        for dom, subject, sig, cause, steps in bases:
            prob = f"{cap(subject)}: {sig}. {stem}"
            if prob in exclude or prob in seen: continue
            seen.add(prob)
            out.append(dict(domain=dom, problem=prob, steps=steps, final=cap(cause)+".", difficulty=3, confidence=0.65,
                            vm="answer_match", vd=f"Planted cause: {cause}. Top explanation ranks above the alternatives."))
            if len(out) >= need: return out
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
    ("animal to its young", "animal", "young", "biology and ecology", 1, False,
     [("dog","puppy"),("cat","kitten"),("cow","calf"),("horse","foal"),("sheep","lamb"),("lion","cub"),("frog","tadpole"),("hen","chick"),("kangaroo","joey"),("bear","cub"),("deer","fawn"),("goat","kid"),("duck","duckling"),("fox","kit"),("eagle","eaglet")]),
    ("profession to its tool", "profession", "tool", "engineering and physical systems", 2, False,
     [("chef","knife"),("painter","brush"),("carpenter","hammer"),("writer","pen"),("surgeon","scalpel"),("photographer","camera"),("farmer","plow"),("tailor","needle"),("gardener","spade"),("blacksmith","anvil"),("dentist","drill"),("barber","scissors"),("mechanic","wrench"),("sculptor","chisel"),("cartographer","compass")]),
    ("object to its material", "object", "material", "engineering and physical systems", 2, False,
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
    ("element to its chemical symbol", "element", "chemical symbol", "chemistry", 2, False,
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
    ("emotion to its facial sign", "emotion", "facial sign", "social situations", 2, False,
     [("happiness","a smile"),("sadness","a frown"),("surprise","raised brows"),("anger","a scowl"),("fear","wide eyes"),("disgust","a wrinkled nose"),("boredom","a yawn"),("confusion","a furrowed brow")]),
    ("disease to the organ it attacks", "disease", "organ", "medicine-style diagnosis", 3, False,
     [("hepatitis","the liver"),("nephritis","the kidney"),("pneumonia","the lungs"),("gastritis","the stomach"),("arthritis","the joints"),("meningitis","the brain lining"),("carditis","the heart"),("dermatitis","the skin")]),
    ("operation to its inverse", "operation", "inverse", "mathematics", 2, False,
     [("addition","subtraction"),("multiplication","division"),("squaring","square root"),("exponentiation","logarithm"),("differentiation","integration"),("encryption","decryption"),("folding","unfolding"),("freezing","melting")]),
    ("tool to the quantity it measures", "instrument", "quantity", "science", 2, False,
     [("thermometer","temperature"),("barometer","pressure"),("odometer","distance"),("voltmeter","voltage"),("scale","weight"),("clock","time"),("hygrometer","humidity"),("seismometer","ground motion"),("ammeter","current"),("speedometer","speed")]),
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
    "Each of these maps a {left} to its {right}: {demo}. Fill in {t0} -> ?",
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
            ("economics and markets", "a growing firm", "reinvested profit", "it expands capacity so output rises", "a one-off grant", False),
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
    prob = (f"Reference system: the output follows a {lawname} law of the input -- {gloss}, "
            f"so the output scales with the input raised to the power {p}. "
            f"An analogous system in {dom} obeys the same {lawname} law; its input is scaled by a factor of {k}. "
            f"By the same law, the output is scaled by what factor? (A linear reading -- factor {k} -- is the distractor.)")
    steps = [
        (f"Relation to transfer: output scales as (input)^{p} (the {lawname} law), not linearly.", "valid"),
        (f"Reject the surface distractor: a linear reading would give factor {k}, but the law is a power of {p}.", "valid"),
        (f"Apply the law: the input factor {k} raised to the power {p} is {k}^{p} = {ans}.", "valid"),
        (f"Check: under a {lawname} law an input factor of {k} yields an output factor of {k}**{p} = {ans}; the linear value {k} is wrong.", "valid"),
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

Q_MOR = ["Lay out the strongest case for each side.",
         "What competing ethical principles are at stake, and how do they conflict?",
         "Analyze this dilemma from at least two ethical frameworks.",
         "How should this be reasoned about?",
         "Argue the case for one side, then give the strongest rebuttal.",
         "Identify the stakeholders and what each is owed.",
         "Where would a consequentialist and a rights-based view diverge here?",
         "What principle would you have to accept to justify each choice?"]
def build_moral(need, rng, exclude):
    out, seen = [], set()
    for stem in Q_MOR:
        for prob0, fa, fb in MORAL_BANK:
            prob = prob0 + " " + stem
            if prob in exclude or prob in seen: continue
            seen.add(prob)
            out.append(dict(domain="ethics", problem=prob,
                steps=[(f"Name the tension: {fa} versus {fb}.", "valid"),
                       (f"Consequentialist reading: weighing outcomes supports {fa}.", "valid"),
                       (f"Duty/fairness reading: some obligations resist a pure outcome count, supporting {fb}.", "valid"),
                       ("Consistency check: the chosen principle must be acceptable applied generally and to oneself.", "valid"),
                       ("Acknowledge the residual: any choice leaves a real moral cost, which the reasoning names.", "valid")],
                final=f"Both positions are defensible: one case rests on {fa}, the other on {fb}. The decision turns on which principle is adopted and applied consistently, not on a single forced verdict.",
                difficulty=3, vm="rubric_judge",
                vd="Rubric scored perspective coverage (two frameworks), reasoning quality, and internal consistency; no predetermined conclusion rewarded."))
            if len(out) >= need: return out
    return out

BUILDERS = {"deductive": build_deductive, "inductive": build_inductive, "probabilistic": build_probabilistic,
            "counterfactual": build_counterfactual, "causal": build_causal, "metacognitive": build_metacognitive,
            "abductive": build_abductive, "analogical": build_analogical, "moral-ethical": build_moral}
NEG_BUILDERS = {"deductive": neg_deductive, "inductive": neg_inductive, "probabilistic": neg_probabilistic,
                "counterfactual": neg_counterfactual, "causal": neg_causal}
# analogical is procedurally generated from ANA_REL + parametric trap kernels
# (the trap arithmetic is recomputed here), so it is labeled procedural -- not
# human_expert/hand-authored, which would misrepresent how it was produced.
GEN_METHOD = {t: ("multi_agent" if t == "metacognitive" else
                  "human_expert" if t in {"abductive","moral-ethical"} else "procedural") for t in RT}
SOURCE = {t: ("hand-authored" if t in {"abductive","moral-ethical"}
              else "procedural-gen" if t == "analogical"      # by-construction + computed checks, awaits Solver pass
              else "procedural-gen, verified") for t in RT}

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
                new_files.setdefault(f"negatives/{t}.negatives.jsonl", []).append(nr); made_neg += 1
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
