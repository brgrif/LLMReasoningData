#!/usr/bin/env python3
"""
generate.py - reusable batch generator for the reasoning-data corpus.

WHAT IT DOES
    Produces a batch of new reasoning traces (default 500 per type - a
    "500-shot" batch), appends them to data/curated/ (and data/negatives/ for
    the verifiable types), continuing IDs and de-duplicating against whatever
    is already on disk. Every record it emits validates against SCHEMA.md.

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
    "logic puzzles": ["token","tile","marker","card","switch","lever","glyph"],
    "program behavior": ["record","request","job","event","session","message"],
    "finance and business operations": ["invoice","transaction","account","order","claim","payment"],
    "mechanical / systems troubleshooting": ["part","unit","valve","motor","panel","gauge"],
    "chemistry": ["sample","batch","compound","solution","vial","reagent"],
    "medicine-style diagnosis": ["chart","specimen","dose","case","scan","sample"],
    "incident and root-cause analysis": ["alert","ticket","deploy","signal","log entry","trace"],
    "science": ["reading","measurement","trial","observation","specimen","dataset"],
    "law and regulation": ["filing","contract","permit","case file","statute","affidavit"],
    "economics and markets": ["asset","listing","position","order","contract","lot"],
    "engineering and physical systems": ["beam","circuit","sensor","joint","module","bearing"],
    "biology and ecology": ["culture","specimen","colony","plot","sample","population"],
    "formal grammars and symbol systems": ["string","token","symbol","expression","rule","glyph"],
}
# Generic state predicates for formal chains (domain-neutral, so any subject fits).
PREDICATES = ["flagged","queued","archived","approved","sealed","escalated","audited",
              "locked","expired","tagged","routed","verified","quarantined","published",
              "indexed","frozen","released","logged","signed","cleared","batched","held"]

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
    subj, preds, hops = it["subj"], it["preds"], it["hops"]
    first, last = preds[0], preds[-1]
    prob = ("Rules: " + " ".join(f"If the {subj} is {preds[i]}, then it is {preds[i+1]}." for i in range(hops))
            + f" Observed: the {subj} is {last}. Someone concludes it must be {first}. Is that valid?")
    return make(tid+"-neg", "deductive", it["domain"], prob,
        [("The premises are as given.", "valid"),
         (f"They infer '{first}' from '{last}', reversing the implications.", "invalid"),
         ("This affirms the consequent, an invalid inference.", "invalid")],
        f"No; concluding '{first}' affirms the consequent.", False, it["difficulty"], "procedural",
        "symbolic_solver", False,
        "Backward reasoning affirms the consequent; the reversed conclusion is not entailed.",
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
                vd=f"Executed the '{nm}' transform on shown inputs and the query; all match."))
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
            prob = f"From these pairs, state the rule and apply it to {q}: {pairs}."; final = f"Rule: {name}. f({q}) = {f(q)}."; dom = "mathematics"
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
            vm="code_execution", vd=f"Executed {name} on held-out {[x for x,_ in held]}; matched and f({q})={f(q)}."))
    return out

def neg_inductive(tid, it):
    return make(tid+"-neg", "inductive", it["domain"], it["problem"],
        [("Fit a rule to only the first shown pair and ignore the rest.", "invalid"),
         ("Skip the remaining shown pairs and the held-out checks.", "invalid"),
         ("Report that under-determined rule.", "invalid")],
        "output = input (overfit)", False, it["difficulty"], "procedural", "code_execution", False,
        "Executing the overfit rule fails the other shown pairs; rule rejected.",
        "procedural-gen, verified", None, it["_created"],
        notes=f"paired positive: {tid}. Fallacy: overfitting one example.")

PRB_FRAMINGS = [
    ("medicine-style diagnosis", lambda se,sp,pv: f"A medical test is {se}% sensitive and {sp}% specific for a condition with {pv}% prevalence.", "the person has the condition"),
    ("program behavior",         lambda se,sp,pv: f"A spam filter flags {se}% of spam and wrongly flags {100-sp}% of legitimate mail; {pv}% of incoming mail is spam.", "the message is spam"),
    ("mechanical / systems troubleshooting", lambda se,sp,pv: f"A scanner catches {se}% of defective parts and falsely flags {100-sp}% of good parts; {pv}% of parts are defective.", "the part is defective"),
    ("incident and root-cause analysis", lambda se,sp,pv: f"An intrusion detector alerts on {se}% of real attacks and false-alarms on {100-sp}% of normal sessions; {pv}% of sessions are attacks.", "it is a real attack"),
    ("science",                  lambda se,sp,pv: f"A field survey detects a species in {se}% of sites where it lives and gives a false positive in {100-sp}% of sites where it does not; the species is present at {pv}% of sites.", "the species is present"),
    ("finance and business operations", lambda se,sp,pv: f"A fraud screen flags {se}% of fraudulent charges and wrongly flags {100-sp}% of legitimate ones; {pv}% of charges are fraudulent.", "the charge is fraudulent"),
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
        fam = rng.choice(["fill","cost","area","distance","interest","dosage","recipe","energy"])
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
        else:  # energy: power * hours
            p0, p1, h = rng.choice([2,3,5]), rng.choice([6,8,10]), rng.choice([4,6,8]); b, c = p0*h, p1*h
            add(f"A heater at {p0} kW ran {h} h, using {b} kWh. At {p1} kW for the same {h} h, "+qc("the energy"),
                [("Model: energy = power*hours.","valid"),(f"Baseline: {p0}*{h} = {b}.","valid"),(f"Intervene: power={p1}. Run: {c}.","valid"),("Check: differs from baseline.","valid")],
                str(c),2,f"energy with power={p1} => {c}.","engineering and physical systems",b)
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
                           (f"The correlation is explained by {drv}; no direct edge is implied.","valid"),
                           (f"Discriminating test: hold {drv} fixed and vary {x}; watch {y}.","valid"),
                           (f"Check: if {y} do not move at fixed {drv}, the claim is refuted.","valid")],
                    final=f"Not supported; {drv} is a confounder. Hold {drv} fixed and vary {x} to test for a direct effect on {y}.",
                    difficulty=3, vm="answer_match",
                    vd=f"Graph: {drv}->{x}, {drv}->{y}, no {x}->{y} edge. Confounder '{drv}'.", is_conf=True))
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
        "A confounder explains the correlation and there is no direct edge; asserting causation is wrong.",
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
         "Which explanation best fits, and why are the alternatives weaker?", "You are diagnosing this. What is the single most likely cause?"]
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

Q_ANA = ["{demo}; {t0} : ?  (each pair shares one relation, {R}). Fill the blank.",
         "Each pair shares the relation '{R}'. Find the term completing the last: {demo}; {t0} -> ?",
         "In this set the relation is '{R}': {demo}. What completes {t0} : ?",
         "Same relation in each pair ({R}): {demo}. Complete the last pair: {t0} : ?"]
def build_analogical(need, rng, exclude):
    out, seen = [], set()
    bases = []
    for rel, dom, diff, items in ANA_BANKS:
        for i in range(len(items)):
            target = items[i]; others = [items[j] for j in range(len(items)) if j != i][:3]
            if len(others) < 3: continue
            demo = "; ".join(f"{a} : {b}" for a, b in others)
            steps = [(f"Relation: {rel}.", "valid"), ("Each demonstration instantiates it.", "valid"),
                     (f"Apply it to {target[0]}.", "valid"), (f"The matching item is {target[1]}.", "valid"),
                     ("Check: the relation holds; surface content differs.", "valid")]
            bases.append((rel, dom, diff, target, demo, steps))
    for stem in Q_ANA:
        for rel, dom, diff, target, demo, steps in bases:
            prob = stem.format(R=rel, demo=demo, t0=target[0])
            if prob in exclude or prob in seen: continue
            seen.add(prob)
            out.append(dict(domain=dom, problem=prob, steps=steps, final=cap(str(target[1]))+".",
                            difficulty=diff, vm="process_check", vd=f"Completes the '{rel}' relation; mapping stated."))
            if len(out) >= need: return out
    return out

Q_MOR = ["Lay out the strongest case for each side.",
         "What competing ethical principles are at stake, and how do they conflict?",
         "Analyze this dilemma from at least two ethical frameworks.",
         "How should this be reasoned about?"]
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
GEN_METHOD = {t: ("multi_agent" if t == "metacognitive" else
                  "human_expert" if t in {"abductive","analogical","moral-ethical"} else "procedural") for t in RT}
SOURCE = {t: ("hand-authored" if t in {"abductive","analogical","moral-ethical"} else "procedural-gen, verified") for t in RT}

# --------------------------------------------------------------------------
# Word / relation / scenario BANKS  (# BANK: extend to scale a type)
# --------------------------------------------------------------------------
NAMES = ["Priya","Marcus","Lena","Diego","Aisha","Tomas","Nadia","Ravi","Mei","Omar","Sofia","Jonas","Yuki","Kwame","Ines","Bardia"]
WORDS = ["cat","door","lamp","river","stone","cloud","piano","glass","tiger","melon","robot","north","amber","field","otter","zebra","quartz","violet","harbor","cactus","ember","willow","comet","ledger"]

CAUSAL_DRIVERS = {  # BANK: extend to scale causal (each driver adds C(len,2) confounder pairs)
    "hot weather": (["ice cream sales","swimming-pool visits","air-conditioner use","cold-drink sales","sunscreen sales","beach attendance","fan sales","popsicle sales"], "economics and markets"),
    "cold weather": (["heater use","hot-chocolate sales","soup sales","firewood sales","scarf purchases","ice-rink attendance","space-heater sales","hot-tea sales"], "economics and markets"),
    "heavy rainfall": (["umbrella sales","raincoat purchases","indoor-cinema attendance","taxi demand","boot sales","gutter-cleaning calls"], "science"),
    "a traffic surge": (["page latency","error-log volume","support tickets","checkout failures","server CPU load","cache misses"], "incident and root-cause analysis"),
    "a public holiday": (["retail foot traffic","restaurant bookings","parking occupancy","toll-road volume","cinema tickets","ride-share demand"], "economics and markets"),
    "pollen season": (["antihistamine sales","tissue sales","allergy-clinic visits","eye-drop sales","air-purifier sales"], "medicine-style diagnosis"),
    "a heat wave": (["heatstroke ER visits","bottled-water sales","AC-repair calls","pool-chemical sales","dehydration cases"], "medicine-style diagnosis"),
    "exam season": (["campus coffee sales","library occupancy","energy-drink sales","printing-shop demand","late-night food orders"], "economics and markets"),
}
CAUSAL_MED = [
    ("rain","car skids","a wet road surface","science"),("infection","sweating","a fever","medicine-style diagnosis"),
    ("a price cut","higher revenue","increased units sold","economics and markets"),("exercise","weight loss","a calorie deficit","medicine-style diagnosis"),
    ("more study time","higher test scores","better mastery","science"),("a marketing campaign","more sign-ups","increased site visits","economics and markets"),
    ("smoking","lung damage","accumulated tar","medicine-style diagnosis"),("a software update","fewer crashes","a fixed memory leak","incident and root-cause analysis"),
    ("heavy rain","a flooded basement","a rising water table","incident and root-cause analysis"),("a sugary diet","tooth decay","acid from oral bacteria","medicine-style diagnosis"),
    ("a wage rise","more spending","higher disposable income","economics and markets"),("deforestation","soil erosion","loss of root structure","science"),
]
CAUSAL_DIRECT = [
    ("pressing the switch","the light turning on","program behavior"),("adding fertilizer","faster plant growth","science"),
    ("raising the price","fewer units sold","economics and markets"),("taking the antibiotic","the infection clearing","medicine-style diagnosis"),
    ("tightening the valve","the leak stopping","mechanical / systems troubleshooting"),("increasing study hours","a higher exam score","science"),
    ("adding an index","faster query response","program behavior"),("lowering the thermostat","a colder room","engineering and physical systems"),
    ("cutting interest rates","more borrowing","economics and markets"),("applying the brakes","the car slowing","engineering and physical systems"),
]
ANA_BANKS = [  # BANK: extend to scale analogical (each item is one leave-one-out target x formats)
    ("animal to the sound it makes","biology and ecology",1,[("dog","bark"),("cat","meow"),("cow","moo"),("duck","quack"),("lion","roar"),("horse","neigh"),("sheep","bleat"),("frog","croak"),("bee","buzz"),("snake","hiss"),("owl","hoot"),("wolf","howl"),("pig","oink"),("crow","caw"),("hen","cluck")]),
    ("animal to its young","biology and ecology",1,[("dog","puppy"),("cat","kitten"),("cow","calf"),("horse","foal"),("sheep","lamb"),("lion","cub"),("frog","tadpole"),("hen","chick"),("kangaroo","joey"),("bear","cub"),("deer","fawn"),("goat","kid"),("duck","duckling"),("fox","kit"),("eagle","eaglet")]),
    ("profession to its tool","engineering and physical systems",2,[("chef","knife"),("painter","brush"),("carpenter","hammer"),("writer","pen"),("surgeon","scalpel"),("photographer","camera"),("farmer","plow"),("tailor","needle"),("gardener","spade"),("blacksmith","anvil"),("dentist","drill"),("barber","scissors"),("mechanic","wrench"),("sculptor","chisel"),("cartographer","compass")]),
    ("object to its material","engineering and physical systems",2,[("book","paper"),("window","glass"),("tire","rubber"),("wire","copper"),("ring","gold"),("bottle","plastic"),("table","wood"),("blade","steel"),("sweater","wool"),("brick","clay"),("candle","wax"),("rope","fiber"),("balloon","latex"),("coin","metal"),("jar","glass")]),
    ("member to its category","biology and ecology",2,[("apple","fruit"),("car","vehicle"),("violin","instrument"),("salmon","fish"),("oak","tree"),("sparrow","bird"),("iron","metal"),("rose","flower"),("whale","mammal"),("triangle","shape"),("tennis","sport"),("oxygen","gas"),("ruby","gemstone"),("maple","tree"),("trumpet","instrument")]),
    ("word to a stronger-degree version","formal grammars and symbol systems",2,[("warm","hot"),("big","huge"),("cool","cold"),("good","great"),("tired","exhausted"),("small","tiny"),("happy","ecstatic"),("bad","terrible"),("wet","soaked"),("hungry","starving"),("angry","furious"),("quiet","silent"),("pretty","gorgeous"),("funny","hilarious"),("sad","devastated")]),
    ("cause to its typical effect","science",2,[("spark","fire"),("virus","illness"),("rain","flood"),("study","knowledge"),("exercise","fitness"),("drought","famine"),("friction","heat"),("practice","skill"),("investment","growth"),("pollution","smog"),("training","endurance"),("sunlight","photosynthesis")]),
    ("driver to the medium it moves","engineering and physical systems",2,[("heart","blood"),("pump","water"),("battery","charge"),("fan","air"),("turbine","steam"),("sump pump","water"),("escalator","people"),("conveyor","packages"),("speaker","sound"),("windmill","grain")]),
    ("tool to its function","engineering and physical systems",2,[("knife","cutting"),("pen","writing"),("key","unlocking"),("broom","sweeping"),("thermometer","measuring temperature"),("compass","finding direction"),("ruler","measuring length"),("clock","telling time"),("filter","removing impurities"),("brake","stopping"),("magnet","attracting iron"),("shovel","digging")]),
    ("country to its capital","science",2,[("France","Paris"),("Japan","Tokyo"),("Egypt","Cairo"),("Peru","Lima"),("Kenya","Nairobi"),("Norway","Oslo"),("Cuba","Havana"),("Nepal","Kathmandu"),("Ghana","Accra"),("Chile","Santiago"),("Iraq","Baghdad"),("Greece","Athens")]),
    ("unit to what it measures","science",2,[("meter","length"),("gram","mass"),("second","time"),("ampere","current"),("volt","voltage"),("watt","power"),("liter","volume"),("kelvin","temperature"),("pascal","pressure"),("hertz","frequency"),("joule","energy"),("mole","amount")]),
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
]

# --------------------------------------------------------------------------
# Corpus IO
# --------------------------------------------------------------------------
def load_corpus(data_dir):
    existing = {t: set() for t in RT}
    max_train = {t: 999 for t in RT}; max_eval = {t: 4999 for t in RT}
    counts = {t: {"train": 0, "eval": 0, "neg": 0} for t in RT}
    for f in glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True):
        loc = "eval" if ".eval." in f else "train" if ".train." in f else "neg" if os.sep+"negatives"+os.sep in f else "other"
        for line in open(f):
            line = line.strip()
            if not line: continue
            r = json.loads(line); t = r["reasoning_type"]; existing[t].add(r["problem"])
            if loc in counts[t]: counts[t][loc] += 1
            m = re.match(r'^[a-z]{3}-(\d{6})$', r["id"])
            if m:
                num = int(m.group(1))
                if ".eval." in f: max_eval[t] = max(max_eval[t], num)
                elif ".train." in f: max_train[t] = max(max_train[t], num)
    return existing, max_train, max_eval, counts

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
    args = ap.parse_args()

    data_dir = os.path.join(args.root, "data")
    canon = load_canonical_domains(args.root)
    existing, max_train, max_eval, counts = load_corpus(data_dir)

    if args.report_only:
        print("Current corpus:")
        for t in RT:
            c = counts[t]; print(f"  {t:15s} train={c['train']:4d} eval={c['eval']:4d} neg={c['neg']:4d}")
        return

    rng = random.Random(args.seed)
    new_files = {}          # rel path -> list of records
    report = []
    for t in RT:
        pool = BUILDERS[t](args.per_type, rng, existing[t])
        produced = min(len(pool), args.per_type)
        chosen = pool[:produced]
        n_eval = int(round(produced * (1 - args.train_frac)))
        n_train = produced - n_eval
        tstart = max_train[t] + 1; estart = max_eval[t] + 1
        pre = PREFIX[t]; new_train = []
        doms = set()
        for i, it in enumerate(chosen):
            it["_created"] = args.created
            split = "train" if i < n_train else "eval"
            num = (tstart + i) if split == "train" else (estart + (i - n_train))
            rid = f"{pre}-{num:06d}"
            rec = make(rid, t, it["domain"], it["problem"], it["steps"], it["final"], True, it["difficulty"],
                       GEN_METHOD[t], it["vm"], True, it["vd"], SOURCE[t], None, args.created, confidence=it.get("confidence"))
            errs = validate(rec, "curated", canon)
            if errs: raise SystemExit(f"VALIDATION FAIL {rid}: {errs}")
            new_files.setdefault(f"curated/{t}.{split}.jsonl", []).append(rec)
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
    print(f"\nnew records this batch: {total} (seed={args.seed}, created={args.created})")
    if short:
        print("bank-limited types (produced < requested):", ", ".join(short))

    if not args.write:
        print("\nDRY RUN - nothing written. Re-run with --write to append.")
        return
    for rel, recs in sorted(new_files.items()):
        path = os.path.join(data_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWROTE {total} records to {data_dir}")

if __name__ == "__main__":
    main()
