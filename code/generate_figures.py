#!/usr/bin/env python3
"""
Generate all figures and statistics for:
"A knowledge-grounded language-model agent uncovers psychological mechanisms
of road crashes from police interrogation records"

Reads the real experimental outputs (knowledge graph, hypothesis scores,
accident-type triage) from ../data and writes publication figures (PDF+PNG)
to ../figures plus a stats.json used by the manuscript.

Usage:  python generate_figures.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIGS = HERE.parent / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ style ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})
C_KG   = "#0072B2"   # LLM+KG group (blue)
C_LLM  = "#E69F00"   # LLM baseline (orange)
C_ACC  = "#009E73"
C_PINK = "#CC79A7"
C_GREY = "#4D4D4D"

NODE_COLS = {                     # knowledge-graph node palette
    "Trait":              "#B3CDE3",
    "TraitCombination":   "#FBB4AE",
    "BehavioralTendency": "#CCEBC5",
    "Behavior":           "#FED9A6",
    "CausalFactor":       "#DECBE4",
    "Outcome":            "#F2F2F2",
}
NODE_EDGE = {
    "Trait": "#3182BD", "TraitCombination": "#DE2D26",
    "BehavioralTendency": "#31A354", "Behavior": "#E6550D",
    "CausalFactor": "#756BB1", "Outcome": "#636363",
}
DC_W, SC_W = 7.2, 3.5             # double/single column width (inch)

def box(ax, xy, w, h, fc, ec, lw=0.8, r=0.035, ls="-", zorder=2):
    p = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad=0.004,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=zorder,
                       mutation_aspect=1.0)
    ax.add_patch(p)
    return p

def arrow(ax, p0, p1, color=C_GREY, lw=1.0, style="-|>", shrinkA=1, shrinkB=1,
          rad=0.0, ls="-", ms=7, zorder=3, alpha=1.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, shrinkA=shrinkA, shrinkB=shrinkB,
                        connectionstyle=f"arc3,rad={rad}", ls=ls,
                        zorder=zorder, alpha=alpha)
    ax.add_patch(a)
    return a

def txt(ax, x, y, s, size=7, weight="normal", ha="center", va="center",
        color="k", style="normal", zorder=5, lh=1.25):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va,
                   color=color, style=style, zorder=zorder,
                   linespacing=lh)

def off(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight", pad_inches=0.02,
                dpi=400)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf (+.png)")

# ------------------------------------------------------------------- data ---
recs   = json.load(open(DATA / "hypothesis_scores_semantic.json", encoding="utf-8"))
kgraph = json.load(open(DATA / "knowledge_graph.json", encoding="utf-8"))
tdist  = json.load(open(DATA / "type_distribution.json", encoding="utf-8"))
bcnt   = json.load(open(DATA / "baseline_counts.json", encoding="utf-8"))

KG  = [r for r in recs if r["source"] == "LLMKG"]
LLM = [r for r in recs if r["source"] == "LLM"]

def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = st.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    df = (va+vb)**2 / (va**2/(len(a)-1) + vb**2/(len(b)-1))
    sp = math.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1)) /
                   (len(a)+len(b)-2))
    d = (a.mean()-b.mean())/sp
    u, pu = st.mannwhitneyu(a, b, alternative="two-sided")
    rrb = 2*u/(len(a)*len(b)) - 1
    return dict(mean_a=a.mean(), sd_a=a.std(ddof=1), mean_b=b.mean(),
                sd_b=b.std(ddof=1), t=t, df=df, p=p, d=d, U=u, p_U=pu,
                rrb=rrb, median_a=float(np.median(a)), median_b=float(np.median(b)),
                min_a=float(a.min()), min_b=float(b.min()),
                max_a=float(a.max()), max_b=float(b.max()))

S = {}
for dim in ("novelty", "innovativeness"):
    S[dim] = welch([r[f"{dim}_raw"] for r in KG], [r[f"{dim}_raw"] for r in LLM])
S["composite"] = welch([(r["novelty_raw"]+r["innovativeness_raw"])/2 for r in KG],
                       [(r["novelty_raw"]+r["innovativeness_raw"])/2 for r in LLM])
S["dispersion"] = welch([r["intra_dist"] for r in KG], [r["intra_dist"] for r in LLM])
for dim in ("novelty", "innovativeness"):
    lv = st.levene([r[f"{dim}_raw"] for r in KG], [r[f"{dim}_raw"] for r in LLM])
    S[dim]["levene_W"], S[dim]["levene_p"] = lv.statistic, lv.pvalue

nodes = kgraph["nodes"]
node_counts = {k: len(v) for k, v in nodes.items()}
rel_counts = {}
for r in kgraph["relationships"]:
    rel_counts[r["rel_type"]] = rel_counts.get(r["rel_type"], 0) + 1
weights = [r["properties"].get("weight") for r in kgraph["relationships"]
           if r.get("properties", {}).get("weight")]
confs = [n["properties"].get("confidence") for g in nodes.values() for n in g
         if n["properties"].get("confidence")]
theory_ids = set()
for g in nodes.values():
    for n in g:
        for t in (n["properties"].get("theory_reference") or []):
            theory_ids.add(t.split(":")[0].strip())
S["kg"] = dict(nodes=node_counts, total_nodes=sum(node_counts.values()),
               relations=rel_counts, total_relations=len(kgraph["relationships"]),
               cases=len(kgraph["cases"]), mean_weight=float(np.mean(weights)),
               mean_conf=float(np.mean(confs)), theories_invoked=len(theory_ids))
S["corpus"] = dict(collected=500, retained=419, groups=bcnt["baseline_groups"],
                   baseline_hypotheses=bcnt["baseline_hypotheses"],
                   undetermined=tdist.get("待推断", 0) + tdist.get("无法判断", 0),
                   n_kg_hyp=len(KG), n_llm_hyp=len(LLM))
json.dump(S, open(DATA / "stats.json", "w"), indent=1, default=float)
print("stats.json written:",
      {k: round(v, 4) if isinstance(v, float) else "" for k, v in S["novelty"].items() if k in ("p", "d")})

# ================================================================ FIGURE 1 ==
# System architecture -----------------------------------------------------
fig, ax = plt.subplots(figsize=(DC_W, 4.5)); off(ax)

BH, GAP = 0.182, 0.0155           # band height / gap
YS = {k: 0.006 + i*(BH+GAP) for i, k in enumerate("edcba")}
LAYERS = [
    ("a", "Data acquisition",     "#F7FBFF", "#9ECAE1"),
    ("b", "Knowledge grounding",  "#FFF5EB", "#FDBE85"),
    ("c", "Mechanism reasoning",  "#F7FCF5", "#A1D99B"),
    ("d", "Hypothesis discovery", "#FCFBFD", "#BCBDDC"),
    ("e", "Evaluation",           "#FFF7F3", "#FCC5C0"),
]
for key, name, fc, ec in LAYERS:
    y = YS[key]
    box(ax, (0.012, y), 0.976, BH, fc, ec, lw=0.9, r=0.012)
    txt(ax, 0.024, y+BH-0.024, key, size=8.6, weight="bold", ha="left")
    txt(ax, 0.044, y+BH-0.024, name, size=7.4, weight="bold", ha="left")

CH = 0.118                        # cell height
def cell(x, y, w, title, body, fc="white", ec="#888888", tsize=6.3,
         bsize=5.6, h=CH, tdy=0.021):
    box(ax, (x, y), w, h, fc, ec, lw=0.7, r=0.015, zorder=3)
    nt = title.count("\n")
    txt(ax, x+w/2, y+h-tdy-0.007*nt, title, size=tsize, weight="bold",
        zorder=6, lh=1.1)
    bh = y+h-0.036-0.014*nt        # top of body zone
    txt(ax, x+w/2, (bh+y+0.008)/2, body, size=bsize, zorder=6, lh=1.4)

# --- layer a: data
y = YS["a"]+0.012
cell(0.05, y, 0.175, "Police interrogation\nrecords",
     "500 scanned crash dossiers\n(de-identified)")
cell(0.285, y, 0.145, "OCR digitisation", "RapidOCR engine,\nbatch API")
cell(0.49, y, 0.16, "Quality screening", "narrative length\n$\\geq$ 100 characters")
cell(0.71, y, 0.20, "Analysable corpus",
     "419 first-person narratives;\ntriage into 81 case groups", fc="#EBF4FA")
for x0, x1 in [(0.225, 0.285), (0.43, 0.49), (0.65, 0.71)]:
    arrow(ax, (x0, y+CH/2), (x1, y+CH/2), lw=1.1)

# --- layer b: knowledge
y = YS["b"]+0.012
cell(0.05, y, 0.20, "Traffic-psychology\ntheory library",
     "50 curated theories (situation\nawareness, risk homeostasis,\nDBQ, cognitive load, …)")
cell(0.315, y, 0.245, "LLM structured extraction",
     "schema-constrained prompting;\n5 node types $\\cdot$ 7 relation types;\nevidence snippets + confidence")
cell(0.625, y, 0.315, "Incremental knowledge graph",
     "entity resolution (literal match / semantic\nsimilarity > 0.85 / new node);  83 nodes,\n107 weighted edges in 11-case deep pilot",
     fc="#FDF1E4")
arrow(ax, (0.25, y+CH/2), (0.315, y+CH/2), lw=1.1)
arrow(ax, (0.56, y+CH/2), (0.625, y+CH/2), lw=1.1)
arrow(ax, (0.81, YS["a"]), (0.44, y+CH+0.004), lw=1.2, rad=0.18)  # corpus -> extraction

# --- layer c: reasoning
y = YS["c"]+0.012
cell(0.05, y, 0.27, "Psychological-mechanism\ninterpretation",
     "trait $\\to$ tendency $\\to$ behaviour chains;\ntheory attribution for every node")
cell(0.385, y, 0.26, "Causal-chain assembly",
     "TRIGGERS / MANIFESTS_AS /\nCONTRIBUTES_TO / AMPLIFIES;\nedge weights 0.60–1.00")
cell(0.71, y, 0.23, "Cross-case regularities",
     "recurring motifs, e.g. habituation\nbias present in 10/11 cases")
arrow(ax, (0.32, y+CH/2), (0.385, y+CH/2), lw=1.1)
arrow(ax, (0.645, y+CH/2), (0.71, y+CH/2), lw=1.1)
arrow(ax, (0.782, YS["b"]), (0.782, y+CH+0.004), lw=1.2)

# --- layer d: discovery
y = YS["d"]+0.012
cell(0.05, y, 0.36, "Hypothesis generation (LLM + KG)",
     "graph-conditioned prompting per accident type;\n30 mechanism-level hypotheses (matched domain)",
     fc="#EFEDF5")
cell(0.475, y, 0.30, "Baseline (LLM only)",
     "direct narrative-to-hypothesis prompting;\n248 hypotheses / 81 groups (25 matched)")
cell(0.835, y, 0.115, "Output format",
     "IF–THEN form,\nvariables, test\ndesign, intervention", bsize=5.3)
arrow(ax, (0.23, YS["c"]), (0.23, y+CH+0.004), lw=1.2)
# direct route corpus -> baseline along right margin (bypasses KG)
ax.plot([0.91, 0.968, 0.968], [YS["a"]+0.07, YS["a"]+0.07, YS["d"]+0.155],
        color="#8C6BB1", lw=1.2, ls=(0, (4, 2.5)), zorder=2)
arrow(ax, (0.968, YS["d"]+0.155), (0.75, y+CH+0.004), lw=1.2,
      color="#8C6BB1", ls=(0, (4, 2.5)), rad=-0.15)
txt(ax, 0.982, 0.52, "raw narratives (no KG)", size=5.6, color="#8C6BB1",
    style="italic", ha="center", va="center")
ax.texts[-1].set_rotation(90)

# --- layer e: evaluation
y = YS["e"]+0.012
cell(0.05, y, 0.29, "Blinded automated judging",
     "LLM-as-judge (temperature 0);\nnovelty + innovativeness, 1–10;\npooled z-standardisation")
cell(0.40, y, 0.27, "Semantic-dispersion analysis",
     "embedding-based intra-group\ndistance of hypothesis statements")
cell(0.73, y, 0.22, "Expert panel review",
     "novelty, plausibility, practical\nvalue (three-dimension protocol)")
arrow(ax, (0.195, YS["d"]), (0.195, y+CH+0.004), lw=1.2)
arrow(ax, (0.60, YS["d"]), (0.60, y+CH+0.004), lw=1.2)
save(fig, "fig1_architecture")

# ================================================================ FIGURE 2 ==
# Knowledge-graph schema + instantiation ----------------------------------
fig = plt.figure(figsize=(DC_W, 3.3))
gs = fig.add_gridspec(2, 2, width_ratios=[1.52, 1], height_ratios=[1, 1],
                      wspace=0.52, hspace=0.55, left=0.01, right=0.985,
                      top=0.92, bottom=0.10)
axA = fig.add_subplot(gs[:, 0]); off(axA)
txt(axA, 0.012, 0.985, "a", size=9, weight="bold", ha="left")

def schema_node(x, y, w, h, kind, title, sub):
    box(axA, (x, y), w, h, NODE_COLS[kind], NODE_EDGE[kind], lw=1.0, r=0.02)
    txt(axA, x+w/2, y+h-0.05, title, size=6.6, weight="bold")
    txt(axA, x+w/2, y+h/2-0.045, sub, size=5.1, lh=1.35)

schema_node(0.025, 0.58, 0.25, 0.33, "Trait", "Trait",
            "individual state / disposition;\n7 dimensions A1–A7:\nphysiology · affect · risk\nperception · motivation · speed\nintent · attention · demographics")
schema_node(0.025, 0.10, 0.25, 0.30, "TraitCombination", "TraitCombination",
            "co-occurring trait pattern\nwith joint explanatory\npower; graded risk level\n(low → critical)")
schema_node(0.395, 0.34, 0.235, 0.35, "BehavioralTendency", "BehavioralTendency",
            "latent psychological\nmechanism; 5-step\nbehaviour chain (situation\n→ cognition → motivation\n→ decision → act)")
schema_node(0.745, 0.58, 0.235, 0.33, "Behavior", "Behavior",
            "observable driving act;\n6 classes: pre-drive decision,\nspeed, attention lapse,\ncontrol error, hazard\nresponse, post-crash")
schema_node(0.745, 0.10, 0.235, 0.30, "CausalFactor", "CausalFactor",
            "structural context;\n5 levels: road design,\nregulatory, occupational,\ncultural, vehicle")

def rel(p0, p1, label, n, rad, ly=None, lx=None, color=C_GREY):
    arrow(axA, p0, p1, color=color, lw=1.0, rad=rad, ms=8)
    if not label:
        return
    lx = lx if lx is not None else (p0[0]+p1[0])/2
    ly = ly if ly is not None else (p0[1]+p1[1])/2
    txt(axA, lx, ly, f"{label}\n(n={n})", size=5.3, color=color, lh=1.15,
        weight="bold")

rel((0.15, 0.58), (0.15, 0.40), "COMBINED_WITH", rel_counts.get("COMBINED_WITH", 0), 0.0, lx=0.15, ly=0.49)
rel((0.275, 0.25), (0.43, 0.35), "TRIGGERS", rel_counts.get("TRIGGERS", 0), -0.15, lx=0.345, ly=0.255)
rel((0.275, 0.74), (0.47, 0.69), "HAS_TRAIT*", rel_counts.get("HAS_TRAIT", 0), 0.18, lx=0.35, ly=0.80)
rel((0.63, 0.62), (0.745, 0.70), "MANIFESTS_AS", rel_counts.get("MANIFESTS_AS", 0), -0.15, lx=0.678, ly=0.565)
rel((0.83, 0.58), (0.83, 0.40), "LEADS_TO", rel_counts.get("LEADS_TO", 0), 0.0, lx=0.778, ly=0.49)
rel((0.745, 0.22), (0.63, 0.42), "CONTRIBUTES_TO", rel_counts.get("CONTRIBUTES_TO", 0), -0.15, lx=0.638, ly=0.29, color="#756BB1")
rel((0.895, 0.40), (0.895, 0.58), "", rel_counts.get("AMPLIFIES", 0), 0.30, color="#756BB1")
txt(axA, 0.955, 0.49, "AMPLIFIES\n(n=9)", size=5.3, color="#756BB1", lh=1.15, weight="bold")
txt(axA, 0.02, 0.028,
    "*HAS_TRAIT links tendencies and cases to constituent traits.  Every node carries\ntheory_reference, source_snippet, confidence and case_ids properties.",
    size=5.2, ha="left", color="#555555", lh=1.3)

axB = fig.add_subplot(gs[0, 1])
labels = ["Trait", "Trait\ncombination", "Behavioural\ntendency", "Behaviour", "Causal\nfactor"]
vals = [node_counts["traits"], node_counts["trait_combinations"],
        node_counts["behavioral_tendencies"], node_counts["behaviors"],
        node_counts["causal_factors"]]
cols = [NODE_COLS[k] for k in ("Trait", "TraitCombination", "BehavioralTendency", "Behavior", "CausalFactor")]
ecs  = [NODE_EDGE[k] for k in ("Trait", "TraitCombination", "BehavioralTendency", "Behavior", "CausalFactor")]
bars = axB.bar(range(5), vals, color=cols, edgecolor=ecs, lw=0.8, width=0.66)
for i, v in enumerate(vals):
    axB.text(i, v+0.5, str(v), ha="center", fontsize=6.2, fontweight="bold")
axB.set_xticks(range(5)); axB.set_xticklabels(labels, fontsize=5.4)
axB.set_ylabel("Nodes (n)"); axB.set_ylim(0, 29)
axB.spines[["top", "right"]].set_visible(False)
axB.set_title("b   Instantiated graph (11-case pilot)", loc="left", fontsize=7.2, fontweight="bold", pad=3)

axC = fig.add_subplot(gs[1, 1])
motifs = [
    ("Habituation bias (A3)", 10),
    ("Degraded situation awareness", 9),
    ("Calm affective state (A2)", 9),
    ("Delayed risk appraisal", 8),
    ("Freight time pressure", 6),
    ("Prof. driver $\\times$ habituation", 5),
    ("Visual distraction $\\times$ speeding", 3),
]
mcols = [NODE_COLS["Trait"], NODE_COLS["BehavioralTendency"], NODE_COLS["Trait"],
         NODE_COLS["BehavioralTendency"], NODE_COLS["CausalFactor"],
         NODE_COLS["TraitCombination"], NODE_COLS["TraitCombination"]]
mecs = [NODE_EDGE["Trait"], NODE_EDGE["BehavioralTendency"], NODE_EDGE["Trait"],
        NODE_EDGE["BehavioralTendency"], NODE_EDGE["CausalFactor"],
        NODE_EDGE["TraitCombination"], NODE_EDGE["TraitCombination"]]
ypos = np.arange(len(motifs))[::-1]
axC.barh(ypos, [m[1] for m in motifs], color=mcols, edgecolor=mecs, lw=0.8, height=0.62)
for y0, m in zip(ypos, motifs):
    axC.text(m[1]+0.15, y0, f"{m[1]}/11", va="center", fontsize=5.6)
axC.set_yticks(ypos); axC.set_yticklabels([m[0] for m in motifs], fontsize=5.2)
axC.set_xlabel("Cases containing node (n)"); axC.set_xlim(0, 12.4)
axC.spines[["top", "right"]].set_visible(False)
axC.set_title("c   Recurring nodes across cases", loc="left", fontsize=7.2, fontweight="bold", pad=3)
save(fig, "fig2_kg_schema")

# ================================================================ FIGURE 3 ==
# Two experimental paths ---------------------------------------------------
fig, ax = plt.subplots(figsize=(DC_W, 3.35)); off(ax)

def pcell(x, y, w, h, title, body, fc="white", ec="#888888", tsize=6.4, bsize=5.6):
    box(ax, (x, y), w, h, fc, ec, lw=0.8, r=0.018, zorder=3)
    nt = title.count("\n")
    txt(ax, x+w/2, y+h-0.042-0.018*nt, title, size=tsize, weight="bold", lh=1.15)
    txt(ax, x+w/2, (y+h-0.075-0.036*nt+y+0.012)/2, body, size=bsize, lh=1.4)

# shared input
pcell(0.012, 0.36, 0.16, 0.30, "Shared corpus",
      "500 records collected;\n419 retained after\nOCR quality screening;\ntype triage into\n81 case groups",
      fc="#EBF4FA", ec="#4292C6")
# Path A (baseline, top)
box(ax, (0.21, 0.585), 0.555, 0.385, "#FFF8EE", C_LLM, lw=1.2, r=0.015)
txt(ax, 0.225, 0.935, "Path A  —  LLM baseline (direct prompting)", size=7.2,
    weight="bold", ha="left", color="#8A5A00")
pcell(0.235, 0.63, 0.145, 0.22, "Group narratives",
      "raw testimony\ntexts grouped by\naccident type")
pcell(0.425, 0.63, 0.145, 0.22, "Direct generation",
      "single prompt:\nnarratives →\nresearch hypotheses")
pcell(0.615, 0.63, 0.13, 0.22, "Output A",
      "248 hypotheses,\n81 groups;\nmatched domain:\nn = 25", fc="#FDF3E0")
arrow(ax, (0.38, 0.74), (0.425, 0.74), lw=1.1)
arrow(ax, (0.57, 0.74), (0.615, 0.74), lw=1.1)

# Path B (KG, bottom)
box(ax, (0.21, 0.045), 0.555, 0.475, "#F2F8FD", C_KG, lw=1.2, r=0.015)
txt(ax, 0.225, 0.485, "Path B  —  LLM + knowledge graph (ours)", size=7.2,
    weight="bold", ha="left", color="#0B4F79")
pcell(0.235, 0.245, 0.145, 0.20, "Structured\nextraction",
      "theory-anchored\nnode/edge parsing\nper narrative")
pcell(0.425, 0.245, 0.145, 0.20, "Incremental KG",
      "entity resolution,\ncross-case merging;\n83 nodes, 107 edges")
pcell(0.615, 0.245, 0.13, 0.20, "Graph patterns",
      "recurring motifs,\ncausal chains,\nrisk combinations")
pcell(0.29, 0.062, 0.41, 0.14, "Graph-conditioned generation",
      "structured summaries + theory references → hypotheses;\nOutput B: n = 30 (6 groups $\\times$ 5)",
      fc="#E4F0F9")
arrow(ax, (0.38, 0.345), (0.425, 0.345), lw=1.1)
arrow(ax, (0.57, 0.345), (0.615, 0.345), lw=1.1)
arrow(ax, (0.68, 0.245), (0.60, 0.20), lw=1.1)
# corpus into both paths
arrow(ax, (0.17, 0.565), (0.235, 0.74), lw=1.3, color="#8A5A00")
arrow(ax, (0.17, 0.44), (0.235, 0.345), lw=1.3, color="#0B4F79")

# Evaluation (right)
box(ax, (0.80, 0.045), 0.185, 0.925, "#FBF3F7", "#C994C7", lw=1.2, r=0.015)
txt(ax, 0.8925, 0.925, "Identical blinded\nevaluation", size=7.0, weight="bold", lh=1.25)
pcell(0.812, 0.665, 0.16, 0.185, "LLM-as-judge",
      "novelty +\ninnovativeness\n(1–10, temp. 0)")
pcell(0.812, 0.43, 0.16, 0.175, "Pooled z-scores",
      "55 hypotheses\nstandardised\njointly")
pcell(0.812, 0.20, 0.16, 0.175, "Semantic\ndispersion",
      "intra-group\nembedding distance")
pcell(0.812, 0.062, 0.16, 0.09, "Expert panel",
      "3-dimension review", bsize=5.4)
arrow(ax, (0.745, 0.74), (0.812, 0.755), lw=1.3, color="#8A5A00")
arrow(ax, (0.70, 0.13), (0.812, 0.115), lw=1.3, color="#0B4F79", rad=0.0)
save(fig, "fig3_pipeline")

# ================================================================ FIGURE 4 ==
# Main results -------------------------------------------------------------
rng = np.random.default_rng(7)
fig = plt.figure(figsize=(DC_W, 4.3))
gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.52,
                      left=0.065, right=0.985, top=0.93, bottom=0.075)

def sigstars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."

def panel_scores(ax, dim, letter):
    a = np.array([r[f"{dim}_raw"] for r in KG], float)
    b = np.array([r[f"{dim}_raw"] for r in LLM], float)
    for i, (v, c) in enumerate([(a, C_KG), (b, C_LLM)]):
        parts = ax.violinplot([v], positions=[i], widths=0.7, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(c); pc.set_alpha(0.28); pc.set_edgecolor("none")
        x = rng.normal(i, 0.055, len(v))
        ax.scatter(x, v + rng.normal(0, 0.055, len(v)), s=9, color=c,
                   alpha=0.85, lw=0.3, edgecolor="white", zorder=3)
        ax.hlines(v.mean(), i-0.24, i+0.24, color="k", lw=1.3, zorder=4)
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), color="k", lw=0.9,
                    capsize=3, zorder=4)
    r = S[dim]
    if r["p"] < 1e-3:
        exp = int(math.floor(math.log10(r["p"])))
        pstr = f"$P$ = {r['p']/10**exp:.1f}$\\times$10$^{{{exp}}}$"
    else:
        pstr = f"$P$ = {r['p']:.3f}"
    y0 = max(a.max(), b.max()) + 0.55
    ax.plot([0, 0, 1, 1], [y0, y0+0.15, y0+0.15, y0], color="k", lw=0.8)
    ax.text(0.5, y0+0.28, f"{sigstars(r['p'])}   {pstr},  $d$ = {r['d']:.2f}",
            ha="center", fontsize=6.0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["LLM+KG\n(n=30)", "LLM\n(n=25)"])
    ax.set_ylim(2.4, 10.2); ax.set_ylabel(f"{dim.capitalize()} score (1–10)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{letter}   {dim.capitalize()}", loc="left", fontsize=7.4,
                 fontweight="bold", pad=3)

panel_scores(fig.add_subplot(gs[0, 0]), "novelty", "a")
panel_scores(fig.add_subplot(gs[0, 1]), "innovativeness", "b")

# (c) pooled z means with 95% CI
axc = fig.add_subplot(gs[0, 2])
pos = np.array([0, 1]); width = 0.34
for j, dim in enumerate(("novelty", "innovativeness")):
    for k, (grp, c) in enumerate([(KG, C_KG), (LLM, C_LLM)]):
        z = np.array([r[f"{dim}_z"] for r in grp], float)
        m, se = z.mean(), z.std(ddof=1)/math.sqrt(len(z))
        ci = se*st.t.ppf(0.975, len(z)-1)
        x = j + (k-0.5)*width
        axc.bar(x, m, width*0.92, color=c, alpha=0.85, zorder=3)
        axc.errorbar(x, m, yerr=ci, color="k", lw=0.9, capsize=3, zorder=4)
axc.axhline(0, color="#888888", lw=0.7)
axc.set_xticks(pos); axc.set_xticklabels(["Novelty", "Innovativeness"])
axc.set_ylabel("Pooled z-score (mean ± 95% CI)")
axc.spines[["top", "right"]].set_visible(False)
axc.set_title("c   Standardised effect", loc="left", fontsize=7.4,
              fontweight="bold", pad=3)
axc.legend(handles=[Line2D([], [], marker="s", ls="", color=C_KG, label="LLM+KG"),
                    Line2D([], [], marker="s", ls="", color=C_LLM, label="LLM")],
           frameon=False, loc="upper right", handletextpad=0.2)

# (d) score distributions – quality floor
axd = fig.add_subplot(gs[1, 0])
bins = np.arange(2.5, 11, 1)
a = [r["novelty_raw"] for r in KG]; b = [r["novelty_raw"] for r in LLM]
axd.hist(b, bins=bins, color=C_LLM, alpha=0.75, label="LLM", width=0.86,
         align="mid", edgecolor="white", lw=0.4)
axd.hist(a, bins=bins, color=C_KG, alpha=0.62, label="LLM+KG", width=0.86,
         align="mid", edgecolor="white", lw=0.4)
axd.axvline(6.5, color="#B22222", lw=0.9, ls=(0, (4, 2)))
axd.text(6.32, axd.get_ylim()[1]*0.99, "no LLM+KG hypothesis\nscored below 7",
         fontsize=5.6, color="#B22222", ha="right", va="top", linespacing=1.25)
axd.set_xlabel("Novelty score"); axd.set_ylabel("Hypotheses (n)")
axd.spines[["top", "right"]].set_visible(False)
axd.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.42))
axd.set_title("d   Quality floor (novelty)", loc="left", fontsize=7.4,
              fontweight="bold", pad=3)

# (e) semantic dispersion
axe = fig.add_subplot(gs[1, 1])
a = np.array([r["intra_dist"] for r in KG]); b = np.array([r["intra_dist"] for r in LLM])
bp = axe.boxplot([a, b], positions=[0, 1], widths=0.5, patch_artist=True,
                 medianprops=dict(color="k", lw=1.1),
                 boxprops=dict(lw=0.7), whiskerprops=dict(lw=0.7),
                 capprops=dict(lw=0.7), showfliers=False)
for patch, c in zip(bp["boxes"], [C_KG, C_LLM]):
    patch.set_facecolor(c); patch.set_alpha(0.45)
for i, (v, c) in enumerate([(a, C_KG), (b, C_LLM)]):
    axe.scatter(rng.normal(i, 0.06, len(v)), v, s=8, color=c, alpha=0.85,
                lw=0.3, edgecolor="white", zorder=3)
r = S["dispersion"]
y0 = max(a.max(), b.max()) + 0.02
axe.plot([0, 0, 1, 1], [y0, y0+0.008, y0+0.008, y0], color="k", lw=0.8)
axe.text(0.5, y0+0.014, f"***   $P$ = {r['p']:.2g}", ha="center", fontsize=6.0)
axe.set_xticks([0, 1]); axe.set_xticklabels(["LLM+KG", "LLM"])
axe.set_ylabel("Intra-group semantic distance")
axe.spines[["top", "right"]].set_visible(False)
axe.set_title("e   Thematic coherence", loc="left", fontsize=7.4,
              fontweight="bold", pad=3)

# (f) per-group composite means
axf = fig.add_subplot(gs[1, 2])
def group_means(rs):
    g = {}
    for r in rs:
        g.setdefault(r["accident_type"], []).append(
            (r["novelty_raw"]+r["innovativeness_raw"])/2)
    return sorted([float(np.mean(v)) for v in g.values()], reverse=True)
gm_kg, gm_llm = group_means(KG), group_means(LLM)
axf.scatter(rng.normal(0, 0.045, len(gm_kg)), gm_kg, s=26, color=C_KG,
            alpha=0.85, lw=0.4, edgecolor="white", zorder=3)
axf.scatter(1+rng.normal(0, 0.045, len(gm_llm)), gm_llm, s=26, color=C_LLM,
            alpha=0.85, lw=0.4, edgecolor="white", zorder=3)
axf.hlines(np.mean(gm_kg), -0.2, 0.2, color=C_KG, lw=1.4)
axf.hlines(np.mean(gm_llm), 0.8, 1.2, color=C_LLM, lw=1.4)
for v in gm_kg: axf.plot([-0.06, 0.06], [v, v], color=C_KG, lw=0.5, alpha=0.5)
axf.set_xticks([0, 1]); axf.set_xticklabels(["LLM+KG\n(6 groups)", "LLM\n(5 groups)"])
axf.set_xlim(-0.55, 1.55); axf.set_ylabel("Group-mean composite score")
axf.spines[["top", "right"]].set_visible(False)
axf.set_title("f   Robustness across case groups", loc="left", fontsize=7.4,
              fontweight="bold", pad=3)
save(fig, "fig4_main_results")

# ================================================================ FIGURE 5 ==
# Causal mechanism graph (real subgraph, rear-end domain) ------------------
fig, ax = plt.subplots(figsize=(DC_W, 4.15)); off(ax)

NW, NH = 0.148, 0.088
def gnode(cx, cy, kind, label, sub=None, w=NW, h=NH):
    box(ax, (cx-w/2, cy-h/2), w, h, NODE_COLS[kind], NODE_EDGE[kind], lw=1.0,
        r=0.02, zorder=4)
    dy = 0.012 if sub else 0.0
    txt(ax, cx, cy+dy, label, size=5.9, weight="bold", zorder=6, lh=1.15)
    if sub:
        txt(ax, cx, cy-h/2+0.018, sub, size=5.0, color="#444444", zorder=6)
    return (cx, cy)

REL_STYLE = {
    "TRIGGERS":       dict(color="#D95F02", ls="-"),
    "MANIFESTS_AS":   dict(color="#1B9E77", ls="-"),
    "CONTRIBUTES_TO": dict(color="#756BB1", ls=(0, (5, 2))),
    "AMPLIFIES":      dict(color="#C51B8A", ls=(0, (2, 1.6))),
    "COMBINED_WITH":  dict(color="#7A7A7A", ls="-"),
    "LEADS_TO":       dict(color="#4D4D4D", ls="-"),
}
def gedge(p0, p1, relt, w=None, rad=0.0, lab_frac=0.5, dx=0, dy=0.014, lw_scale=2.6):
    stl = REL_STYLE[relt]
    lw = 0.5 + (w or 0.7)*lw_scale*0.55
    arrow(ax, p0, p1, color=stl["color"], lw=lw, rad=rad, ls=stl["ls"],
          shrinkA=3, shrinkB=3, ms=8, alpha=0.9)
    if w is not None:
        lx = p0[0] + (p1[0]-p0[0])*lab_frac + dx
        ly = p0[1] + (p1[1]-p0[1])*lab_frac + dy
        txt(ax, lx, ly, f"{w:.2f}", size=5.2, color=stl["color"], weight="bold",
            zorder=7)

# column headers
for x, t in [(0.085, "Traits (A1–A7)"), (0.30, "Trait combinations"),
             (0.525, "Behavioural tendencies"), (0.755, "Behaviours"),
             (0.93, "Outcome")]:
    txt(ax, x, 0.975, t, size=6.6, weight="bold", color="#333333")

# Traits
t_prof  = gnode(0.085, 0.865, "Trait", "Professional\ndriver", "A7 · 7/11 cases")
t_hab   = gnode(0.085, 0.715, "Trait", "Habituation\nbias", "A3 · 10/11 cases")
t_vis   = gnode(0.085, 0.565, "Trait", "Visual\ndistraction", "A6 · 4/11 cases")
t_spd   = gnode(0.085, 0.415, "Trait", "Unintentional\nspeeding", "A5 · 4/11 cases")
t_slp   = gnode(0.085, 0.265, "Trait", "Sleep\ndeprivation", "A1 · 2/11 cases")
t_ddl   = gnode(0.085, 0.115, "Trait", "Occupational\ndeadline pressure", "A4 · 5/11 cases")

# Combinations
c_ph = gnode(0.30, 0.79, "TraitCombination", "Prof. driver ×\nhabituation bias", "risk: high · 5 cases")
c_vs = gnode(0.30, 0.49, "TraitCombination", "Visual distraction ×\nunint. speeding", "risk: critical · 3 cases")
c_sd = gnode(0.30, 0.19, "TraitCombination", "Sleep deprivation ×\ndeadline pressure", "risk: critical · 1 case")

for p0, p1 in [(t_prof, c_ph), (t_hab, c_ph), (t_vis, c_vs), (t_spd, c_vs),
               (t_slp, c_sd), (t_ddl, c_sd)]:
    gedge((p0[0]+NW/2, p0[1]), (p1[0]-NW/2, p1[1]), "COMBINED_WITH", None, rad=0.06)

# Tendencies
td_sa = gnode(0.525, 0.66, "BehavioralTendency", "Degraded situation\nawareness", "9/11 cases", w=0.165)
td_ra = gnode(0.525, 0.33, "BehavioralTendency", "Delayed risk\nappraisal", "8/11 cases", w=0.165)

gedge((c_ph[0]+NW/2, c_ph[1]), (td_sa[0]-0.0825, td_sa[1]+0.012), "TRIGGERS", 0.80, rad=0.05, lab_frac=0.30, dy=0.024, dx=0.008)
gedge((c_ph[0]+NW/2, c_ph[1]-0.02), (td_ra[0]-0.0825, td_ra[1]+0.02), "TRIGGERS", 0.80, rad=0.10, lab_frac=0.30, dy=0.020, dx=-0.012)
gedge((c_vs[0]+NW/2, c_vs[1]+0.012), (td_sa[0]-0.0825, td_sa[1]-0.018), "TRIGGERS", 0.85, rad=-0.06, lab_frac=0.26, dy=0.022, dx=0.010)
gedge((c_vs[0]+NW/2, c_vs[1]-0.012), (td_ra[0]-0.0825, td_ra[1]+0.005), "TRIGGERS", 0.70, rad=0.05, lab_frac=0.22, dy=-0.022)
gedge((c_sd[0]+NW/2, c_sd[1]), (td_sa[0]-0.0825, td_sa[1]-0.032), "TRIGGERS", 0.80, rad=-0.24, lab_frac=0.18, dy=-0.024)

# Behaviours
b_warn = gnode(0.755, 0.845, "Behavior", "Insufficient\nrear warning", "hazard response")
b_head = gnode(0.755, 0.655, "Behavior", "Inadequate\nheadway keeping", "speed mgmt.")
b_stop = gnode(0.755, 0.465, "Behavior", "Delayed stopping\nafter vehicle fault", "hazard response")
b_ramp = gnode(0.755, 0.275, "Behavior", "Missed emergency\nescape ramp", "control error")

gedge((td_sa[0]+0.0825, td_sa[1]+0.02), (b_warn[0]-NW/2, b_warn[1]), "MANIFESTS_AS", 0.60, rad=-0.08, lab_frac=0.82, dy=0.022)
gedge((td_sa[0]+0.0825, td_sa[1]), (b_head[0]-NW/2, b_head[1]), "MANIFESTS_AS", 0.80, rad=0.0, lab_frac=0.80, dy=0.018)
gedge((td_sa[0]+0.0825, td_sa[1]-0.02), (b_ramp[0]-NW/2, b_ramp[1]+0.02), "MANIFESTS_AS", 0.70, rad=0.12, lab_frac=0.86, dy=0.020, dx=-0.010)
gedge((td_ra[0]+0.0825, td_ra[1]+0.02), (b_warn[0]-NW/2, b_warn[1]-0.025), "MANIFESTS_AS", 0.75, rad=-0.28, lab_frac=0.10, dy=0.024, dx=-0.006)
gedge((td_ra[0]+0.0825, td_ra[1]), (b_stop[0]-NW/2, b_stop[1]), "MANIFESTS_AS", 0.90, rad=-0.05, lab_frac=0.80, dy=0.018)
gedge((td_ra[0]+0.0825, td_ra[1]-0.02), (b_ramp[0]-NW/2, b_ramp[1]-0.01), "MANIFESTS_AS", 0.80, rad=0.05, lab_frac=0.80, dy=-0.020)

# Outcome
out = gnode(0.93, 0.56, "Outcome", "Rear-end /\nfixed-object\ncollision", "11 index crashes", w=0.11, h=0.13)
for b in (b_warn, b_head, b_stop, b_ramp):
    gedge((b[0]+NW/2, b[1]), (out[0]-0.055, out[1]), "LEADS_TO", None, rad=0.0)

# Causal factors (bottom band)
cf_frt = gnode(0.40, 0.055, "CausalFactor", "Freight-industry\ntime pressure", "occupational · 6 cases", w=0.16)
cf_trf = gnode(0.64, 0.055, "CausalFactor", "Dense motorway\ntraffic", "road design · 4 cases", w=0.15)
gedge((cf_frt[0], cf_frt[1]+0.045), (td_ra[0]-0.05, td_ra[1]-0.055), "CONTRIBUTES_TO", 0.75, rad=-0.1, lab_frac=0.45, dx=-0.045, dy=0.005)
gedge((cf_frt[0]-0.08, cf_frt[1]), (t_ddl[0]+NW/2, t_ddl[1]-0.01), "AMPLIFIES", 0.80, rad=-0.1, lab_frac=0.5, dy=-0.024)
gedge((cf_trf[0], cf_trf[1]+0.045), (td_sa[0]+0.05, td_sa[1]-0.055), "CONTRIBUTES_TO", 0.60, rad=0.25, lab_frac=0.10, dx=0.030)
gedge((cf_trf[0]+0.075, cf_trf[1]+0.01), (b_head[0], b_head[1]-0.048), "AMPLIFIES", 0.70, rad=0.32, lab_frac=0.92, dx=0.040, dy=-0.005)

# legend below the drawing
leg_items = [("TRIGGERS", "TRIGGERS"), ("MANIFESTS_AS", "MANIFESTS_AS"),
             ("CONTRIBUTES_TO", "CONTRIBUTES_TO"), ("AMPLIFIES", "AMPLIFIES"),
             ("COMBINED_WITH", "COMBINED_WITH"), ("LEADS_TO", "case-linked outcome")]
handles = [Line2D([], [], color=REL_STYLE[k]["color"], ls=REL_STYLE[k]["ls"],
                  lw=1.5, label=lab) for k, lab in leg_items]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, -0.006),
          ncol=6, frameon=False, columnspacing=0.8, handlelength=1.4,
          borderaxespad=0, fontsize=5.8)
txt(ax, 0.985, 0.895, "edge labels: LLM-assigned evidence\nweight (0-1); line width $\\propto$ weight",
    size=5.2, ha="right", color="#555555", lh=1.3)
save(fig, "fig5_causal_mechanism")

print("All figures generated.")
