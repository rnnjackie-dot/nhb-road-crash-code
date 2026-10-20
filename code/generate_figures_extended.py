#!/usr/bin/env python3
"""
Extended experimental figures and statistics (Figs on corpus/extraction,
theory grounding, score analytics, robustness) for the NHB manuscript.
Reads real outputs from ../data; writes ../figures and ../data/stats_extended.json.
Run AFTER generate_figures.py.  Usage: python generate_figures_extended.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIGS = HERE.parent / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.0, "axes.titlesize": 7.4, "axes.labelsize": 7.0,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 6.0,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})
C_KG, C_LLM, C_GREY = "#0072B2", "#E69F00", "#4D4D4D"
TYPE_COLS = {"traits": "#B3CDE3", "trait_combinations": "#FBB4AE",
             "behavioral_tendencies": "#CCEBC5", "behaviors": "#FED9A6",
             "causal_factors": "#DECBE4"}
TYPE_EDGE = {"traits": "#3182BD", "trait_combinations": "#DE2D26",
             "behavioral_tendencies": "#31A354", "behaviors": "#E6550D",
             "causal_factors": "#756BB1"}
TYPE_LAB = {"traits": "Trait", "trait_combinations": "Trait combination",
            "behavioral_tendencies": "Behavioural tendency",
            "behaviors": "Behaviour", "causal_factors": "Causal factor"}
REL_ORDER = ["HAS_TRAIT", "CONTRIBUTES_TO", "MANIFESTS_AS", "LEADS_TO",
             "TRIGGERS", "AMPLIFIES", "COMBINED_WITH"]

def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf (+.png)")

# ---------------------------------------------------------------- load data --
recs   = json.load(open(DATA / "hypothesis_scores_semantic.json", encoding="utf-8"))
kg     = json.load(open(DATA / "knowledge_graph.json", encoding="utf-8"))
tdist  = json.load(open(DATA / "type_distribution.json", encoding="utf-8"))
KGH  = [r for r in recs if r["source"] == "LLMKG"]
LLMH = [r for r in recs if r["source"] == "LLM"]
rng = np.random.default_rng(2026)

# English names for accident categories (translation of triage labels)
CAT_EN = {
    "变道事故": "Lane change", "行人碰撞事故": "Pedestrian collision",
    "闯红灯事故": "Red-light running", "酒驾事故": "Drink-driving",
    "机动车与非机动车碰撞": "Motor vs non-motorised", "追尾事故": "Rear-end",
    "交通事故": "Unspecified collision", "车辆碰撞事故": "Vehicle collision",
    "分心驾驶事故": "Distracted driving", "机动车与电动车碰撞": "Car vs e-bike",
    "车辆碰撞": "Vehicle collision (other)", "倒车碰撞事故": "Reversing collision",
    "行人与机动车碰撞": "Pedestrian vs car", "待推断": "Undetermined",
    "无法判断": "Unclassifiable",
}
THEORY_EN = {
    "31": "Stress-performance theory", "40": "Automaticity theory",
    "9": "Risk homeostasis (Wilde)", "16": "Heuristic decision-making",
    "37": "Limited-capacity attention", "42": "Fatigue risk theory",
    "11": "Subjective risk theory", "25": "Big Five personality",
    "3": "Driver Behaviour Questionnaire", "10": "Risk compensation",
    "5": "Human-factors systems theory", "22": "Cognitive dissonance",
    "4": "Accident proneness", "8": "Situation awareness (Endsley)",
    "6": "Risk management (Rasmussen)", "32": "Emotion regulation",
    "43": "Sleep regulation", "1": "Human error (Reason)",
    "38": "Cognitive load theory", "12": "Hazard perception",
    "21": "Self-efficacy", "30": "Driving style theory",
    "2": "Swiss cheese model", "41": "Visual search theory",
    "time_discount_bias": "Temporal discounting",
}
THEORY_NAME_EN = {  # for hypothesis 'related_theories' free-text names
    "自动化": "Automaticity theory", "情境意识": "Situation awareness (Endsley)",
    "认知负荷": "Cognitive load theory", "有限资源": "Limited-capacity attention",
    "压力": "Stress-performance theory", "风险稳态": "Risk homeostasis (Wilde)",
    "驾驶行为问卷": "Driver Behaviour Questionnaire", "人为错误": "Human error (Reason)",
    "危险察觉": "Hazard perception", "视觉搜索": "Visual search theory",
    "风险管理": "Risk management (Rasmussen)", "疲劳": "Fatigue risk theory",
    "睡眠": "Sleep regulation", "启发式": "Heuristic decision-making",
    "时间折扣": "Temporal discounting", "风险补偿": "Risk compensation",
    "主观风险": "Subjective risk theory", "情绪调节": "Emotion regulation",
    "自我效能": "Self-efficacy", "人因": "Human-factors systems theory",
    "瑞士奶酪": "Swiss cheese model", "五大人格": "Big Five personality",
    "认知失调": "Cognitive dissonance", "事故倾向": "Accident proneness",
    "驾驶风格": "Driving style theory", "唤醒": "Arousal theory",
    "注意": "Limited-capacity attention", "期望": "Expectancy theory",
}

EX = {}  # extended stats accumulator

# =========================================================== FIGURE: corpus ==
fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.52, left=0.09, right=0.985,
                      top=0.94, bottom=0.09)

# (a) accident-type distribution (top specific categories)
axa = fig.add_subplot(gs[0, 0])
spec = [(k, v) for k, v in tdist.items() if k not in ("待推断", "无法判断")]
spec.sort(key=lambda x: -x[1])
top = spec[:12]
labels = [CAT_EN.get(k, k) for k, _ in top]
vals = [v for _, v in top]
y = np.arange(len(top))[::-1]
axa.barh(y, vals, color="#7FB3D5", edgecolor="#2E86C1", lw=0.6, height=0.68)
for yi, v in zip(y, vals):
    axa.text(v + 0.6, yi, str(v), va="center", fontsize=5.8)
axa.set_yticks(y); axa.set_yticklabels(labels, fontsize=5.8)
axa.set_xlabel("Narratives (n)")
axa.spines[["top", "right"]].set_visible(False)
axa.set_title("a   Accident-type triage (top specific categories, of 419)",
              loc="left", fontsize=7.4, fontweight="bold", pad=3)
n_und = tdist.get("待推断", 0) + tdist.get("无法判断", 0)
axa.text(0.97, 0.06, f"undetermined/unclassifiable: {n_und} (41.5%)",
         transform=axa.transAxes, ha="right", fontsize=5.6, color="#666666")
EX["top_categories"] = [(CAT_EN.get(k, k), v) for k, v in spec[:14]]

# (b) per-case extraction yield (stacked, from node case_ids)
axb = fig.add_subplot(gs[0, 1])
cases = [c["case_id"] for c in kg["cases"]]
bottom = np.zeros(len(cases))
for nt in TYPE_COLS:
    counts = np.array([sum(1 for n in kg["nodes"][nt]
                           if cid in (n.get("case_ids") or [])) for cid in cases], float)
    axb.bar(range(len(cases)), counts, bottom=bottom, color=TYPE_COLS[nt],
            edgecolor=TYPE_EDGE[nt], lw=0.5, width=0.7, label=TYPE_LAB[nt])
    bottom += counts
axb.set_xticks(range(len(cases)))
axb.set_xticklabels([c.replace("OCR-0", "C") for c in cases], fontsize=5.6)
axb.set_ylabel("Graph nodes linked to case (n)")
axb.set_xlabel("Pilot case")
axb.legend(frameon=False, fontsize=5.2, ncol=2, loc="upper right",
           handlelength=1.0, columnspacing=0.8)
axb.spines[["top", "right"]].set_visible(False)
axb.set_title("b   Per-case extraction yield (11-case pilot)",
              loc="left", fontsize=7.4, fontweight="bold", pad=3)
EX["per_case_nodes"] = {cid: float(b) for cid, b in zip(cases, bottom)}

# (c) node confidence by type
axc = fig.add_subplot(gs[1, 0])
conf_by = []
for nt in TYPE_COLS:
    conf_by.append([n["properties"].get("confidence") for n in kg["nodes"][nt]
                    if n["properties"].get("confidence") is not None])
bp = axc.boxplot(conf_by, positions=range(5), widths=0.55, patch_artist=True,
                 medianprops=dict(color="k", lw=1.0), showfliers=False,
                 boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
for patch, nt in zip(bp["boxes"], TYPE_COLS):
    patch.set_facecolor(TYPE_COLS[nt]); patch.set_alpha(0.8)
for i, (v, nt) in enumerate(zip(conf_by, TYPE_COLS)):
    axc.scatter(rng.normal(i, 0.07, len(v)), v, s=6, color=TYPE_EDGE[nt],
                alpha=0.75, lw=0, zorder=3)
axc.axhline(0.5, color="#B22222", lw=0.8, ls=(0, (4, 2)))
axc.text(4.45, 0.515, "acceptance threshold", fontsize=5.4, color="#B22222", ha="right")
axc.set_xticks(range(5))
axc.set_xticklabels(["Trait", "Trait\ncombo", "Behav.\ntendency", "Behaviour",
                     "Causal\nfactor"], fontsize=5.8)
axc.set_ylabel("Extraction confidence"); axc.set_ylim(0.45, 1.02)
axc.spines[["top", "right"]].set_visible(False)
axc.set_title("c   Node confidence by type", loc="left", fontsize=7.4,
              fontweight="bold", pad=3)
EX["conf_by_type"] = {TYPE_LAB[nt]: [float(np.mean(v)), float(np.std(v, ddof=1)), len(v)]
                      for nt, v in zip(TYPE_COLS, conf_by)}

# (d) edge weight by relation type
axd = fig.add_subplot(gs[1, 1])
w_by = {r: [] for r in REL_ORDER}
for r in kg["relationships"]:
    w = r.get("properties", {}).get("weight")
    if w is not None and r["rel_type"] in w_by:
        w_by[r["rel_type"]].append(w)
data = [w_by[r] for r in REL_ORDER]
bp = axd.boxplot(data, positions=range(len(REL_ORDER)), widths=0.55,
                 patch_artist=True, medianprops=dict(color="k", lw=1.0),
                 showfliers=False, boxprops=dict(lw=0.6),
                 whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
rel_cols = ["#8C8C8C", "#756BB1", "#1B9E77", "#4D4D4D", "#D95F02", "#C51B8A", "#7A7A7A"]
for patch, c in zip(bp["boxes"], rel_cols):
    patch.set_facecolor(c); patch.set_alpha(0.35)
for i, (v, c) in enumerate(zip(data, rel_cols)):
    axd.scatter(rng.normal(i, 0.07, len(v)), v, s=6, color=c, alpha=0.8, lw=0, zorder=3)
axd.set_xticks(range(len(REL_ORDER)))
axd.set_xticklabels(["HAS_\nTRAIT", "CONTRIB-\nUTES_TO", "MANIF-\nESTS_AS",
                     "LEADS_\nTO", "TRIG-\nGERS", "AMPLI-\nFIES", "COMBIN-\nED_WITH"],
                    fontsize=5.4)
axd.set_ylabel("Edge evidence weight"); axd.set_ylim(0.42, 1.05)
axd.spines[["top", "right"]].set_visible(False)
axd.set_title("d   Edge evidence weight by relation type (107 edges)",
              loc="left", fontsize=7.4, fontweight="bold", pad=3)
EX["weight_by_rel"] = {r: [float(np.mean(v)), float(np.std(v, ddof=1)), len(v)]
                       for r, v in w_by.items()}
save(fig, "fig6_corpus_extraction")

# ==================================================== FIGURE: theory ground ==
fig = plt.figure(figsize=(7.2, 4.4))
gs = fig.add_gridspec(2, 2, wspace=0.55, hspace=0.55, left=0.22, right=0.97,
                      top=0.94, bottom=0.10, width_ratios=[1, 1.15])

def theory_counts_nodes():
    cnt = {}
    for g in kg["nodes"].values():
        for n in g:
            seen = set()
            for t in (n["properties"].get("theory_reference") or []):
                tid = t.split(":")[0].strip()
                if tid not in seen:
                    seen.add(tid)
                    name = THEORY_EN.get(tid, tid)
                    cnt[name] = cnt.get(name, 0) + 1
    return cnt

# (a) theories invoked in KG node annotations
axa = fig.add_subplot(gs[0, 0])
tc = sorted(theory_counts_nodes().items(), key=lambda x: -x[1])[:12]
y = np.arange(len(tc))[::-1]
axa.barh(y, [v for _, v in tc], color="#F5B041", edgecolor="#B9770E", lw=0.6, height=0.68)
for yi, (_, v) in zip(y, tc):
    axa.text(v + 0.08, yi, str(v), va="center", fontsize=5.6)
axa.set_yticks(y); axa.set_yticklabels([k for k, _ in tc], fontsize=5.5)
axa.set_xlabel("Nodes citing theory (n)")
axa.spines[["top", "right"]].set_visible(False)
axa.set_title("a   Theory anchoring of graph nodes", loc="left",
              fontsize=7.4, fontweight="bold", pad=3)
EX["node_theories"] = tc

# (b) theories cited by LLM+KG hypotheses
axb = fig.add_subplot(gs[1, 0])
hyp_theories = {}
n_thr = []
for fp in (DATA / "hypotheses_kg").glob("*_hypotheses.json"):
    d = json.load(open(fp, encoding="utf-8"))
    for h in d.get("hypotheses", []):
        ths = h.get("related_theories") or []
        n_thr.append(len(ths))
        seen = set()
        for name in ths:
            en = next((v for k, v in THEORY_NAME_EN.items() if k in name), "Other/domain-specific")
            if en not in seen:
                seen.add(en)
                hyp_theories[en] = hyp_theories.get(en, 0) + 1
tc2 = sorted(hyp_theories.items(), key=lambda x: -x[1])[:12]
y = np.arange(len(tc2))[::-1]
axb.barh(y, [v for _, v in tc2], color="#85C1E9", edgecolor="#2874A6", lw=0.6, height=0.68)
for yi, (_, v) in zip(y, tc2):
    axb.text(v + 0.08, yi, str(v), va="center", fontsize=5.6)
axb.set_yticks(y); axb.set_yticklabels([k for k, _ in tc2], fontsize=5.5)
axb.set_xlabel("LLM+KG hypotheses citing theory (n)")
axb.spines[["top", "right"]].set_visible(False)
axb.set_title("b   Theory anchoring of generated hypotheses", loc="left",
              fontsize=7.4, fontweight="bold", pad=3)
EX["hyp_theories"] = tc2
EX["theories_per_hyp"] = [float(np.mean(n_thr)), float(np.std(n_thr, ddof=1))]

# (c) node x case coverage matrix (top recurring nodes)
axc = fig.add_subplot(gs[:, 1])
node_entries = []
for nt in TYPE_COLS:
    for n in kg["nodes"][nt]:
        ids = n.get("case_ids") or []
        if len(ids) >= 3:
            p = n["properties"]
            label = (p.get("value") or p.get("combination_label") or
                     p.get("tendency_label") or p.get("behavior_label") or
                     p.get("factor_label"))
            node_entries.append((nt, label, ids))
NODE_EN = {
    "习惯化偏差": "Habituation bias (T)", "平稳型": "Calm affect (T)",
    "无疲劳迹象": "No fatigue signs (T)", "职业驾驶员": "Professional driver (T)",
    "视觉分心型": "Visual distraction (T)", "无意识超速": "Unintentional speeding (T)",
    "驾龄3年": "3-yr licence (T)", "中年男性": "Middle-aged male (T)",
    "青年男性": "Young male (T)", "驾龄25年": "25-yr licence (T)",
    "职业驾驶员+习惯化偏差": "Prof. driver x habituation (C)",
    "视觉分心型+无意识超速": "Vis. distraction x speeding (C)",
    "情境意识不足": "Degraded situation awareness (BT)",
    "风险认知延迟": "Delayed risk appraisal (BT)",
    "事故后优先救人": "Post-crash rescue priority (B)",
    "事故后寻求救援": "Post-crash help seeking (B)",
    "未充分预警后方": "Insufficient rear warning (B)",
    "跟随前车减速": "Following-deceleration (B)",
    "高速路段车流量大": "Dense motorway traffic (CF)",
    "后车未保持安全距离": "Following car: short headway (CF)",
    "货运行业时限压力": "Freight time pressure (CF)",
    "职业期限压力": "Occupational deadline pressure (T)",
    "人际催促压力": "Interpersonal urging pressure (T)",
    "未保持安全车距": "Inadequate headway keeping (B)",
}
node_entries.sort(key=lambda x: -len(x[2]))
sel = node_entries[:12]
M = np.zeros((len(sel), len(cases)))
for i, (_, _, ids) in enumerate(sel):
    for j, cid in enumerate(cases):
        M[i, j] = 1 if cid in ids else 0
im_colors = {"traits": 1, "trait_combinations": 2, "behavioral_tendencies": 3,
             "behaviors": 4, "causal_factors": 5}
Mc = M.copy()
for i, (nt, _, _) in enumerate(sel):
    Mc[i] = M[i] * im_colors[nt]
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#FFFFFF", TYPE_COLS["traits"], TYPE_COLS["trait_combinations"],
                       TYPE_COLS["behavioral_tendencies"], TYPE_COLS["behaviors"],
                       TYPE_COLS["causal_factors"]])
axc.imshow(Mc, aspect="auto", cmap=cmap, vmin=0, vmax=5)
axc.set_xticks(range(len(cases)))
axc.set_xticklabels([c.replace("OCR-0", "C") for c in cases], fontsize=5.6)
axc.set_yticks(range(len(sel)))
axc.set_yticklabels([NODE_EN.get(lab, lab) for _, lab, _ in sel], fontsize=5.6)
for i in range(len(sel)):
    for j in range(len(cases)):
        if M[i, j]:
            axc.plot(j, i, marker="s", ms=3.2, color=TYPE_EDGE[sel[i][0]],
                     mec="white", mew=0.3, ls="")
axc.set_xticks(np.arange(-0.5, len(cases), 1), minor=True)
axc.set_yticks(np.arange(-0.5, len(sel), 1), minor=True)
axc.grid(which="minor", color="#DDDDDD", lw=0.4)
axc.tick_params(which="minor", length=0)
axc.set_xlabel("Pilot case")
axc.set_title("c   Cross-case recurrence of resolved entities ($\\geq$3 cases)",
              loc="left", fontsize=7.4, fontweight="bold", pad=3)
axc.text(1.0, -0.10, "T trait · C combination · BT tendency · B behaviour · CF causal factor",
         transform=axc.transAxes, ha="right", fontsize=5.2, color="#555555")
EX["recurrence_matrix_nodes"] = [(NODE_EN.get(lab, lab), int(M[i].sum()))
                                 for i, (_, lab, _) in enumerate(sel)]
save(fig, "fig7_theory_grounding")

# ===================================================== FIGURE: score detail ==
fig = plt.figure(figsize=(7.2, 4.4))
gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.52, left=0.08, right=0.985,
                      top=0.94, bottom=0.10)

nov_kg = np.array([r["novelty_raw"] for r in KGH], float)
inn_kg = np.array([r["innovativeness_raw"] for r in KGH], float)
nov_ll = np.array([r["novelty_raw"] for r in LLMH], float)
inn_ll = np.array([r["innovativeness_raw"] for r in LLMH], float)

# (a) joint scatter novelty x innovativeness
axa = fig.add_subplot(gs[0, 0])
axa.scatter(nov_ll + rng.normal(0, 0.09, len(nov_ll)),
            inn_ll + rng.normal(0, 0.09, len(inn_ll)),
            s=16, color=C_LLM, alpha=0.8, lw=0.4, edgecolor="white", label="LLM (n=25)")
axa.scatter(nov_kg + rng.normal(0, 0.09, len(nov_kg)),
            inn_kg + rng.normal(0, 0.09, len(inn_kg)),
            s=16, color=C_KG, alpha=0.8, lw=0.4, edgecolor="white", label="LLM+KG (n=30)")
allnov = np.concatenate([nov_kg, nov_ll]); allinn = np.concatenate([inn_kg, inn_ll])
rho, prho = st.spearmanr(allnov, allinn)
axa.plot([2.5, 8.5], np.poly1d(np.polyfit(allnov, allinn, 1))([2.5, 8.5]),
         color="#666666", lw=0.8, ls=(0, (4, 2)))
axa.text(0.03, 0.96, f"Spearman $\\rho$ = {rho:.2f}, $P$ = {prho:.1e}",
         transform=axa.transAxes, fontsize=5.8, va="top")
axa.set_xlabel("Novelty score"); axa.set_ylabel("Innovativeness score")
axa.set_xlim(2.4, 8.9); axa.set_ylim(3.4, 8.4)
axa.legend(frameon=False, loc="lower right", handletextpad=0.2)
axa.spines[["top", "right"]].set_visible(False)
axa.set_title("a   Joint score distribution", loc="left", fontsize=7.4,
              fontweight="bold", pad=3)
EX["spearman_dims"] = [float(rho), float(prho)]

# (b) ECDFs of composite
axb = fig.add_subplot(gs[0, 1])
comp_kg = (nov_kg + inn_kg) / 2; comp_ll = (nov_ll + inn_ll) / 2
for v, c, lab in [(comp_ll, C_LLM, "LLM"), (comp_kg, C_KG, "LLM+KG")]:
    xs = np.sort(v); ys = np.arange(1, len(v) + 1) / len(v)
    axb.step(np.concatenate([[xs[0]], xs]), np.concatenate([[0], ys]),
             where="post", color=c, lw=1.4, label=lab)
ks, pks = st.ks_2samp(comp_kg, comp_ll)
axb.text(0.03, 0.96, f"Kolmogorov--Smirnov $D$ = {ks:.2f}, $P$ = {pks:.4f}",
         transform=axb.transAxes, fontsize=5.8, va="top")
axb.set_xlabel("Composite score (mean of two dimensions)")
axb.set_ylabel("Cumulative proportion")
axb.legend(frameon=False, loc="lower right")
axb.spines[["top", "right"]].set_visible(False)
axb.set_title("b   Empirical CDFs (composite)", loc="left", fontsize=7.4,
              fontweight="bold", pad=3)
EX["ks_composite"] = [float(ks), float(pks)]

# (c) per-group means, both dims
axc = fig.add_subplot(gs[1, 0])
GRP_EN = {
    "追尾碰撞死亡事故": "Fatal rear-end", "连环追尾碰撞": "Multi-veh. pile-up",
    "追尾碰撞": "Rear-end", "单车侧翻闯入施工区伤人事故": "Rollover, work zone",
    "单车撞固定物事故": "Fixed object", "单车撞固定物（护栏）": "Guardrail strike",
    "追尾事故": "Rear-end", "连环追尾事故": "Multi-veh. pile-up",
    "疲劳驾驶事故": "Fatigue-related", "撞固定物事故": "Fixed object",
    "施工区域碰撞事故": "Work zone",
}
def per_group(rs):
    g = {}
    for r in rs:
        g.setdefault(r["accident_type"], []).append(r)
    out = []
    for k, v in g.items():
        out.append((GRP_EN.get(k, k),
                    float(np.mean([x["novelty_raw"] for x in v])),
                    float(np.mean([x["innovativeness_raw"] for x in v])), len(v)))
    return sorted(out, key=lambda x: -(x[1] + x[2]))
gk, gl = per_group(KGH), per_group(LLMH)
names = [f"{n}" for n, *_ in gk] + [f"{n}" for n, *_ in gl]
y = np.arange(len(names))[::-1]
for i, (n, nv, iv, cnt) in enumerate(gk + gl):
    col = C_KG if i < len(gk) else C_LLM
    axc.plot([nv], [y[i]], marker="o", ms=4.5, color=col, ls="")
    axc.plot([iv], [y[i]], marker="s", ms=4, color=col, mfc="white", mew=1.1, ls="")
axc.axhline(y[len(gk)] + 0.5, color="#999999", lw=0.7, ls=(0, (3, 2)))
axc.text(8.35, y[0] + 0.15, "LLM+KG groups", fontsize=5.6, color=C_KG,
         ha="right", weight="bold")
axc.text(8.35, y[len(gk)] - 0.05, "LLM groups", fontsize=5.6, color=C_LLM,
         ha="right", weight="bold")
axc.set_yticks(y); axc.set_yticklabels(names, fontsize=5.6)
axc.set_xlabel("Group mean score"); axc.set_xlim(5.2, 8.5)
axc.legend(handles=[Line2D([], [], marker="o", ls="", color="#555555", label="Novelty"),
                    Line2D([], [], marker="s", ls="", color="#555555", mfc="white",
                           label="Innovativeness")],
           frameon=False, loc="lower right", fontsize=5.4, handletextpad=0.2)
axc.spines[["top", "right"]].set_visible(False)
axc.set_title("c   Group-level means (11 case groups)", loc="left",
              fontsize=7.4, fontweight="bold", pad=3)
EX["per_group"] = {"KG": gk, "LLM": gl}

# (d) innovativeness distribution (complement of main-results panel)
axd = fig.add_subplot(gs[1, 1])
bins = np.arange(3.5, 9.5, 1)
axd.hist(inn_ll, bins=bins, color=C_LLM, alpha=0.75, label="LLM", width=0.86,
         align="mid", edgecolor="white", lw=0.4)
axd.hist(inn_kg, bins=bins, color=C_KG, alpha=0.62, label="LLM+KG", width=0.86,
         align="mid", edgecolor="white", lw=0.4)
axd.axvline(5.5, color="#B22222", lw=0.9, ls=(0, (4, 2)))
axd.text(5.38, axd.get_ylim()[1] * 0.98, "no LLM+KG hypothesis\nscored below 6",
         fontsize=5.6, color="#B22222", ha="right", va="top", linespacing=1.25)
axd.set_xlabel("Innovativeness score"); axd.set_ylabel("Hypotheses (n)")
axd.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.42))
axd.spines[["top", "right"]].set_visible(False)
axd.set_title("d   Quality floor (innovativeness)", loc="left", fontsize=7.4,
              fontweight="bold", pad=3)
save(fig, "fig8_score_analytics")

# ======================================================= FIGURE: robustness ==
fig = plt.figure(figsize=(7.2, 4.3))
gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.55, left=0.08, right=0.985,
                      top=0.94, bottom=0.10)
NBOOT = NPERM = 10000

def boot_diff(a, b):
    d = np.empty(NBOOT)
    for i in range(NBOOT):
        d[i] = rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
    return d

def perm_p(a, b):
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); n = len(a); cnt = 0
    for _ in range(NPERM):
        rng.shuffle(pool)
        if abs(pool[:n].mean() - pool[n:].mean()) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (NPERM + 1)

for j, (dim, a, b) in enumerate([("novelty", nov_kg, nov_ll),
                                 ("innovativeness", inn_kg, inn_ll)]):
    ax = fig.add_subplot(gs[0, j])
    bd = boot_diff(a, b)
    lo, hi = np.percentile(bd, [2.5, 97.5])
    ax.hist(bd, bins=42, color="#A9CCE3" if j == 0 else "#AED6B5",
            edgecolor="white", lw=0.3)
    ax.axvline(0, color="#B22222", lw=1.0)
    ax.axvline(bd.mean(), color="k", lw=1.0)
    ax.axvspan(lo, hi, color="#555555", alpha=0.12)
    ax.text(0.03, 0.96,
            f"$\\Delta$ = {a.mean()-b.mean():.2f}\n95% CI [{lo:.2f}, {hi:.2f}]\n"
            f"P(boot $\\Delta\\leq$0) = {float((bd<=0).mean()):.4f}",
            transform=ax.transAxes, fontsize=5.8, va="top", linespacing=1.4)
    ax.set_xlabel(f"Bootstrap mean difference, {dim} (LLM+KG $-$ LLM)")
    ax.set_ylabel("Resamples (n)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{'ab'[j]}   Bootstrap difference ({dim})", loc="left",
                 fontsize=7.4, fontweight="bold", pad=3)
    EX[f"boot_{dim}"] = [float(a.mean()-b.mean()), float(lo), float(hi),
                         float((bd <= 0).mean())]

# (c) permutation tests
axc = fig.add_subplot(gs[1, 0])
res = {}
for dim, a, b, c in [("Novelty", nov_kg, nov_ll, "#2E86C1"),
                     ("Innovativeness", inn_kg, inn_ll, "#28B463")]:
    pool = np.concatenate([a, b]); n = len(a)
    null = np.empty(NPERM)
    for i in range(NPERM):
        rng.shuffle(pool)
        null[i] = pool[:n].mean() - pool[n:].mean()
    obs = a.mean() - b.mean()
    p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    p = (p * NPERM + 1) / (NPERM + 1)
    res[dim] = (obs, p)
    axc.hist(null, bins=42, alpha=0.45, color=c, edgecolor="white", lw=0.2,
             label=f"{dim}: $P$ = {p:.4f}")
    axc.axvline(obs, color=c, lw=1.4)
axc.set_xlabel("Null distribution of mean difference (10,000 permutations)")
axc.set_ylabel("Permutations (n)")
axc.legend(frameon=False, fontsize=5.6, loc="upper left")
axc.spines[["top", "right"]].set_visible(False)
axc.set_title("c   Permutation tests (observed vs null)", loc="left",
              fontsize=7.4, fontweight="bold", pad=3)
EX["perm"] = {k: [float(v[0]), float(v[1])] for k, v in res.items()}

# (d) leave-one-group-out effect sizes
axd = fig.add_subplot(gs[1, 1])
def cohend(a, b):
    sp = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) /
                   (len(a)+len(b)-2))
    return (a.mean()-b.mean())/sp
groups = sorted({r["accident_type"] for r in recs})
logo = {"novelty": [], "innovativeness": []}
lognames = []
for gname in groups:
    K = [r for r in KGH if r["accident_type"] != gname]
    L = [r for r in LLMH if r["accident_type"] != gname]
    lognames.append(GRP_EN.get(gname, gname))
    for dim in logo:
        logo[dim].append(cohend(np.array([r[f"{dim}_raw"] for r in K]),
                                np.array([r[f"{dim}_raw"] for r in L])))
y = np.arange(len(groups))[::-1]
axd.plot(logo["novelty"], y, "o", ms=4.5, color="#2E86C1", label="Novelty")
axd.plot(logo["innovativeness"], y, "s", ms=4, color="#28B463", mfc="white",
         mew=1.1, label="Innovativeness")
full_d_nov = cohend(nov_kg, nov_ll); full_d_inn = cohend(inn_kg, inn_ll)
axd.axvline(full_d_nov, color="#2E86C1", lw=0.8, ls=(0, (4, 2)))
axd.axvline(full_d_inn, color="#28B463", lw=0.8, ls=(0, (4, 2)))
axd.axvline(0, color="#B22222", lw=0.8)
axd.set_yticks(y); axd.set_yticklabels(lognames, fontsize=5.5)
axd.set_xlabel("Cohen's $d$ after excluding group")
axd.legend(frameon=False, fontsize=5.4, loc="lower left", handletextpad=0.2)
axd.spines[["top", "right"]].set_visible(False)
axd.set_title("d   Leave-one-group-out sensitivity", loc="left",
              fontsize=7.4, fontweight="bold", pad=3)
EX["logo"] = {"groups": lognames,
              "novelty": [float(x) for x in logo["novelty"]],
              "innovativeness": [float(x) for x in logo["innovativeness"]],
              "full": [float(full_d_nov), float(full_d_inn)]}
save(fig, "fig9_robustness")

json.dump(EX, open(DATA / "stats_extended.json", "w"), indent=1, default=float)
print("stats_extended.json written")
print("\nKey extended numbers:")
print(" spearman dims:", EX["spearman_dims"])
print(" KS composite:", EX["ks_composite"])
print(" boot novelty:", EX["boot_novelty"])
print(" boot innovativeness:", EX["boot_innovativeness"])
print(" perm:", EX["perm"])
print(" LOGO ranges: nov [%.3f, %.3f]  inn [%.3f, %.3f]" % (
    min(EX["logo"]["novelty"]), max(EX["logo"]["novelty"]),
    min(EX["logo"]["innovativeness"]), max(EX["logo"]["innovativeness"])))
print(" theories/hyp:", EX["theories_per_hyp"])
print(" conf by type:", EX["conf_by_type"])
print(" weight by rel:", EX["weight_by_rel"])
print(" per-group:", EX["per_group"])
