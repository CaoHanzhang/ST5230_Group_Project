"""
ST5230 Group 23 — Formatting Bias in LLM-as-a-Judge
Full Analysis Script

Sections:
  0. Setup & data loading
  1. Data overview & sanity checks
  2. Position bias (standalone finding)
  3. Format leakage (standalone finding)
  4. Pairwise win rates by format, judge, category
  5. Statistical tests (binomial + Bonferroni + bootstrap CIs)
  6. Inter-judge agreement (Cohen's κ)
  7. Visualisations (4 figures from proposal)
  8. Failure mode diagnosis (swap inconsistency + cross-judge disagreement)
  9. Summary table for report
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, textwrap, os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── output folder ──────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_output")
os.makedirs(OUT, exist_ok=True)

def savefig(name):
    path = f"{OUT}/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved {path}")

# ══════════════════════════════════════════════════════════════════════════════
# 0. Load data
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("0. Loading data")
print("=" * 70)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(SCRIPT_DIR, "data", "judgements.csv"))

# Resolve actual winning format from positional label
df["winning_format"] = np.where(df["final_winner"] == "A", df["format_a"], df["format_b"])

# Canonical format order for display
FORMAT_ORDER = ["prose", "bullet", "header"]
JUDGE_LABELS  = {"gpt4o_mini": "GPT-4o-mini", "claude_haiku": "Claude Haiku"}
CAT_ORDER     = ["factual", "opinion", "procedural"]
PALETTE       = {"prose": "#4C72B0", "bullet": "#DD8452", "header": "#55A868"}

print(f"Rows         : {len(df)}")
print(f"Questions    : {df['id'].nunique()}")
print(f"Judges       : {df['judge'].unique().tolist()}")
print(f"Format pairs : {df.groupby(['format_a','format_b']).size().reset_index().values.tolist()}")
print(f"Categories   : {df['our_category'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Data overview & sanity checks
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. Data overview")
print("=" * 70)

# Expected: 60 q × 3 pairs × 2 judges = 360 rows
assert len(df) == 360, f"Expected 360 rows, got {len(df)}"

swap_rate = df["swap_consistent"].mean()
leakage_rate = df["any_format_leakage"].mean()
print(f"Swap-consistent rate : {swap_rate:.1%}")
print(f"Format leakage rate  : {leakage_rate:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Position bias (standalone finding)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. Position bias")
print("=" * 70)

# In AB ordering  : chose position B (second) ↔ winner_ab == "B"
# In BA ordering  : format_b is in position A, format_a is in position B
#                   → chose position B (second) ↔ winner_ba == "A" (format_a won)
df["chose_second_ab"] = (df["winner_ab"] == "B")
df["chose_second_ba"] = (df["winner_ba"] == "A")

total_judgments = 2 * len(df)   # each row has two individual verdicts
chose_second    = df["chose_second_ab"].sum() + df["chose_second_ba"].sum()
pos_bias_rate   = chose_second / total_judgments

bt = stats.binomtest(int(chose_second), total_judgments, p=0.5, alternative="two-sided")
print(f"Total individual verdicts : {total_judgments}")
print(f"Chose second (position B) : {chose_second}  ({pos_bias_rate:.1%})")
print(f"Binomial p-value          : {bt.pvalue:.4g}   "
      f"({'significant' if bt.pvalue < 0.05 else 'not significant'} at α=0.05)")

# By judge
print("\nPosition bias by judge:")
for judge, g in df.groupby("judge"):
    n  = 2 * len(g)
    k  = g["chose_second_ab"].sum() + g["chose_second_ba"].sum()
    r  = k / n
    p  = stats.binomtest(int(k), n, p=0.5).pvalue
    print(f"  {JUDGE_LABELS[judge]:20s}: {r:.1%}  (p={p:.4g})")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Format leakage (standalone finding)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. Format leakage")
print("=" * 70)

print(f"Overall any_format_leakage : {df['any_format_leakage'].mean():.1%}")
print(f"  format_leakage_ab        : {df['format_leakage_ab'].mean():.1%}")
print(f"  format_leakage_ba        : {df['format_leakage_ba'].mean():.1%}")

print("\nLeakage by judge:")
for judge, g in df.groupby("judge"):
    print(f"  {JUDGE_LABELS[judge]:20s}: {g['any_format_leakage'].mean():.1%}")

print("\nLeakage by format pair:")
for (fa, fb), g in df.groupby(["format_a", "format_b"]):
    print(f"  {fa:7s} vs {fb:7s}: {g['any_format_leakage'].mean():.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Pairwise win rates
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. Pairwise win rates")
print("=" * 70)

def win_rate_table(data):
    """Build a long-form win-rate table from a slice of df."""
    rows = []
    for (fa, fb), g in data.groupby(["format_a", "format_b"]):
        n_total = len(g)
        n_a     = (g["final_winner"] == "A").sum()
        n_b     = (g["final_winner"] == "B").sum()
        rows.append({"format_a": fa, "format_b": fb,
                     "n": n_total, "wins_a": n_a, "wins_b": n_b,
                     "winrate_a": n_a / n_total, "winrate_b": n_b / n_total})
    return pd.DataFrame(rows)

# Overall
print("\nOverall win rates:")
wrt_all = win_rate_table(df)
print(wrt_all.to_string(index=False))

# By judge
print("\nWin rates by judge:")
for judge, g in df.groupby("judge"):
    print(f"\n  [{JUDGE_LABELS[judge]}]")
    print(win_rate_table(g).to_string(index=False))

# By category
print("\nWin rates by category:")
for cat, g in df.groupby("our_category"):
    print(f"\n  [{cat}]")
    print(win_rate_table(g).to_string(index=False))

# ── overall format win rate (summing across all pairwise comparisons) ──
def format_win_counts(data):
    """Count total wins for each format across all pairwise comparisons."""
    counts = {f: 0 for f in FORMAT_ORDER}
    totals = {f: 0 for f in FORMAT_ORDER}
    for _, row in data.iterrows():
        winner = row["winning_format"]
        loser  = row["format_b"] if row["final_winner"] == "A" else row["format_a"]
        counts[winner] = counts.get(winner, 0) + 1
        totals[winner] = totals.get(winner, 0) + 1
        totals[loser]  = totals.get(loser, 0)  + 1
    return counts, totals

print("\nOverall format win counts (across all comparisons):")
counts, totals = format_win_counts(df)
for f in FORMAT_ORDER:
    c, t = counts[f], totals[f]
    print(f"  {f:8s}: {c}/{t} wins  ({c/t:.1%})")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Statistical tests: binomial + bootstrap CIs + Bonferroni
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. Statistical tests")
print("=" * 70)

N_COMPARISONS = 18   # 3 pairs × 3 categories × 2 judges
ALPHA         = 0.05
ALPHA_BONF    = ALPHA / N_COMPARISONS
N_BOOT        = 5000

print(f"Bonferroni-corrected α : {ALPHA_BONF:.4f}  ({N_COMPARISONS} comparisons)")

def bootstrap_winrate(wins_arr, n_boot=N_BOOT):
    """Bootstrap 95% CI for win rate. wins_arr is 0/1 array."""
    boot_rates = [np.mean(np.random.choice(wins_arr, size=len(wins_arr), replace=True))
                  for _ in range(n_boot)]
    return np.percentile(boot_rates, [2.5, 97.5])

stat_rows = []

for judge, gj in df.groupby("judge"):
    for cat, gc in gj.groupby("our_category"):
        for (fa, fb), gp in gc.groupby(["format_a", "format_b"]):
            wins_a = (gp["final_winner"] == "A").values.astype(int)
            n      = len(wins_a)
            k      = wins_a.sum()
            wr     = k / n
            ci_lo, ci_hi = bootstrap_winrate(wins_a)
            p_val  = stats.binomtest(int(k), n, p=0.5, alternative="two-sided").pvalue
            sig    = p_val < ALPHA_BONF
            stat_rows.append({
                "judge": JUDGE_LABELS[judge],
                "category": cat,
                "format_a": fa, "format_b": fb,
                "n": n, "wins_a": k,
                "winrate_a": wr,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "p_value": p_val,
                "significant_bonf": sig,
            })

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(f"{OUT}/statistical_tests.csv", index=False)
print(f"\nFull results saved to {OUT}/statistical_tests.csv")

n_sig = stat_df["significant_bonf"].sum()
print(f"Significant after Bonferroni correction: {n_sig} / {len(stat_df)} comparisons")

if n_sig > 0:
    print("\nSignificant comparisons:")
    sig = stat_df[stat_df["significant_bonf"]]
    for _, r in sig.iterrows():
        winner_fmt = r["format_a"] if r["winrate_a"] > 0.5 else r["format_b"]
        loser_fmt  = r["format_b"] if r["winrate_a"] > 0.5 else r["format_a"]
        print(f"  [{r['judge']}] {r['category']:12s}: "
              f"{winner_fmt} beats {loser_fmt}  "
              f"wr={r['winrate_a']:.2f}  p={r['p_value']:.4g}")

# ── bootstrap CI for overall pairwise win rates ──────────────────────────────
print("\nBootstrap 95% CIs for overall pairwise win rate (format_a):")
for (fa, fb), g in df.groupby(["format_a", "format_b"]):
    wins = (g["final_winner"] == "A").values.astype(int)
    wr   = wins.mean()
    lo, hi = bootstrap_winrate(wins)
    p    = stats.binomtest(int(wins.sum()), len(wins), p=0.5).pvalue
    print(f"  {fa:7s} vs {fb:7s}: wr={wr:.2f}  95%CI=[{lo:.2f},{hi:.2f}]  p={p:.4g}")

# ══════════════════════════════════════════════════════════════════════════════
# 5b. Bootstrap rank stability
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5b. Bootstrap rank stability")
print("=" * 70)

# For each bootstrap resample of the 60 questions, compute each format's
# overall win rate and record the resulting rank order.
# A stable finding = header > bullet > prose holds in nearly all resamples.

question_ids = df["id"].unique()   # 60 questions

rank_records = []
for _ in range(N_BOOT):
    # Resample questions with replacement
    sampled_ids = np.random.choice(question_ids, size=len(question_ids), replace=True)
    boot = df[df["id"].isin(sampled_ids)]

    win_counts  = {f: 0 for f in FORMAT_ORDER}
    total_counts = {f: 0 for f in FORMAT_ORDER}
    for _, row in boot.iterrows():
        winner = row["winning_format"]
        loser  = row["format_b"] if row["final_winner"] == "A" else row["format_a"]
        win_counts[winner]   += 1
        total_counts[winner] += 1
        total_counts[loser]  += 1

    rates = {f: win_counts[f] / total_counts[f] if total_counts[f] > 0 else 0
             for f in FORMAT_ORDER}

    # Rank: 1 = highest win rate
    sorted_fmts = sorted(FORMAT_ORDER, key=lambda f: rates[f], reverse=True)
    rank_records.append({
        "rank1": sorted_fmts[0],
        "rank2": sorted_fmts[1],
        "rank3": sorted_fmts[2],
        "wr_prose":  rates["prose"],
        "wr_bullet": rates["bullet"],
        "wr_header": rates["header"],
    })

rank_df = pd.DataFrame(rank_records)

# How often does the observed ordering hold?
canonical = ("header", "bullet", "prose")
canonical_rate = ((rank_df["rank1"] == canonical[0]) &
                  (rank_df["rank2"] == canonical[1]) &
                  (rank_df["rank3"] == canonical[2])).mean()

print(f"Canonical ordering (header > bullet > prose): {canonical_rate:.1%} of bootstrap resamples")

# Distribution of rank-1 format
print("\nRank-1 format distribution across resamples:")
for fmt, pct in rank_df["rank1"].value_counts(normalize=True).items():
    print(f"  {fmt:8s}: {pct:.1%}")

# Bootstrap CIs for each format's win rate
print("\nBootstrap 95% CIs for overall format win rates:")
for fmt in FORMAT_ORDER:
    col  = f"wr_{fmt}"
    lo   = np.percentile(rank_df[col], 2.5)
    hi   = np.percentile(rank_df[col], 97.5)
    mean = rank_df[col].mean()
    print(f"  {fmt:8s}: mean={mean:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]")

# Full rank ordering frequency table
print("\nFull rank ordering frequencies (top 5):")
rank_df["ordering"] = rank_df["rank1"] + " > " + rank_df["rank2"] + " > " + rank_df["rank3"]
top_orders = rank_df["ordering"].value_counts(normalize=True).head(5)
for order, pct in top_orders.items():
    print(f"  {order}: {pct:.1%}")

# ── visualisation: bootstrap win-rate distributions ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
for ax, fmt in zip(axes, FORMAT_ORDER):
    col = f"wr_{fmt}"
    ax.hist(rank_df[col], bins=40, color=PALETTE[fmt], edgecolor="white", linewidth=0.5)
    lo = np.percentile(rank_df[col], 2.5)
    hi = np.percentile(rank_df[col], 97.5)
    ax.axvline(rank_df[col].mean(), color="black", linewidth=1.5, linestyle="-",  label=f"mean={rank_df[col].mean():.2f}")
    ax.axvline(lo, color="black", linewidth=1,   linestyle="--", label=f"95% CI")
    ax.axvline(hi, color="black", linewidth=1,   linestyle="--")
    ax.set_title(fmt.capitalize())
    ax.set_xlabel("Win rate")
    ax.legend(fontsize=8)

axes[0].set_ylabel("Bootstrap frequency")
plt.suptitle(f"Bootstrap Win-Rate Distributions (n={N_BOOT} resamples)\n"
             f"Canonical ordering header > bullet > prose holds in {canonical_rate:.1%} of resamples",
             fontsize=11)
plt.tight_layout()
savefig("5b_bootstrap_rank_stability")

rank_df.to_csv(f"{OUT}/bootstrap_rank_stability.csv", index=False)
print(f"\nBootstrap samples saved → {OUT}/bootstrap_rank_stability.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Inter-judge agreement (Cohen's κ)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. Inter-judge agreement (Cohen's κ)")
print("=" * 70)

# Pivot: one row per (question_id, format_pair), two columns for judges
pivot = df.pivot_table(
    index=["id", "format_a", "format_b"],
    columns="judge",
    values="winning_format",
    aggfunc="first"
).reset_index()

pivot.columns.name = None
judges = [j for j in df["judge"].unique()]
j1, j2 = judges[0], judges[1]

# Drop rows where either judge's result is missing
pivot_clean = pivot.dropna(subset=[j1, j2])
print(f"Matched pairs for κ: {len(pivot_clean)}")

kappa_overall = cohen_kappa_score(pivot_clean[j1], pivot_clean[j2])
print(f"Cohen's κ (overall): {kappa_overall:.3f}")

# By category
pivot_cat = pivot_clean.merge(
    df[["id", "format_a", "format_b", "our_category"]].drop_duplicates(),
    on=["id", "format_a", "format_b"], how="left"
)

kappa_rows = []
print("\nCohen's κ by category:")
for cat, g in pivot_cat.groupby("our_category"):
    if len(g) < 2:
        continue
    k = cohen_kappa_score(g[j1], g[j2])
    agree_rate = (g[j1] == g[j2]).mean()
    kappa_rows.append({"category": cat, "kappa": k, "agreement_rate": agree_rate, "n": len(g)})
    print(f"  {cat:12s}: κ={k:.3f}  agree={agree_rate:.1%}  (n={len(g)})")

kappa_df = pd.DataFrame(kappa_rows)

# By format pair
print("\nCohen's κ by format pair:")
for (fa, fb), g in pivot_cat.groupby(["format_a", "format_b"]):
    if len(g) < 2:
        continue
    k = cohen_kappa_score(g[j1], g[j2])
    agree_rate = (g[j1] == g[j2]).mean()
    print(f"  {fa:7s} vs {fb:7s}: κ={k:.3f}  agree={agree_rate:.1%}  (n={len(g)})")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Visualisations
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. Visualisations")
print("=" * 70)

sns.set_style("whitegrid")

# ── 7a. Grouped bar chart: win rates by format × judge ───────────────────────
print("\n  [7a] Grouped bar chart: win rates by format × judge")

# Aggregate: for each judge, for each format, compute overall win rate across all pairs
records = []
for judge, gj in df.groupby("judge"):
    c, t = format_win_counts(gj)
    for fmt in FORMAT_ORDER:
        records.append({
            "judge": JUDGE_LABELS[judge],
            "format": fmt,
            "win_rate": c[fmt] / t[fmt] if t[fmt] > 0 else 0,
            "wins": c[fmt],
            "total": t[fmt],
        })
    # add overall
    c_all, t_all = format_win_counts(df)
    # already done for df
for fmt in FORMAT_ORDER:
    records.append({
        "judge": "Overall",
        "format": fmt,
        "win_rate": counts[fmt] / totals[fmt] if totals[fmt] > 0 else 0,
        "wins": counts[fmt],
        "total": totals[fmt],
    })

bar_df = pd.DataFrame(records)

fig, ax = plt.subplots(figsize=(9, 5))
judge_groups = ["GPT-4o-mini", "Claude Haiku", "Overall"]
x  = np.arange(len(judge_groups))
w  = 0.22
for i, fmt in enumerate(FORMAT_ORDER):
    vals = [bar_df[(bar_df["judge"] == jg) & (bar_df["format"] == fmt)]["win_rate"].values[0]
            for jg in judge_groups]
    bars = ax.bar(x + (i - 1) * w, vals, width=w, label=fmt.capitalize(),
                  color=PALETTE[fmt], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.0%}", ha="center", va="bottom", fontsize=8)

ax.axhline(1/3, color="black", linestyle="--", linewidth=1, label="Chance (33%)")
ax.set_xticks(x)
ax.set_xticklabels(judge_groups)
ax.set_ylabel("Overall win rate")
ax.set_ylim(0, 0.75)
ax.set_title("Format Win Rates by Judge Model\n(across all pairwise comparisons)")
ax.legend(title="Format", loc="upper left")
savefig("7a_winrate_by_judge")

# ── 7b. Heatmap: win rate of format_a across format×category×judge ───────────
print("  [7b] Heatmap: win rate by format pair × category × judge")

# Build matrix: rows = (judge, format_a), cols = (category, format_b) — or simpler:
# one heatmap per judge showing format_a win rate by format_pair × category
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, judge in zip(axes, df["judge"].unique()):
    gj   = df[df["judge"] == judge]
    pairs = [f"{fa} vs {fb}" for fa, fb in [("prose","bullet"),("prose","header"),("bullet","header")]]
    heat  = np.zeros((len(pairs), len(CAT_ORDER)))

    for r, (fa, fb) in enumerate([("prose","bullet"),("prose","header"),("bullet","header")]):
        for c, cat in enumerate(CAT_ORDER):
            sub = gj[(gj["format_a"] == fa) & (gj["format_b"] == fb) & (gj["our_category"] == cat)]
            if len(sub) == 0:
                heat[r, c] = np.nan
            else:
                heat[r, c] = (sub["final_winner"] == "A").mean()

    heat_df = pd.DataFrame(heat, index=pairs, columns=[c.capitalize() for c in CAT_ORDER])

    sns.heatmap(heat_df, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, center=0.5, linewidths=0.5,
                cbar_kws={"label": "Win rate of left format"})
    ax.set_title(f"{JUDGE_LABELS[judge]}\n(cell = win rate of format on left)")
    ax.set_xlabel("Question category")
    ax.set_ylabel("Format comparison")

plt.suptitle("Heatmap: Win Rate of Left Format by Pair × Category × Judge", y=1.02, fontsize=13)
plt.tight_layout()
savefig("7b_heatmap_format_category_judge")

# ── 7c. Inter-judge agreement table (visual) ──────────────────────────────────
print("  [7c] Inter-judge agreement table")

agree_rows = []
# Overall
agree_rows.append({
    "Scope": "Overall", "κ": f"{kappa_overall:.3f}",
    "Agreement rate": f"{(pivot_clean[j1]==pivot_clean[j2]).mean():.1%}",
    "n": len(pivot_clean)
})
# By category
for cat, g in pivot_cat.groupby("our_category"):
    if len(g) < 2: continue
    k = cohen_kappa_score(g[j1], g[j2])
    agree_rows.append({
        "Scope": f"  {cat.capitalize()}", "κ": f"{k:.3f}",
        "Agreement rate": f"{(g[j1]==g[j2]).mean():.1%}",
        "n": len(g)
    })
# By format pair
for (fa, fb), g in pivot_cat.groupby(["format_a", "format_b"]):
    if len(g) < 2: continue
    k = cohen_kappa_score(g[j1], g[j2])
    agree_rows.append({
        "Scope": f"  {fa} vs {fb}", "κ": f"{k:.3f}",
        "Agreement rate": f"{(g[j1]==g[j2]).mean():.1%}",
        "n": len(g)
    })

agree_tbl = pd.DataFrame(agree_rows)

fig, ax = plt.subplots(figsize=(7, len(agree_rows) * 0.45 + 1))
ax.axis("off")
tbl = ax.table(
    cellText=agree_tbl.values,
    colLabels=agree_tbl.columns,
    cellLoc="center", loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.1, 1.6)
# Header row colour
for j in range(len(agree_tbl.columns)):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
# Alternate row shading
for i in range(1, len(agree_rows) + 1):
    for j in range(len(agree_tbl.columns)):
        tbl[i, j].set_facecolor("#f0f0f0" if i % 2 == 0 else "white")

ax.set_title(f"Inter-Judge Agreement: {JUDGE_LABELS[j1]} vs {JUDGE_LABELS[j2]}",
             fontsize=12, pad=12, fontweight="bold")
savefig("7c_interjudge_agreement")

# ── 7d. Pairwise confusion table: wins for each format against each other ─────
print("  [7d] Pairwise win/loss table")

# confusion_matrix[i][j] = number of times FORMAT_ORDER[i] beat FORMAT_ORDER[j]
n_fmt = len(FORMAT_ORDER)
conf_mat = np.full((n_fmt, n_fmt), np.nan)

for (fa, fb), g in df.groupby(["format_a", "format_b"]):
    i = FORMAT_ORDER.index(fa)
    j = FORMAT_ORDER.index(fb)
    wins_a = (g["final_winner"] == "A").sum()
    wins_b = (g["final_winner"] == "B").sum()
    conf_mat[i, j] = wins_a
    conf_mat[j, i] = wins_b

conf_df = pd.DataFrame(conf_mat, index=FORMAT_ORDER, columns=FORMAT_ORDER)

fig, ax = plt.subplots(figsize=(6, 5))
mask = np.eye(n_fmt, dtype=bool)
annot = np.where(mask, "—",
                 conf_df.values.astype(object))
# Also show win rate in cells
for i in range(n_fmt):
    for j in range(n_fmt):
        if i != j and not np.isnan(conf_mat[i, j]) and not np.isnan(conf_mat[j, i]):
            total = conf_mat[i, j] + conf_mat[j, i]
            if total > 0:
                annot[i, j] = f"{int(conf_mat[i,j])}\n({conf_mat[i,j]/total:.0%})"

sns.heatmap(
    pd.DataFrame(np.where(mask, np.nan, conf_mat),
                 index=FORMAT_ORDER, columns=FORMAT_ORDER),
    ax=ax, annot=annot, fmt="", cmap="Blues",
    linewidths=1, linecolor="white",
    cbar_kws={"label": "Wins"},
    mask=mask
)
ax.set_title("Pairwise Win/Loss Table\n(row format beats column format; N=120 per cell)")
ax.set_xlabel("Loses to →")
ax.set_ylabel("← Wins against")
plt.tight_layout()
savefig("7d_pairwise_confusion")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Failure mode diagnosis
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. Failure mode diagnosis")
print("=" * 70)

# ── 8a. Swap inconsistencies ──────────────────────────────────────────────────
swap_incon = df[~df["swap_consistent"]].copy()
print(f"\nSwap-inconsistent rows: {len(swap_incon)} / {len(df)}  ({len(swap_incon)/len(df):.1%})")

print("\nSwap inconsistency by judge:")
for judge, g in swap_incon.groupby("judge"):
    print(f"  {JUDGE_LABELS[judge]:20s}: {len(g)} cases")

print("\nSwap inconsistency by format pair:")
for (fa, fb), g in swap_incon.groupby(["format_a", "format_b"]):
    print(f"  {fa:7s} vs {fb:7s}: {len(g)} cases")

print("\nSwap inconsistency by category:")
for cat, g in swap_incon.groupby("our_category"):
    print(f"  {cat:12s}: {len(g)} cases")

# Sample 10 cases for qualitative inspection
sample_swap = swap_incon.sample(min(10, len(swap_incon)), random_state=42)
sample_cols = ["id", "our_category", "format_a", "format_b", "judge",
               "winner_ab", "winner_ba", "reason_ab", "reason_ba"]
sample_swap[sample_cols].to_csv(f"{OUT}/sample_swap_inconsistent.csv", index=False)
print(f"\nSampled {len(sample_swap)} swap-inconsistent cases → {OUT}/sample_swap_inconsistent.csv")

# ── 8b. Cross-judge disagreements ────────────────────────────────────────────
# pivot_clean already has both judges; find rows where they disagree
cross_disagree = pivot_clean[pivot_clean[j1] != pivot_clean[j2]].copy()
cross_disagree = cross_disagree.merge(
    df[["id","format_a","format_b","our_category","question"]].drop_duplicates(),
    on=["id","format_a","format_b"], how="left"
)

print(f"\nCross-judge disagreements: {len(cross_disagree)} / {len(pivot_clean)}  "
      f"({len(cross_disagree)/len(pivot_clean):.1%})")

print("\nDisagreements by category:")
for cat, g in cross_disagree.groupby("our_category"):
    print(f"  {cat:12s}: {len(g)} cases")

print("\nDisagreements by format pair:")
for (fa, fb), g in cross_disagree.groupby(["format_a", "format_b"]):
    print(f"  {fa:7s} vs {fb:7s}: {len(g)} cases")

# For each disagreement, attach both judges' reasons
reasons = df[["id","format_a","format_b","judge","winning_format","reason_ab","reason_ba"]].copy()
reasons_pivot = reasons.pivot_table(
    index=["id","format_a","format_b"],
    columns="judge",
    values=["winning_format","reason_ab","reason_ba"],
    aggfunc="first"
).reset_index()
reasons_pivot.columns = ["_".join(c).strip("_") for c in reasons_pivot.columns]

cross_detail = cross_disagree.merge(reasons_pivot, on=["id","format_a","format_b"], how="left")
cross_detail.to_csv(f"{OUT}/sample_cross_judge_disagreement.csv", index=False)
print(f"Full cross-judge disagreement details → {OUT}/sample_cross_judge_disagreement.csv")

# ── 8c. Length confound check ─────────────────────────────────────────────────
print("\nLength confound check:")
print("  Does the longer response tend to win?")
df["longer_wins"] = (
    ((df["len_a"] > df["len_b"]) & (df["final_winner"] == "A")) |
    ((df["len_b"] > df["len_a"]) & (df["final_winner"] == "B"))
)
longer_rate = df["longer_wins"].mean()
bt_len = stats.binomtest(int(df["longer_wins"].sum()), len(df), p=0.5)
print(f"  Longer-wins rate: {longer_rate:.1%}  (p={bt_len.pvalue:.4g})")

# ══════════════════════════════════════════════════════════════════════════════
# 9. Summary table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("9. Summary for report")
print("=" * 70)

summary_lines = [
    ("Total judgements (rows)", len(df)),
    ("Total individual verdicts (AB + BA)", total_judgments),
    ("Swap-consistent rate", f"{swap_rate:.1%}"),
    ("Format leakage rate (any)", f"{leakage_rate:.1%}"),
    ("Chose 2nd position (position bias)", f"{pos_bias_rate:.1%}"),
    ("Position bias p-value", f"{bt.pvalue:.4g}"),
    ("Sig. comparisons after Bonferroni", f"{n_sig}/{len(stat_df)}"),
    ("Cohen's κ (overall)", f"{kappa_overall:.3f}"),
    ("Swap-inconsistent cases", len(swap_incon)),
    ("Cross-judge disagreements", len(cross_disagree)),
    ("Longer-response wins rate", f"{longer_rate:.1%}"),
]

print(f"\n{'Metric':<45} {'Value':>10}")
print("-" * 57)
for k, v in summary_lines:
    print(f"{k:<45} {str(v):>10}")

# Save summary
summary_df = pd.DataFrame(summary_lines, columns=["Metric", "Value"])
summary_df.to_csv(f"{OUT}/summary.csv", index=False)
print(f"\nSummary → {OUT}/summary.csv")

print("\n" + "=" * 70)
print("Analysis complete. All outputs in:", OUT)
print("=" * 70)
