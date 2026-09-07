"""Revised Figure 4: endpoint, search, stopping AND process generation.

Panels
  a  matched-K best-found minus human silhouette (per base stimulus)
  b  search efficiency (human vs point-information-oracle actions)
  c  out-of-fold stopping-hazard log loss (process vs quality vs threshold)
  d  K(N): human vs quality-guided stopping walk vs greedy builder,
     separately for Anchor/full (exp3) and Anchor/aperture (exp4);
     Model-H dashed for reference.
  e  granularity slope summary (group-number slope per additional point),
     models on rows, b = 0 reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

import build_revised_five as brf  # noqa: E402

# Match the final Figure 5 export and typography standard in every format.
matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.size": 6.2,
    "axes.labelsize": 6.2,
    "axes.titlesize": 7,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 5.5,
    "savefig.dpi": 400,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
RESULTS = Path(__file__).resolve().parents[2] / "revision_analysis" / "revision_analysis" / "outputs"

COL = brf.COL
LABEL = brf.LABEL
panel = brf.panel
mean_ci = brf.mean_ci
summary_point = brf.summary_point
raincloud_v = brf.raincloud_v
raincloud_h = brf.raincloud_h
fmt_p = brf.fmt_p
save = brf.save
rng = np.random.default_rng(17)

BINS = [(10, 15), (20, 25), (30, 35), (40, 40)]


def clean_zero(x, _pos=None):
    """Use a plain 0 while retaining compact decimals elsewhere."""
    if np.isclose(x, 0):
        return "0"
    return f"{x:g}"


def binned(human: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mids = [0.5 * (lo + hi) for lo, hi in BINS]
    means = []
    for lo, hi in BINS:
        g = human[(human.n_points >= lo) & (human.n_points <= hi)]
        means.append(g.k.mean() if len(g) else np.nan)
    return np.asarray(mids, float), np.asarray(means, float)


def panel_d(ax, cond, oos):
    """Horizontal rainclouds for participant-level human and generated K(N)."""
    d = oos[oos.condition == cond]
    bins = [(40.0, "40"), (32.5, "30–35"), (22.5, "20–25"), (12.5, "10–15")]
    centers = np.arange(4)[::-1]
    for i, ((nmid, _), y) in enumerate(zip(bins, centers)):
        dd = d[d.n_mid == nmid]
        raincloud_h(ax, dd.observed_k.to_numpy(float), y + .13, "#444444",
                    width=.20, seed=510+i, alpha=.12, summary_offset=.040)
        raincloud_h(ax, dd.predicted_k.to_numpy(float), y - .13, COL["quality"],
                    width=.20, seed=520+i, alpha=.14, summary_offset=.040)
    bias = float((d.observed_k - d.predicted_k).mean())
    g = d.groupby("n_mid")[["observed_k", "predicted_k"]].mean().reset_index()
    hs = np.polyfit(g.n_mid, g.observed_k, 1)[0]
    ms = np.polyfit(g.n_mid, g.predicted_k, 1)[0]
    ax.text(.98, .98, f"slope {hs:.3f} / {ms:.3f}   bias {bias:.2f} K",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
            color="#666666")
    ax.set_yticks(centers, [b[1] for b in bins])
    ax.set_ylim(-.5, 3.5)
    ax.set_xlim(1.45, 5.55)
    ax.set_xlabel("Mean group count (K)")
    ax.set_ylabel("Point-count bin (N)")
    ax.set_title(LABEL[cond], fontsize=7, loc="left", color="#444444")
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color="#444444",
                              label="Human"),
                       Line2D([0], [0], marker="o", ls="", color=COL["quality"],
                              label="Held-out walk")],
              loc="lower right", fontsize=6.2, frameon=False, handletextpad=.25)


def panel_oos_diagnostic(ax_slope, ax_bias, cond, oos, letter):
    """Direct participant-level tests of slope transfer and K calibration."""
    d = oos[oos.condition == cond]
    slope_diff, k_bias = [], []
    for _, g in d.groupby("person_hash"):
        hs = np.polyfit(g.n_mid.to_numpy(float), g.observed_k.to_numpy(float), 1)[0]
        ms = np.polyfit(g.n_mid.to_numpy(float), g.predicted_k.to_numpy(float), 1)[0]
        slope_diff.append(hs - ms)
        k_bias.append(float((g.observed_k - g.predicted_k).mean()))
    slope_diff = np.asarray(slope_diff)
    k_bias = np.asarray(k_bias)

    ax_slope.axvspan(-.01, .01, color="#DDF2E8", alpha=.95, zorder=0)
    ax_slope.axvline(0, color="#777777", lw=.7, ls=(0, (3, 2)), zorder=1)
    raincloud_h(ax_slope, slope_diff, 0, COL["quality"], width=.31,
                seed=610 + int(cond[-1]), alpha=.17, summary_offset=.065)
    ax_slope.set_xlim(-.07, .25)
    ax_slope.set_ylim(-.43, .43)
    ax_slope.set_yticks([])
    ax_slope.set_xlabel("Human − OOS slope (K per point)", fontsize=6.5)
    ax_slope.xaxis.set_major_formatter(FuncFormatter(clean_zero))
    ax_slope.tick_params(axis="both", labelsize=5.5)
    tost = fmt_p(.276 if cond == "exp3" else .006)
    ax_slope.text(.98, .92, tost, transform=ax_slope.transAxes,
                  ha="right", va="top", fontsize=6.2, color="#555555")
    ax_slope.set_title(LABEL[cond], fontsize=7, loc="left", color="#444444")
    panel(ax_slope, letter)

    ax_bias.axvline(0, color="#777777", lw=.7, ls=(0, (3, 2)), zorder=1)
    raincloud_h(ax_bias, k_bias, 0, COL["comparator"], width=.31,
                seed=620 + int(cond[-1]), alpha=.17, summary_offset=.065)
    ax_bias.set_xlim(-2.7, 4.6)
    ax_bias.set_ylim(-.43, .43)
    ax_bias.set_yticks([])
    ax_bias.set_xlabel("Human − OOS group count (K)", fontsize=6.5)
    ax_bias.xaxis.set_major_formatter(FuncFormatter(clean_zero))
    ax_bias.tick_params(axis="both", labelsize=5.5)


def panel_f(ax, ablation):
    order = ["full", "process_only", "process_no_npoints", "fixed_k=3"]
    labels = ["Full", "Process", "− Numerosity", "Fixed K"]
    yy = np.arange(len(order))[::-1]
    offsets = {"exp3": .10, "exp4": -.10}
    colors = {"exp3": "#168F67", "exp4": "#57BFA1"}
    for e in ("exp3", "exp4"):
        d = ablation.query("condition == @e").set_index("covariate_set")
        for y, key in zip(yy, order):
            r = d.loc[key]
            yp = y + offsets[e]
            ax.plot([r.ci_lo, r.ci_hi], [yp, yp], color=colors[e], lw=5.0,
                    alpha=.30, solid_capstyle="round", zorder=2)
            ax.plot([r.ci_lo, r.ci_hi], [yp, yp], color=colors[e], lw=.85,
                    solid_capstyle="round", zorder=3)
            ax.plot(r.slope, yp, marker="|", ms=8.0, mew=1.35,
                    color=colors[e], zorder=4)
    ax.axvline(0, color="#888888", lw=.65, ls=(0, (3, 2)))
    ax.set_yticks(yy, labels)
    ax.tick_params(axis="both", labelsize=5.5, pad=2)
    ax.set_xlim(-.025, .052)
    ax.set_xlabel("Generated K(N) slope", fontsize=6.2)
    ax.xaxis.set_major_formatter(FuncFormatter(clean_zero))
    ax.legend(handles=[Line2D([0], [0], lw=3, color=colors["exp3"], label="Full view"),
                       Line2D([0], [0], lw=3, color=colors["exp4"], label="Aperture")],
              loc="lower center", bbox_to_anchor=(.5, 1.10), ncol=2,
              fontsize=5.5, columnspacing=.8, handletextpad=.3)


def panel_d_old(ax, cond, human, walk, greedy, mh):
    h = human[human.condition == cond]
    w = walk[walk.condition == cond]
    g = greedy[greedy.condition == cond]
    m = mh[mh.condition == cond]

    ax.scatter(g.n_points, g.k, s=9, color=COL["comparator"], alpha=.30,
               edgecolor="none", zorder=1, label="Greedy")
    gx, gy = binned(g)
    ax.plot(gx, gy, color=COL["comparator"], lw=1.6, marker="o", ms=3.6,
            mfc="white", mec=COL["comparator"], mew=.7, zorder=2)

    ax.scatter(h.n_points, h.k, s=9, color="#111111", alpha=.30,
               edgecolor="none", zorder=3, label="Human")
    hx, hy = binned(h)
    ax.plot(hx, hy, color="#111111", lw=2.0, marker="o", ms=3.8,
            mfc="white", mec="#111111", mew=.8, zorder=4)

    ax.plot(w.n_mid, w.k, color=COL["quality"], lw=2.0, marker="s", ms=3.4,
            mfc="white", mec=COL["quality"], mew=.7, zorder=5, label="Stopping walk")
    ax.fill_between(w.n_mid, w.ci_lo, w.ci_hi, color=COL["quality"], alpha=.18,
                    lw=0, zorder=4)

    ax.plot(m.n_points, m.k, color=COL["model"], lw=1.2, ls=(0, (4, 2)),
            alpha=.8, zorder=5, label="Model-H")

    ax.set_xlim(8, 42)
    ax.set_xlabel("Point count (N)")
    ax.set_ylabel("Mean group count (K)")
    ax.set_title(LABEL[cond], fontsize=7, loc="left", color="#444444")


def panel_e(ax, slopes):
    order = ["Human", "Stopping walk", "Greedy", "Model-H"]
    n = len(order)
    yy = np.arange(n)
    colors = {"Human": "#111111", "Stopping walk": COL["quality"],
              "Greedy": COL["comparator"], "Model-H": COL["model"]}
    for y, model in enumerate(order):
        r = slopes[(slopes.model == model) & (slopes.condition == "exp4")].iloc[0]
        c = colors[model]
        if np.isfinite(r.ci_lo) and np.isfinite(r.ci_hi):
            ax.errorbar(r.slope, y, xerr=[[r.slope - r.ci_lo], [r.ci_hi - r.slope]],
                        fmt="s", ms=4.6, capsize=2, color=c, mfc="none",
                        mec=c, mew=.9, elinewidth=.8, zorder=3, lw=.8)
        else:
            ax.plot(r.slope, y, "s", ms=4.6, mfc="none", mec=c, mew=.9, color=c, zorder=3)
    ax.axvline(0, color="#888888", lw=.8, zorder=1)
    ax.set_yticks(yy, order)
    ax.set_xlim(-.035, .075)
    ax.set_xlabel("Group-number slope\n(K per point)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="s", color="w", mfc="none",
                              mec="#444444", ls="none",
                              label="Anchor/aperture (Exp 4)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=5.4, frameon=False)


def save_local(fig, stem: str) -> Path:
    for ext in ("png", "pdf", "svg"):
        try:
            fig.savefig(OUT / f"{stem}.{ext}")
        except PermissionError:
            # A PDF may be open in a viewer on Windows; continue exporting the
            # other publication formats rather than aborting the whole build.
            pass
    return OUT / f"{stem}.png"


def main() -> Path:
    matched = pd.read_csv(RESULTS / "matched_k_ideal_all.csv")
    minf = pd.read_csv(RESULTS / "matched_k_ideal_all_inference.csv")
    search = pd.read_csv(RESULTS / "funnel_search_trials.csv")
    headline = pd.read_csv(RESULTS / "headline_gap_inference.csv")
    stop_pred = pd.read_csv(RESULTS / "stopping_oof_predictions.csv")

    oos = pd.read_csv(RESULTS / "process_oos_participant_k_errors.csv")
    ablation = pd.read_csv(RESULTS / "process_ablation_slopes.csv")

    fig = plt.figure(figsize=(7.08, 6.65))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[.82, 1.18],
                              left=.105, right=.985, top=.96, bottom=.085,
                              hspace=.46)
    top = outer[0].subgridspec(1, 3, wspace=.58)
    bottom = outer[1].subgridspec(1, 3, width_ratios=[1.18, 1.18, .82], wspace=.45)

    # ---------------- a | matched-K endpoint gap ----------------
    ax = fig.add_subplot(top[0, 0])
    base = matched.groupby(["exp", "base_uuid"], as_index=False).silhouette_gap.mean()
    yy = np.arange(4)[::-1]
    colors = [COL["lasso"], COL["lasso"], COL["anchor"], COL["anchor"]]
    for i, (ypos, (e, color)) in enumerate(zip(yy, zip(brf.LABEL, colors))):
        values = base.query("exp == @e").silhouette_gap.to_numpy(float)
        raincloud_h(ax, values, ypos, color, width=.34, seed=100+i,
                    alpha=.18, summary_offset=.075)
    ax.axvline(0, color="#999999", lw=.55, ls=(0, (3, 2)))
    ax.set_yticks(yy, ["L/F", "L/A", "A/F", "A/A"])
    ax.set_xlabel("Best-found − human\nsilhouette", fontsize=6.2)
    ax.tick_params(axis="both", labelsize=5.5)
    silhouette_test = minf.query("metric == 'silhouette'").iloc[0]
    ax.text(.98, .96, fmt_p(silhouette_test.p_two_sided),
            transform=ax.transAxes, ha="right", va="top", fontsize=5.5)
    panel(ax, "a")

    # ---------------- b | search efficiency ----------------
    ax = fig.add_subplot(top[0, 1])
    g = search.groupby(["exp", "base_uuid"], as_index=False)[
        ["human_actions_all", "oracle_actions_all"]].mean()
    pair_positions = [(3, 2), (1, 0)]
    max_x = 0.0
    for pair_i, (e, (yh, yo)) in enumerate(zip(["exp2", "exp4"], pair_positions)):
        d = g.query("exp == @e").sort_values("base_uuid")
        hvals = d.human_actions_all.to_numpy(float)
        ovals = d.oracle_actions_all.to_numpy(float)
        for hv, ov in zip(hvals, ovals):
            ax.plot([hv, ov], [yh, yo], color="#A8A8A8", lw=.35,
                    alpha=.16, zorder=0)
        raincloud_h(ax, hvals, yh, COL["human"], width=.31,
                    seed=120+pair_i, alpha=.16, summary_offset=.070)
        raincloud_h(ax, ovals, yo, COL["comparator"], width=.31,
                    seed=130+pair_i, alpha=.18, summary_offset=.070)
        max_x = max(max_x, float(np.max(hvals)), float(np.max(ovals)))
        r = headline.query(
            "analysis == 'funnel_policy' and condition == @e and estimand.str.contains('actions')",
            engine="python").iloc[0]
        ax.text(.98, .80 if pair_i == 0 else .29, fmt_p(r.p_two_sided),
                transform=ax.transAxes, ha="right", va="center", fontsize=6.5)
    ax.set_yticks([3, 2, 1, 0], ["L human", "L oracle", "A human", "A oracle"])
    ax.set_xlabel("Actions to reveal all points")
    ax.set_xlim(1.5, max_x + .8)
    panel(ax, "b")

    # ---------------- c | stopping hazard ----------------
    ax = fig.add_subplot(top[0, 2])
    eps = 1e-12
    for model in ["process", "linear_quality", "threshold"]:
        p = np.clip(stop_pred[model].to_numpy(float), eps, 1 - eps)
        y = stop_pred.stop.to_numpy(float)
        stop_pred[model + "_loss"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    base_loss = stop_pred.groupby("base_uuid")[[m + "_loss" for m in
                                                ["process", "linear_quality", "threshold"]]].mean()
    order = ["process", "linear_quality", "threshold"]
    point_colors = [COL["process"], COL["quality"], COL["threshold"]]
    yy = np.arange(3)[::-1]
    for _, row in base_loss.iterrows():
        ax.plot([row[m + "_loss"] for m in order], yy,
                color="#AAAAAA", lw=.35, alpha=.14, zorder=0)
    for i, (ypos, (model, color)) in enumerate(zip(yy, zip(order, point_colors))):
        values = base_loss[model + "_loss"].to_numpy(float)
        raincloud_h(ax, values, ypos, color, width=.31, seed=150+i,
                    alpha=.14, summary_offset=.070)
    ax.set_yticks(yy, ["Process", "+ current Q", "+ threshold"])
    ax.set_xlabel("OOF log loss\n(stimulus grouped)", fontsize=6.5)
    ax.set_xlim(base_loss.min().min() - .015, base_loss.max().max() + .018)
    r1 = headline.query("analysis == 'stopping_oof' and estimand.str.contains('linear_quality over process')", engine="python").iloc[0]
    r2 = headline.query("analysis == 'stopping_oof' and estimand.str.contains('threshold over linear_quality')", engine="python").iloc[0]
    ax.text(.98, .80, fmt_p(r1.p_two_sided), transform=ax.transAxes,
            ha="right", va="center", fontsize=6.5)
    ax.text(.98, .30, fmt_p(r2.p_two_sided), transform=ax.transAxes,
            ha="right", va="center", fontsize=6.5)
    panel(ax, "c")

    # ---------------- d/e | direct OOS diagnostics ----------------
    dgrid = bottom[0, 0].subgridspec(2, 1, hspace=.62)
    panel_oos_diagnostic(fig.add_subplot(dgrid[0, 0]),
                         fig.add_subplot(dgrid[1, 0]), "exp3", oos, "d")

    egrid = bottom[0, 1].subgridspec(2, 1, hspace=.62)
    panel_oos_diagnostic(fig.add_subplot(egrid[0, 0]),
                         fig.add_subplot(egrid[1, 0]), "exp4", oos, "e")

    ax = fig.add_subplot(bottom[0, 2])
    panel_f(ax, ablation)
    panel(ax, "f")

    return save_local(fig, "figure4_search_stopping_generation")


if __name__ == "__main__":
    p = main()
    print(f"-> {p}")
