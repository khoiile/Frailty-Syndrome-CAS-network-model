import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.lines import Line2D

from model import FrailtyNetworkModel, NODES, DEFAULT_PARAMS, GROUP_PARAMS
from data import dataset, get_group_initial_state
from utils import run_ensemble

os.makedirs("outputs", exist_ok=True)

#Global font settings
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "DejaVu Sans", "Arial"],
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.titlesize":  11,
})

STEPS   = 300 #system may take longer to adapt, so more steps than emergence
N_RUNS  = 50
GROUPS  = ["robust", "pre-frail", "frail"]
GCOLS   = {"robust": "#55A868", "pre-frail": "#DD8452", "frail": "#C44E52"}


def so_1():
    """
    Start from many random initial states.
    Show system converges to one of a few stable attractors
    (robust or frail) — self-organization without external factors
    positive feedback
    .00 = 0 deficits (robust),

    0.25 = 1, 0.50 = 2, (pre)
    
    0.75 = 3, 1.00 = 4 (frail)
    """
    print("\n[SO1] Attractor convergence from random initial states")

    rng = np.random.default_rng(123)  #fixed seed=123
    n_sims = 30 #number of runs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    final_fis = []
    for i in range(n_sims):
        # Random binary initial state
        init = {n: int(rng.random() > 0.5) for n in NODES}
        m = FrailtyNetworkModel(init, seed=i)
        m.run(STEPS)
        fi = m.fi_series()
        final_fi = fi[-1]
        final_fis.append(final_fi)


        color = "#2166AC" if final_fi < 0.35 else (   # blue = robust
                "#F4A582" if final_fi < 0.65 else      # light red = pre-frail
                "#B2182B")                              # dark red = frail
        ax1.plot(fi, color=color, alpha=0.4, linewidth=1)

    ax1.axhline(0.5, color="red", linestyle="--", linewidth=1)
    legend_elements = [
        Line2D([0], [0], color="#2166AC", label="Robust (FI < 0.35)"),
        Line2D([0], [0], color="#F4A582", label="Pre-frail (0.35 ≤ FI < 0.65)"),
        Line2D([0], [0], color="#B2182B", label="Frail (FI ≥ 0.65)"),
        Line2D([0], [0], color="red", linestyle="--", label="Frail threshold (FI=0.5)"),
    ]
    ax1.legend(handles=legend_elements, fontsize=8)
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Frailty Index ")
    ax1.set_title(f"{n_sims} random initial states → convergence")
    ax1.set_ylim(-0.05, 1.05)

    DISCRETE_BINS = [-0.125, 0.125, 0.375, 0.625, 0.875, 1.125]
    FI_TICKS      = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax2.hist(final_fis, bins=DISCRETE_BINS, color="#4C72B0", alpha=0.7, edgecolor="white")
    ax2.axvline(0.5, color="red", linestyle="--", linewidth=1,
                label="Frail threshold (FI=0.5)")
    ax2.set_xticks(FI_TICKS)
    ax2.set_xticklabels([str(v) for v in FI_TICKS])
    ax2.set_xlabel("Final FI")
    ax2.set_ylabel("Count (out of 30 runs)")
    ax2.set_title("Distribution of final states\n(attractors)")
    ax2.legend(fontsize=8)

    fig.suptitle("SO1 — Attractor Convergence \n",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/SO1_attractor_convergence.png", dpi=150)
    plt.close()
    print("  Saved: outputs/SO1_attractor_convergence.png")


def so_2():
    """
    same initial state, vary repair resistance R 
    Cminus = C0/R .. : the larger R ther less repair
    low R: easy repair - converge to robust
    high R: cant repair - converge to frail
    """
    print("\n[SO2] Repair rate: different stable states")

    # Start all from pre-frail (2 damaged)
    init = {"weakness": 1, "slowness": 0,
            "low_activity": 1, "exhaustion": 0}

    R_values = [0.2, 0.35, 0.5, 0.8, 1.5, 3.0]
    palette  = plt.cm.RdYlGn(np.linspace(0.85, 0.1, len(R_values)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    final_means = []
    for R_val, c in zip(R_values, palette):
        params = dict(DEFAULT_PARAMS)
        params["R"] = R_val

        fi_runs = run_ensemble(init, params=params, steps=STEPS, n_runs=N_RUNS)
        fi_mean = fi_runs.mean(axis=0)
        final_means.append(fi_mean[-1])

        ax1.plot(fi_mean, color=c, linewidth=2,
                 label=f"R={R_val:.1f} → FI={fi_mean[-1]:.2f}")

    ax1.axhline(0.5, color="red", linestyle="--", linewidth=1)
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Mean FI")
    ax1.set_title("FI trajectories by repair resistance R")
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.05, 1.05)

    # Final FI vs R
    ax2.plot(R_values, final_means, "s-", color="#4C72B0",
             markersize=10, linewidth=2)
    ax2.axhline(0.5, color="red", linestyle="--", linewidth=1,
                label="frail threshold")
    ax2.set_xlabel("Repair resistance R")
    ax2.set_ylabel("Final mean FI")
    ax2.set_title("Self-organized stable state vs resilience")
    ax2.legend(fontsize=9)

    fig.suptitle("SO2 — Resilience Determines Attractor \n",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/SO2_resilience_states.png", dpi=150)
    plt.close()
    print("  Saved: outputs/SO2_resilience_states.png")



def so_3():
    """
    Measure variance and autocorrelation for each frailty group.

    use same intitial state 2 damaged nodes: slow and low_act
    with dif R values: R=0.25 R=0.5 R=1.5


    the more state changes and FI variance(changes/difference) the more chance to repair
    
    """
    print("\n[SO3] Critical slowing down — complexity metrics")

    init = {"weakness": 0, "slowness": 1,
            "low_activity": 1, "exhaustion": 0}
    so3_steps = 500 #more steps for clearer observation

    trans_data = {g: [] for g in GROUPS}  # transition counts  times nodes flip between 0-1
    var_data   = {g: [] for g in GROUPS}  # FI variance #latest 200finalsteps.  high var: high fluctuate amplitude, not yet settle into attractor
    mean_fi    = {g: [] for g in GROUPS}  # mean FI  #latest 200final steps avg --> rob: low FI, frail: high mean FI (meanfi use for verification)

    for group in GROUPS:
        params = GROUP_PARAMS[group]
        for seed in range(N_RUNS):
            m = FrailtyNetworkModel(init, params=params, seed=seed)
            m.run(so3_steps)
            df = m.get_history_df()
            fi = m.fi_series()

            # count total transitions (state flips) across all nodes
            transitions = 0
            for node in NODES:
                vals = df[node].values
                transitions += np.sum(np.abs(np.diff(vals)))
            trans_data[group].append(transitions)

            # use last 200 of 500 steps to measure steady-state variance
            # (skip the first 300 steps where the system is still settling).
            fi_tail = fi[-200:]
            var_data[group].append(np.var(fi_tail))
            mean_fi[group].append(np.mean(fi_tail))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: transition count
    bp1 = axes[0].boxplot([trans_data[g] for g in GROUPS],
                           positions=[0,1,2], widths=0.5, patch_artist=True)
    for patch, g in zip(bp1["boxes"], GROUPS):
        patch.set_facecolor(GCOLS[g]); patch.set_alpha(0.7)
    axes[0].set_xticks([0,1,2]); axes[0].set_xticklabels(["robust\nR=0.25", "pre-frail\nR=0.50", "frail\nR=1.50"])
    axes[0].set_ylabel("Total state transitions")
    axes[0].set_title("State transitions\n(robust = active repair; frail = rigid/stuck)")

    # Middle: FI variance
    bp2 = axes[1].boxplot([var_data[g] for g in GROUPS],
                           positions=[0,1,2], widths=0.5, patch_artist=True)
    for patch, g in zip(bp2["boxes"], GROUPS):
        patch.set_facecolor(GCOLS[g]); patch.set_alpha(0.7)
    axes[1].set_xticks([0,1,2]); axes[1].set_xticklabels(["robust\nR=0.25", "pre-frail\nR=0.50", "frail\nR=1.50"])
    axes[1].set_ylabel("Variance of FI")
    axes[1].set_title("FI variance\n(frail = low variance = loss of complexity")

    # Right: mean FI for verification
    bp3 = axes[2].boxplot([mean_fi[g] for g in GROUPS],
                           positions=[0,1,2], widths=0.5, patch_artist=True)
    for patch, g in zip(bp3["boxes"], GROUPS):
        patch.set_facecolor(GCOLS[g]); patch.set_alpha(0.7)
    axes[2].set_xticks([0,1,2]); axes[2].set_xticklabels(["robust\nR=0.25", "pre-frail\nR=0.50", "frail\nR=1.50"])
    axes[2].set_ylabel("Mean FI")
    axes[2].set_title("Mean frailty level\n(confirms group separation)")
    axes[2].axhline(0.5, color="red", linestyle="--", linewidth=1,
                    label="Frail threshold (FI=0.5)")


    median_proxy = Line2D([0], [0], color="orange", linewidth=2, label="Median")
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [median_proxy],
                  labels=labels + ["Median"],
                  fontsize=8, loc="upper right")

    fig.suptitle("SO3 — Complexity Metrics\n",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/SO3_complexity_metrics.png", dpi=150)
    plt.close()
    print("  Saved: outputs/SO3_complexity_metrics.png")

    # Print summary
    print("\n  Summary (mean ± std):")
    print(f"  {'Group':10s}  {'Transitions':>14s}  {'FI Variance':>14s}  {'Mean FI':>10s}")
    for g in GROUPS:
        t_m = np.mean(trans_data[g]); t_s = np.std(trans_data[g])
        v_m = np.mean(var_data[g]);   v_s = np.std(var_data[g])
        f_m = np.mean(mean_fi[g]);    f_s = np.std(mean_fi[g])
        print(f"  {g:10s}  {t_m:8.1f}±{t_s:.1f}  {v_m:8.4f}±{v_s:.4f}  {f_m:6.3f}±{f_s:.3f}")



# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  SELF-ORGANIZATION EXPERIMENTS")
    print("=" * 55)
    so_1()
    so_2()
    so_3()
    print("\nDone. See outputs/")
