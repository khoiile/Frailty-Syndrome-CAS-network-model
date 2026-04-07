import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from itertools import combinations
from matplotlib.lines import Line2D

from model import FrailtyNetworkModel, NODES, EDGES, DEFAULT_PARAMS
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

STEPS   = 200
N_RUNS  = 50   # Monte Carlo runs for averaging (stochastic model) from garcia, rockwd&mitnit
COLORS  = {"weakness": "#4C72B0", "slowness": "#DD8452",
           "low_activity": "#55A868", "exhaustion": "#C44E52"}

#cascade
def emergence_1():
    """
    start with 1 random damaged node, observe how c+ c- draw the system to
    """
    print("\n[E1] Cascade Propagation — single node perturbation")

    scenarios = {}
    for node in NODES:
        init = {n: 0 for n in NODES}
        init[node] = 1  # only this node damaged
        scenarios[node] = init
    scenarios["none (control)"] = {n: 0 for n in NODES}

    # One color per starting node; control is gray dashed
    SCENARIO_COLORS = {
        "weakness":       "#4C72B0",
        "slowness":       "#DD8452",
        "low_activity":   "#55A868",
        "exhaustion":     "#C44E52",
        "none (control)": "#888888",
    }

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

    for label, init in scenarios.items():
        fi_runs = run_ensemble(init, steps=STEPS, n_runs=N_RUNS)
        fi_mean = fi_runs.mean(axis=0)
        color = SCENARIO_COLORS[label]
        ls = "--" if label == "none (control)" else "-"
        ax.plot(fi_mean, color=color, linewidth=2, linestyle=ls,
                label=f"{label}  (FI_final={fi_mean[-1]:.2f})")

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1,
               label="Frail threshold (FI=0.5)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Frailty Index")
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("E1 — Cascade Propagation",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/E1_cascade_propagation.png", dpi=200)
    plt.close()
    print("  Saved: outputs/E1_cascade_propagation.png")


def emergence_2():
    """
    initialize the damaged node with numbers: 0 1 2 3 4
    observe the nonlinearity of the systems when increasing the number of dmg nodes

    50 runs x 200 steps 
    and taking the final value FI_final of 50 runs in the last step
    <=> 50 FI values

    the number of runs depends on number of dmg nodes
    4C0 = 1 -> 1 x 50 runs = 50runs
    4C1 = 4 = 200 runs
    4C2=6 = 300 runs
    4C3=4=200
    4C4=1=50
    """
    print("\n[E2] Nonlinear Threshold — initial damage count")

    results = {}  

    for n_damaged in range(5):  # 0, 1, 2, 3, 4
        fi_finals = []
        for combo in combinations(NODES, n_damaged):
            init = {n: (1 if n in combo else 0) for n in NODES}
            fi_runs = run_ensemble(init, steps=STEPS, n_runs=30)
            fi_finals.extend(fi_runs[:, -1].tolist())

        results[n_damaged] = fi_finals

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: stacked bar — shows proportion of runs landing at each FI attractor
    positions = list(results.keys())
    fi_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    fi_colors = ["#2166AC", "#92C5DE", "#221A1A", "#F4A582", "#B2182B"]
    bar_width = 0.6

    for pos, k in enumerate(positions):
        vals = results[k]
        # Count how many runs ended at each discrete FI value
        counts = [sum(abs(v - fv) < 0.01 for v in vals) for fv in fi_values]
        bottom = 0
        for count, color, fv in zip(counts, fi_colors, fi_values):
            ax1.bar(pos, count, bottom=bottom, color=color,
                    width=bar_width, alpha=0.85,
                    label=f"FI={fv}" if pos == 0 else "")
            bottom += count

    ax1.set_xticks(range(5))
    ax1.set_xticklabels([str(i) for i in range(5)])
    ax1.set_xlabel("Number of initially damaged nodes")
    ax1.set_ylabel("Number of runs")
    ax1.set_title("Final FI distribution per initial damage count\n(stacked = proportion at each attractor)")
    ax1.legend(title="Final FI", fontsize=8, loc="upper right")

    # Right: mean FI curve showing nonlinearity
    means = [np.mean(results[k]) for k in positions]
    ax2.plot(positions, means, "o-", color="#4C72B0",
             linewidth=2.5, markersize=10)
    ax2.axhline(0.5, color="red", linestyle="--", linewidth=1,
                label="frail threshold")
    # Linear reference: a straight line from the 0-damage mean to the 4-damage mean.
    # If the curve rises above this line it is super-linear (threshold effect).
    ax2.plot(positions, [means[0] + (means[-1]-means[0])*i/4 for i in range(5)],
             "--", color="gray", linewidth=1, label="linear reference")
    ax2.set_xlabel("Number of initially damaged nodes")
    ax2.set_ylabel("Mean final FI")
    ax2.set_title("Nonlinear emergence threshold")
    ax2.legend(fontsize=9)

    fig.suptitle("E2 — Nonlinear Threshold \n",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/E2_nonlinear_threshold.png", dpi=150)
    plt.close()
    print("  Saved: outputs/E2_nonlinear_threshold.png")




def emergence_3():
    """
    Compare network model vs isolated nodes.
    Connected: normal model with edges (damage propagates)
    Isolated: same model but c_plus=0 (no neighbour influence)

    Shows frailty ONLY emerges with network interactions.

  
    2 case: iso & conn
    each case 50runs and y-axis of the graph shows run counts
    total cases spread out of 5 values of FI 0-1.0, in sum up = 50

    """
    print("\n[E3] Network vs Isolated")

    init = {n: 1 for n in NODES[:2]}  # 2 nodes damaged
    init.update({n: 0 for n in NODES[2:]})

    # Connected: normal parameters (damage propagates through edges)
    fi_connected = run_ensemble(init, steps=STEPS, n_runs=N_RUNS)

    # Isolated: c_plus=0 removes neighbour coupling, so each node
    # transitions only at the baseline rate C0 with no cascade effect.
    isolated_params = dict(DEFAULT_PARAMS)
    isolated_params["c_plus"] = 0.0
    fi_isolated = run_ensemble(init, params=isolated_params, steps=STEPS, n_runs=N_RUNS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mean FI lines only (no shading)
    for fi_runs, label, color in [
        (fi_connected, "Connected (network)", "#C44E52"),
        (fi_isolated,  "Isolated (no coupling)", "#55A868"),
    ]:
        fi_mean = fi_runs.mean(axis=0)
        ax1.plot(fi_mean, color=color, linewidth=2, label=label)

    ax1.axhline(0.5, color="red", linestyle="--", linewidth=1,
                label="Frail threshold (FI=0.5)")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Frailty Index ")
    ax1.set_title("FI trajectories: connected vs isolated")
    ax1.legend(fontsize=9)
    ax1.set_ylim(-0.05, 1.05)

    # Right: count of runs per FI value (5 discrete values: 0, 0.25, 0.5, 0.75, 1.0).
    # Bin edges centered on each FI value; edgecolor + integer y-axis matches SO1 style.
    DISCRETE_BINS = [-0.125, 0.125, 0.375, 0.625, 0.875, 1.125]
    FI_TICKS      = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax2.hist(fi_connected[:, -1], bins=DISCRETE_BINS, alpha=0.6,
             color="#C44E52", label="Connected", edgecolor="white", density=False)
    ax2.hist(fi_isolated[:, -1], bins=DISCRETE_BINS, alpha=0.6,
             color="#55A868", label="Isolated", edgecolor="white", density=False)
    ax2.axvline(0.5, color="red", linestyle="--", linewidth=1,
                label="Frail threshold (FI=0.5)")
    ax2.set_xticks(FI_TICKS)
    ax2.set_xticklabels([str(v) for v in FI_TICKS])
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.set_xlabel("Final FI")
    ax2.set_ylabel("Number of runs (out of 50)")
    ax2.set_title("Final FI distribution")
    ax2.legend(fontsize=9)

    fig.suptitle("E3 — Network vs Isolated \n"
                 "Frailty emerges only with network interactions ",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/E3_network_vs_isolated.png", dpi=150)
    plt.close()
    print("  Saved: outputs/E3_network_vs_isolated.png")


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  EMERGENCE EXPERIMENTS")
    print("=" * 55)
    emergence_1()
    emergence_2()
    emergence_3()
    print("\nDone. See outputs/")
