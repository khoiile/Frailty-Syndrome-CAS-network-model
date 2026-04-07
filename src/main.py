"""
Usage:
    python main.py                  # run everything
    python main.py --only emergence
    python main.py --only so
    python main.py --only demo
"""

import argparse
import os
import numpy as np

os.makedirs("outputs", exist_ok=True)

from data import dataset, print_dataset_summary, get_group_initial_state
from model import FrailtyNetworkModel, NODES, DEFAULT_PARAMS
from emergence import emergence_1, emergence_2, emergence_3
from self_organization import so_1, so_2, so_3


def run_demo():
    """Quick demo: run 3 scenarios (robust, pre-frail, frail) from dataset."""
    from model import GROUP_PARAMS, GROUP_INIT
    print("\n── Demo: 3 Frailty Scenarios ──")
    # Fixed seeds so the demo plots look the same every run (reproducibility).
    demo_seeds = {"robust": 13, "pre-frail": 3, "frail": 1}
    for group in ["robust", "pre-frail", "frail"]:
        init   = GROUP_INIT[group]
        params = GROUP_PARAMS[group]
        seed   = demo_seeds[group]
        print(f"\n  {group}: initial = {init}, R = {params['R']}")

        m = FrailtyNetworkModel(init, params=params, seed=seed)
        m.run(200)

        print(f"    Final state: {m.state}")
        print(f"    Final FI:    {m.frailty_index():.2f}")
        print(f"    Final label: {m.frailty_label()}")

        m.plot_history(
            title=f"Scenario: {group}  (R={params['R']})  |  "
                  f"FI={m.frailty_index():.2f}  ({m.frailty_label()})",
            save_path=f"outputs/demo_{group.replace('-','_')}.png"
        )

    # Network structure plot — show frail state for visibility
    m_net = FrailtyNetworkModel(GROUP_INIT["pre-frail"])

    #customize the network structure plot
    #m_net = FrailtyNetworkModel({"weakness": 1, "slowness":1,"low_activity": 1,"exhaustion": 0})

    m_net.plot_network(save_path="outputs/network_structure.png")


def main():
    parser = argparse.ArgumentParser(
        description="Frailty CAS Model")
    parser.add_argument("--only",
                        choices=["demo", "emergence", "so", "all"],
                        default="all",
                        help="Which section to run")
    args = parser.parse_args()

    print("FRAILTY SYSTEM MODEL")
    print(f"  Nodes: {NODES}")
    print(f"  Params: C₀={DEFAULT_PARAMS['C0']}, "
          f"c⁺={DEFAULT_PARAMS['c_plus']}, "
          f"c⁻={DEFAULT_PARAMS['c_minus']}, "
          f"R={DEFAULT_PARAMS['R']}")
    print("=" * 60)

    # Dataset summary
    print("\n[1] Dataset")
    print_dataset_summary(dataset)

    # Each block runs when its section is requested, or when --only is "all"
    # (the default), meaning all three sections run if no flag is given.
    if args.only in ("demo", "all"):
        print("\n[2] Demo Scenarios")
        run_demo()

    if args.only in ("emergence", "all"):
        print("\n[3] Emergence Experiments")
        emergence_1()
        emergence_2()
        emergence_3()

    if args.only in ("so", "all"):
        print("\n[4] Self-Organization Experiments")
        so_1()
        so_2()
        so_3()

    print("\n" + "=" * 60)
    print("  Done. All outputs saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
