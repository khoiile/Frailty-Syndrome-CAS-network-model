"""
Helper functions shared by emergence.py and self_organization.py.

for running the simulation multiple times, store data for ploting
"""

import numpy as np
from model import FrailtyNetworkModel


def run_ensemble(initial_states, steps, n_runs, params=None):

    fi_runs = []
    for seed in range(n_runs):
        m = FrailtyNetworkModel(initial_states, params=params, seed=seed)
        m.run(steps)
        fi_runs.append(m.fi_series())
    return np.array(fi_runs)
