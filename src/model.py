import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


'''
INITIALIZE
'''
NODES = ["weakness", "slowness", "low_activity", "exhaustion"]

EDGES = [
    ("weakness",     "slowness"),
    ("weakness",     "exhaustion"),
    ("slowness",     "low_activity"),
    ("slowness",     "exhaustion"),
    ("low_activity", "weakness"),
    ("low_activity", "exhaustion"),
    ("exhaustion",   "low_activity"),
    ("exhaustion",   "weakness"),
]

DEFAULT_PARAMS = {
    "C0":      0.008,    # base damage probability per step
    "c_plus":  2.0,      # neighbor coupling for damage
    "c_minus": 0.3,      # neighbor coupling for repair
    "R":       0.5,      # repair resistance
}

GROUP_PARAMS = {
    "robust":    {**DEFAULT_PARAMS, "R": 0.25},
    "pre-frail": {**DEFAULT_PARAMS, "R": 0.50},
    "frail":     {**DEFAULT_PARAMS, "R": 1.50},
}

# Group-specific initial states. #by Fried, weak&exhaus tend to have highest in-degree, lead to frail easy
GROUP_INIT = {
    "robust":    {"weakness": 0, "slowness": 0, "low_activity": 0, "exhaustion": 0},
    "pre-frail": {"weakness": 1, "slowness": 0, "low_activity": 0, "exhaustion": 1}, 
    "frail":     {"weakness": 1, "slowness": 1, "low_activity": 1, "exhaustion": 1},
}

'''
MODEL BUILDING
'''
class FrailtyNetworkModel:

    #initialize the model with edged,nodes, etc
    def __init__(self, initial_states, params=None, seed=42): #fixed seed to ensure the reproducibility of stochastic outcomes

        self.nodes  = list(NODES)
        self.params = params if params else dict(DEFAULT_PARAMS)
        self.rng    = np.random.default_rng(seed)

        # Initialize binary states
        self.state = {n: int(initial_states.get(n, 0)) for n in self.nodes}

        # History: list of dicts, one per time step
        self.history = [{n: self.state[n] for n in self.nodes}]
        self.neighbors = {n: [] for n in self.nodes}
        for src, tgt in EDGES:   #src: source, starting point; tgt: target, end point of the edges
            self.neighbors[tgt].append(src)

        # connected_count = in-degree of each node (number of incoming neighbors).
        self.connected_count = {n: len(self.neighbors[n]) for n in self.nodes}

    #local frailty fi = sigma (dj/ki)
    def local_frailty(self, node):
        neighbors = self.neighbors[node]
        if len(neighbors) == 0:
            return 0.0
        damaged_count = sum(self.state[nb] for nb in neighbors)  #number of damaged neighbors/total neighbors
        return damaged_count / len(neighbors)

    #c plus
    def damage_rate(self, node):
        fi = self.local_frailty(node)
        C0 = self.params["C0"]
        cp = self.params["c_plus"]
        rate = C0 * np.exp(cp * fi)
        return min(rate, 1.0)  # cap at probability 1

    #c minus
    def repair_rate(self, node):
        fi = self.local_frailty(node)
        C0 = self.params["C0"]
        cm = self.params["c_minus"]
        R  = self.params["R"]
        rate = (C0 / R) * np.exp(-cm * fi)
        return min(rate, 1.0)  # cap at probability 1 (exp() can exceed 1 at very low f_i)

    #frailty index FI, main para to detect, classify the final state of the system
    #fi= sigma (di/n) 
    def frailty_index(self):
        return sum(self.state[n] for n in self.nodes) / len(self.nodes) 

    #labeling - classify the state
    def frailty_label(self):
        """
        Classify based on the phenotype:
          0 criteria damaged : robust
          1-2 criteria       : pre-frail
          3+ criteria        : frail
        """
        count = sum(self.state[n] for n in self.nodes) 
        if count == 0:
            return "robust"
        elif count <= 2:
            return "pre-frail"
        else:
            return "frail"

    #step for checking the affections between nodes
    def step(self):
        """
        with proba cplus and cminus, each step the sys will check
        if the nodes have enough rate to transit from 0 to 1 and vice versa
        """
        new_state = {}
        for node in self.nodes:
            if self.state[node] == 0:   
                prob = self.damage_rate(node)       #from healthy 0 to be damaged 1
                new_state[node] = 1 if self.rng.random() < prob else 0
            else:
                prob = self.repair_rate(node)       #recover from damage
                new_state[node] = 0 if self.rng.random() < prob else 1

        self.state = new_state      #update system wether the nodes change their state
        self.history.append({n: self.state[n] for n in self.nodes})


    def run(self, steps=200):
        """Run simulation for given number of steps."""
        for _ in range(steps):
            self.step()


    def get_history_df(self):
        """Return history as DataFrame (rows=time, cols=nodes)."""
        return pd.DataFrame(self.history)

    def fi_series(self):
        """Return FI at each time step."""
        df = self.get_history_df()
        return df.mean(axis=1).values

    def variance(self, window=50):
        """
        Variance of node states over the last `window` steps.
        Frail systems show higher variance (instability).

        Source: Gijzel et al. 2017, Results — Table 2 shows frail group
          has significantly higher variance of self-rated health (p < 0.001).
        """
        df = self.get_history_df().tail(window)
        return df.var().to_dict()

    def autocorrelation(self, lag=1, window=50):  
        """ 
        #begining steps are not yet calibrated, still not static enough to show self org
        #lag: delay steps to compare. lag=1: compare just 1 step before the current step
        #window: use to capture the latest steps
        #autocor kinda like a statistical func, to define the self adaptation
        self org occur lately, ending stages -> convergence to attractor
        autocorr tracks the result of self org
        """
        fi = self.fi_series()[-window:]
        series = pd.Series(fi)
        return series.autocorr(lag=lag)

    #Visualization

    def plot_network(self, save_path="network_structure.png"):
        """Plot the network structure with node states."""
        G = nx.DiGraph()
        G.add_nodes_from(NODES)
        G.add_edges_from(EDGES)

        pos = nx.circular_layout(G)
        pos = {n: (x * 1.4, y * 0.8) for n, (x, y) in pos.items()}
        colors = ["#C44E52" if self.state[n] == 1 else "#55A868" for n in NODES]

        fig, ax = plt.subplots(figsize=(7, 4))
        nx.draw_networkx_nodes(G, pos, node_size=3000,
                            node_color=colors, alpha=0.9, ax=ax)
        nx.draw_networkx_labels(G, pos,
                                labels={n: n.replace("_", "\n") for n in NODES},
                                font_size=11, font_color="black",
                                font_weight="bold", ax=ax)
        nx.draw_networkx_edges(G, pos, width=2,
                            edge_color="#000000", arrows=True,
                            arrowsize=20,
                            min_source_margin=38,
                            min_target_margin=38,
                            connectionstyle="arc3,rad=0.12", ax=ax)

        # Degree annotations
        for n, (x, y) in pos.items():
            ax.annotate(f"k={self.connected_count[n]}", (x, y - 0.21),
                        ha="center", fontsize=9, color="#000000")
        ax.margins(0.15)

        ax.set_title("Frailty Network\n"
                     "Green=healthy, Red=damaged",
                     fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved: {save_path}")

    def plot_history(self, title="", save_path="timeseries.png"):
        """Plot time series of node states and FI."""
        df = self.get_history_df()
        fi = self.fi_series()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        sharex=True)
        colors = {"weakness": "#4C72B0", "slowness": "#DD8452",
                  "low_activity": "#55A868", "exhaustion": "#C44E52"}

        # Top: individual node states (jittered for visibility)
        for i, (node, c) in enumerate(colors.items()):
            offset = i * 0.06  # small vertical offset per node
            ax1.fill_between(range(len(df)), offset, df[node] + offset,
                             alpha=0.3, color=c, step="post")
            ax1.step(range(len(df)), df[node] + offset,
                     color=c, linewidth=1.5, where="post",
                     label=node.replace("_", " "))
        ax1.set_ylabel("Node state (0=healthy, 1=damaged)")
        ax1.set_yticks([0, 1])
        ax1.legend(fontsize=8, loc="upper right")
        ax1.set_title(title)

        # Bottom: FI over time
        ax2.plot(fi, color="black", linewidth=2)
        ax2.axhline(0.5, color="red", linestyle="--", linewidth=1,
                     label="frail threshold (≥3 deficits)")
        ax2.set_ylabel("FI")
        ax2.set_xlabel("Time step")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved: {save_path}")
