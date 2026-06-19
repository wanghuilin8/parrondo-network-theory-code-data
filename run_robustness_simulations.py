"""
Deterministic simulations for the topological robustness checks.

This script generates Fig7_additional_robustness.png and CSV files used
for robustness parameters, finite-size checks, and reproducible
Monte Carlo settings.
"""
import csv
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import brentq


OUT_FIG = "figures"
OUT_DATA = "data"
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_DATA, exist_ok=True)

DEFAULT = dict(p1=0.095, p2=0.745, tau=0.03)
BASE_SEED = 20260406


@dataclass(frozen=True)
class CurveSpec:
    label: str
    network: str
    n: int
    k: int | None = None
    pr: float | None = None
    m: int | None = None


def solve_mf(gamma, p1, p2, tau):
    def w_a(a, b):
        if b == (a + 1) % 3:
            return 0.5
        if b == (a - 1) % 3:
            return 0.5 * (1 - tau)
        return 0.0

    def w_b(a, b):
        if a == 0:
            return p1 if b == 1 else (1 - p1 if b == 2 else 0.0)
        if a == 1:
            return (1 - p2) if b == 0 else (p2 if b == 2 else 0.0)
        return p2 if b == 0 else (1 - p2 if b == 1 else 0.0)

    w = np.zeros((3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            if a != b:
                w[a, b] = gamma * w_a(a, b) + (1 - gamma) * w_b(a, b)
        w[a, a] = -w[a, :].sum()
    mat = w.T.copy()
    mat[-1, :] = 1
    rhs = np.zeros(3)
    rhs[-1] = 1
    return np.linalg.solve(mat, rhs)


def f_mix_mf(gamma, p1, p2, tau):
    pi = solve_mf(gamma, p1, p2, tau)
    return gamma * (-tau) + (1 - gamma) * (
        (2 * p2 - 1) - 2 * (p2 - p1) * pi[0]
    )


def mf_window(p1, p2, tau):
    gammas = np.linspace(0.0, 1.0, 1001)
    vals = np.array([f_mix_mf(g, p1, p2, tau) for g in gammas])
    roots = []
    for i in range(len(gammas) - 1):
        if vals[i] == 0:
            roots.append(gammas[i])
        elif vals[i] * vals[i + 1] < 0:
            roots.append(
                brentq(lambda x: f_mix_mf(x, p1, p2, tau), gammas[i], gammas[i + 1])
            )
    return roots, vals.max()


def build_neighbors(spec: CurveSpec, seed: int):
    if spec.network == "ring":
        half = spec.k // 2
        return [
            [(i + d) % spec.n for d in range(-half, half + 1) if d != 0]
            for i in range(spec.n)
        ]
    if spec.network == "ws":
        graph = nx.watts_strogatz_graph(spec.n, spec.k, spec.pr, seed=seed)
        return [list(graph.neighbors(i)) for i in range(spec.n)]
    if spec.network == "ba":
        graph = nx.barabasi_albert_graph(spec.n, spec.m, seed=seed)
        return [list(graph.neighbors(i)) for i in range(spec.n)]
    raise ValueError(f"unknown network type: {spec.network}")


def mc_run(nbrs, gamma, p1, p2, tau, steps_per_node, realizations, rng):
    n = len(nbrs)
    steps = int(steps_per_node * n)
    max_deg = max(len(nb) for nb in nbrs)
    nb_arr = np.zeros((n, max_deg), dtype=np.int32)
    deg_arr = np.array([len(nb) for nb in nbrs], dtype=np.int32)
    for i, nb in enumerate(nbrs):
        nb_arr[i, : len(nb)] = nb

    cap_sum = 0.0
    for _ in range(realizations):
        cap = np.zeros(n, dtype=np.float64)
        nodes = rng.integers(0, n, size=steps)
        game_rands = rng.random(steps)
        contest_rands = rng.random(steps)
        penalty_rands = rng.random(steps)
        play_rands = rng.random(steps)
        nb_rands = rng.integers(0, max_deg, size=steps)

        for t in range(steps):
            i = nodes[t]
            deg = deg_arr[i]
            if deg == 0:
                continue
            if game_rands[t] < gamma:
                j = nb_arr[i, nb_rands[t] % deg]
                if contest_rands[t] < 0.5:
                    cap[i] += 1
                    cap[j] -= 1
                else:
                    cap[i] -= 1
                    cap[j] += 1
                if penalty_rands[t] < tau:
                    cap[i] -= 1
            else:
                state = int(cap[i]) % 3
                p_win = p1 if state == 0 else p2
                cap[i] += 1 if play_rands[t] < p_win else -1
        cap_sum += cap.sum()
    return cap_sum / (n * realizations * steps)


def simulate_curve(spec, gammas, params, reps=3, k_per_rep=20, steps_per_node=50):
    rows = []
    means = []
    sems = []
    for gamma in gammas:
        vals = []
        for rep in range(reps):
            graph_seed = BASE_SEED + 1000 * rep + spec.n + int(gamma * 1000)
            rng_seed = BASE_SEED + 100000 * rep + spec.n * 7 + int(gamma * 10000)
            nbrs = build_neighbors(spec, graph_seed)
            rng = np.random.default_rng(rng_seed)
            val = mc_run(
                nbrs,
                gamma,
                params["p1"],
                params["p2"],
                params["tau"],
                steps_per_node,
                k_per_rep,
                rng,
            )
            vals.append(val)
            rows.append(
                dict(
                    label=spec.label,
                    network=spec.network,
                    n=spec.n,
                    k=spec.k if spec.k is not None else "",
                    pr=spec.pr if spec.pr is not None else "",
                    m=spec.m if spec.m is not None else "",
                    gamma=gamma,
                    rep=rep,
                    fitness=val,
                    p1=params["p1"],
                    p2=params["p2"],
                    tau=params["tau"],
                )
            )
        vals = np.array(vals)
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(sems), rows


def write_csv(path, rows):
    fieldnames = [
        "label",
        "network",
        "n",
        "k",
        "pr",
        "m",
        "gamma",
        "rep",
        "fitness",
        "p1",
        "p2",
        "tau",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def positive_window(gammas, means):
    idx = np.flatnonzero(means > 0)
    if len(idx) == 0:
        return None
    return float(gammas[idx[0]]), float(gammas[idx[-1]])


def fmt_pair(values):
    if values is None:
        return "None"
    return f"({values[0]:.6g}, {values[1]:.6g})"


def fmt_list(values):
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"


def main():
    stress_specs = [
        CurveSpec("WS k=6, p_r=1", "ws", 500, k=6, pr=1.0),
        CurveSpec("BA m=3", "ba", 500, m=3),
        CurveSpec("BA m=5", "ba", 500, m=5),
    ]
    finite_specs = [
        CurveSpec("BA m=5, N=500", "ba", 500, m=5),
        CurveSpec("BA m=5, N=1000", "ba", 1000, m=5),
        CurveSpec("BA m=5, N=2000", "ba", 2000, m=5),
    ]
    panels = [
        (
            "low_p1",
            dict(p1=0.020, p2=0.850, tau=0.010),
            np.linspace(0.02, 0.98, 17),
            stress_specs,
            r"$p_1=0.020,\;p_2=0.850,\;\tau=0.010$",
        ),
        (
            "near_critical",
            dict(p1=0.095, p2=0.745, tau=0.050),
            np.linspace(0.08, 0.48, 17),
            stress_specs,
            r"$p_1=0.095,\;p_2=0.745,\;\tau=0.050$",
        ),
        (
            "finite_size",
            DEFAULT,
            np.linspace(0.05, 0.90, 14),
            finite_specs,
            r"default parameters, BA $m=5$",
        ),
    ]

    colors = ["#1f77b4", "#e67e22", "#2ca02c"]
    markers = ["o", "s", "^"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.1))
    all_rows = []
    summary = []

    for ax, (panel_name, params, gammas, specs, title) in zip(axes, panels):
        roots, peak = mf_window(**params)
        if len(roots) >= 2:
            ax.axvspan(roots[0], roots[1], color="#2ecc71", alpha=0.10)
            ax.axvline(roots[0], color="0.55", ls=":", lw=1.5)
            ax.axvline(roots[1], color="0.55", ls=":", lw=1.5)
        for spec, color, marker in zip(specs, colors, markers):
            means, sems, rows = simulate_curve(spec, gammas, params)
            all_rows.extend(rows)
            window = positive_window(gammas, means)
            summary.append(
                {
                    "panel": panel_name,
                    "label": spec.label,
                    "mc_window": window,
                    "mf_roots": roots,
                    "mf_peak": peak,
                }
            )
            if panel_name == "finite_size":
                ax.plot(
                    gammas,
                    (means > 0).astype(float),
                    color=color,
                    marker=marker,
                    ms=4.8,
                    lw=2.0,
                    label=spec.label,
                )
            else:
                ax.errorbar(
                    gammas,
                    means * 1e4,
                    yerr=sems * 1e4,
                    color=color,
                    marker=marker,
                    ms=4.8,
                    lw=2.0,
                    capsize=2.5,
                    label=spec.label,
                )
        ax.axhline(0, color="k", ls="--", lw=1.2)
        if panel_name == "finite_size":
            ax.axhline(1, color="0.6", ls="--", lw=1.0)
            ax.set_ylim(-0.12, 1.18)
            ax.set_ylabel("Paradox indicator", fontsize=14)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(r"$\gamma$", fontsize=14)
        ax.grid(True, alpha=0.28)
        ax.tick_params(labelsize=12)
    axes[0].set_ylabel(r"Average fitness $(\times 10^{-4})$", fontsize=14)
    axes[0].text(0.03, 0.95, "(a)", transform=axes[0].transAxes, fontsize=15, fontweight="bold", va="top")
    axes[1].text(0.03, 0.95, "(b)", transform=axes[1].transAxes, fontsize=15, fontweight="bold", va="top")
    axes[2].text(0.03, 0.95, "(c)", transform=axes[2].transAxes, fontsize=15, fontweight="bold", va="top")
    axes[0].legend(fontsize=10.5, framealpha=0.92, loc="lower left")
    axes[1].legend(fontsize=10.5, framealpha=0.92, loc="upper right")
    axes[2].legend(fontsize=10.5, framealpha=0.92, loc="lower left")
    fig.tight_layout()
    out_path = os.path.join(OUT_FIG, "Fig7_additional_robustness.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    write_csv(os.path.join(OUT_DATA, "additional_robustness_curves.csv"), all_rows)
    with open(os.path.join(OUT_DATA, "additional_robustness_summary.txt"), "w", encoding="utf-8") as f:
        for item in summary:
            f.write(
                f"{item['panel']} | {item['label']} | MC window {fmt_pair(item['mc_window'])} | "
                f"MF roots {fmt_list(item['mf_roots'])} | MF peak {item['mf_peak']:.6g}\n"
            )
            print(
                f"{item['panel']} | {item['label']} | MC window {fmt_pair(item['mc_window'])} | "
                f"MF roots {fmt_list(item['mf_roots'])} | MF peak {item['mf_peak']:.6g}"
            )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
