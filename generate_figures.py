"""
Publication-style figure generation for the Parrondo network study.

The script keeps the scientific content unchanged and improves figure
readability: clearer schematic layout, consistent fonts, color-blind
friendly palette, lighter legends, and reusable cached Monte Carlo data.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch
from scipy.ndimage import uniform_filter1d
from scipy.optimize import brentq, minimize_scalar


FIG_DIR = Path("figures")
DATA_DIR = Path("data")
CACHE_PATH = DATA_DIR / "figure_data_cache.npz"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

P1 = 0.095
P2 = 0.745
TAU = 0.03
PI0_STAR = (2 * P2 - 1) / (2 * (P2 - P1))
N_MC = 500
T_MC = 25000
BURN_PER_NODE = 100
MEASURE_PER_NODE = 100
CACHE_VERSION = 23
BASE_SEED = 20260603

COL = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#785EF0",
    "sky": "#56B4E9",
    "black": "#222222",
    "gray": "#6B7280",
    "light_green": "#E6F4EA",
    "light_red": "#FDECEC",
    "light_orange": "#FFF6E5",
    "light_purple": "#F4EEFF",
}
STATE = {0: COL["red"], 1: COL["green"], 2: COL["blue"]}


def set_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": 14,
            "axes.titlesize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10.8,
            "axes.linewidth": 0.9,
            "grid.color": "#B8B8B8",
            "grid.alpha": 0.25,
            "savefig.dpi": 300,
        }
    )


def panel_label(ax, label, x=-0.08, y=1.04, ha="left", va="bottom"):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=15.5,
        fontweight="bold",
        color=COL["black"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.2),
        clip_on=False,
        zorder=50,
    )


def finish_axes(ax, grid=True):
    if grid:
        ax.grid(True, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, name):
    fig.savefig(FIG_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def solve_mf(gamma, p1=P1, p2=P2, tau=TAU, lam=1.0):
    """Complete one-node stationary distribution including opponent diffusion."""
    plus = gamma * (1 + lam) / 2.0
    minus = gamma * (1 + lam - tau) / 2.0
    w = np.zeros((3, 3), dtype=float)
    rates = {
        (0, 1): plus + (1 - gamma) * p1,
        (0, 2): minus + (1 - gamma) * (1 - p1),
        (1, 2): plus + (1 - gamma) * p2,
        (1, 0): minus + (1 - gamma) * (1 - p2),
        (2, 0): plus + (1 - gamma) * p2,
        (2, 1): minus + (1 - gamma) * (1 - p2),
    }
    for (a, b), rate in rates.items():
        w[a, b] = rate
    for a in range(3):
        w[a, a] = -w[a, :].sum()
    mat = w.T.copy()
    mat[-1, :] = 1
    rhs = np.zeros(3)
    rhs[-1] = 1
    return np.linalg.solve(mat, rhs)


def f_mix(gamma, p1=P1, p2=P2, tau=TAU, lam=1.0):
    pi = solve_mf(gamma, p1, p2, tau, lam=lam)
    return gamma * (-tau) + (1 - gamma) * (
        (2 * p2 - 1) - 2 * (p2 - p1) * pi[0]
    )


def lambda_values(nbrs):
    deg = np.array([len(nb) for nb in nbrs], dtype=float)
    lam = np.zeros(len(nbrs), dtype=float)
    for i, nb in enumerate(nbrs):
        lam[i] = sum(1.0 / deg[j] for j in nb if deg[j] > 0)
    return lam


def graph_theory_pi0(gamma, nbrs, p1=P1, p2=P2, tau=TAU):
    lam = lambda_values(nbrs)
    return np.mean([solve_mf(gamma, p1, p2, tau, l)[0] for l in lam])


def graph_theory_f(gamma, nbrs, p1=P1, p2=P2, tau=TAU):
    pi0_bar = graph_theory_pi0(gamma, nbrs, p1, p2, tau)
    return gamma * (-tau) + (1 - gamma) * (
        (2 * p2 - 1) - 2 * (p2 - p1) * pi0_bar
    )


def build_ring(n, k):
    half = k // 2
    return [[(i + d) % n for d in range(-half, half + 1) if d != 0] for i in range(n)]


def build_ws(n, k, pr, seed):
    graph = nx.watts_strogatz_graph(n, k, pr, seed=seed)
    return [list(graph.neighbors(i)) for i in range(n)]


def build_ba(n, m, seed):
    graph = nx.barabasi_albert_graph(n, m, seed=seed)
    return [list(graph.neighbors(i)) for i in range(n)]


def mc_stationary(
    nbrs,
    gamma,
    p1=P1,
    p2=P2,
    tau=TAU,
    burn_per_node=BURN_PER_NODE,
    measure_per_node=MEASURE_PER_NODE,
    reps=24,
    seed=0,
):
    rng = np.random.default_rng(seed)
    n = len(nbrs)
    max_deg = max(len(nb) for nb in nbrs)
    nb_arr = np.zeros((n, max_deg), dtype=np.int32)
    deg_arr = np.array([len(nb) for nb in nbrs], dtype=np.int32)
    for i, nb in enumerate(nbrs):
        nb_arr[i, : len(nb)] = nb

    burn_steps = int(burn_per_node * n)
    measure_steps = int(measure_per_node * n)
    sample_interval = max(1, n)
    fit_vals = []
    pi0_vals = []

    def run_phase(cap, steps, accumulate=False):
        delta_sum = 0.0
        pi0_sum = 0.0
        samples = 0
        nodes = rng.integers(0, n, size=steps)
        game_rands = rng.random(steps)
        contest_rands = rng.random(steps)
        penalty_rands = rng.random(steps)
        play_rands = rng.random(steps)
        nb_u = rng.random(steps)
        for t in range(steps):
            i = nodes[t]
            deg = deg_arr[i]
            if deg == 0:
                continue
            delta = 0.0
            if game_rands[t] < gamma:
                idx = min(int(nb_u[t] * deg), deg - 1)
                j = nb_arr[i, idx]
                if contest_rands[t] < 0.5:
                    cap[i] += 1
                    cap[j] -= 1
                else:
                    cap[i] -= 1
                    cap[j] += 1
                if penalty_rands[t] < tau:
                    cap[i] -= 1
                    delta -= 1.0
            else:
                state = int(cap[i]) % 3
                p_win = p1 if state == 0 else p2
                if play_rands[t] < p_win:
                    cap[i] += 1
                    delta += 1.0
                else:
                    cap[i] -= 1
                    delta -= 1.0
            if accumulate:
                delta_sum += delta
                if (t + 1) % sample_interval == 0:
                    pi0_sum += np.mean(np.mod(cap, 3) == 0)
                    samples += 1
        if not accumulate:
            return 0.0, 0.0
        return delta_sum / (n * steps), pi0_sum / max(samples, 1)

    for _ in range(reps):
        cap = np.zeros(n, dtype=np.float64)
        run_phase(cap, burn_steps, accumulate=False)
        fit, pi0 = run_phase(cap, measure_steps, accumulate=True)
        fit_vals.append(fit)
        pi0_vals.append(pi0)
    fit_vals = np.array(fit_vals)
    pi0_vals = np.array(pi0_vals)
    return {
        "fitness_mean": fit_vals.mean(),
        "fitness_sem": fit_vals.std(ddof=1) / np.sqrt(len(fit_vals)) if len(fit_vals) > 1 else 0.0,
        "pi0_mean": pi0_vals.mean(),
        "pi0_sem": pi0_vals.std(ddof=1) / np.sqrt(len(pi0_vals)) if len(pi0_vals) > 1 else 0.0,
    }


def compute_window_peak(tau, p1=P1, p2=P2, n=600):
    gammas = np.linspace(0.0, 1.0, n)
    vals = np.array([f_mix(g, p1, p2, tau) for g in gammas])
    valid = ~np.isnan(vals)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan
    peak_val = vals[valid].max()
    if peak_val <= 0:
        return np.nan, np.nan, peak_val
    peak_g = gammas[valid][np.argmax(vals[valid])]
    gl = np.nan
    gh = np.nan
    left_g = gammas[valid][gammas[valid] < peak_g]
    left_v = vals[valid][gammas[valid] < peak_g]
    for i in range(len(left_g) - 1, 0, -1):
        if left_v[i] > 0 and left_v[i - 1] <= 0:
            gl = brentq(lambda g: f_mix(g, p1, p2, tau), left_g[i - 1], left_g[i])
            break
    if np.isnan(gl) and len(left_g) > 0:
        gl = left_g[0]
    right_g = gammas[valid][gammas[valid] > peak_g]
    right_v = vals[valid][gammas[valid] > peak_g]
    for i in range(len(right_g) - 1):
        if right_v[i] > 0 and right_v[i + 1] <= 0:
            gh = brentq(lambda g: f_mix(g, p1, p2, tau), right_g[i], right_g[i + 1])
            break
    return gl, gh, peak_val


def window_roots(p1=P1, p2=P2, tau=TAU, n=1001):
    gammas = np.linspace(0.0, 1.0, n)
    vals = np.array([f_mix(g, p1, p2, tau) for g in gammas])
    roots = []
    for i in range(len(gammas) - 1):
        if abs(vals[i]) < 1e-13:
            roots.append(float(gammas[i]))
        elif vals[i] * vals[i + 1] < 0:
            roots.append(float(brentq(lambda x: f_mix(x, p1, p2, tau), gammas[i], gammas[i + 1])))
    return roots, float(np.nanmax(vals))


def load_or_compute_data():
    if CACHE_PATH.exists():
        cached = dict(np.load(CACHE_PATH, allow_pickle=True))
        if int(np.ravel(cached.get("cache_version", [-1]))[0]) == CACHE_VERSION:
            print(f"Loading cached data from {CACHE_PATH}")
            return cached
        print(f"Ignoring old cache at {CACHE_PATH}; recomputing v{CACHE_VERSION} data")

    print("Computing Monte Carlo data for figure generation...")
    data = {}

    data["cache_version"] = np.array([CACHE_VERSION])

    gammas_val = np.linspace(0.0, 1.0, 17)
    nbrs = build_ring(N_MC, 6)
    pi0_mean, pi0_sem, fit_mean, fit_sem = [], [], [], []
    for gi, gamma in enumerate(gammas_val):
        out = mc_stationary(nbrs, gamma, reps=28, seed=BASE_SEED + 2000 + gi)
        pi0_mean.append(out["pi0_mean"])
        pi0_sem.append(out["pi0_sem"])
        fit_mean.append(out["fitness_mean"])
        fit_sem.append(out["fitness_sem"])
    data["fig4_validation_g"] = gammas_val
    data["fig4_validation_pi0_mean"] = np.array(pi0_mean)
    data["fig4_validation_pi0_sem"] = np.array(pi0_sem)
    data["fig4_validation_fit_mean"] = np.array(fit_mean)
    data["fig4_validation_fit_sem"] = np.array(fit_sem)
    print("  Fig. 4 stationary validation done")

    pr_list = np.array([0.0, 0.05, 0.15, 0.30, 0.60, 1.0])
    gammas4 = np.linspace(0.02, 0.88, 13)
    fg4 = np.zeros((len(pr_list), len(gammas4)))
    fg4_sem = np.zeros_like(fg4)
    for i, pr in enumerate(pr_list):
        nbrs = build_ws(N_MC, 6, float(pr), seed=BASE_SEED + i)
        for j, gamma in enumerate(gammas4):
            out = mc_stationary(nbrs, gamma, reps=18, seed=BASE_SEED + 40000 + i * 100 + j)
            fg4[i, j] = out["fitness_mean"]
            fg4_sem[i, j] = out["fitness_sem"]
        print(f"  Fig. 4 p_r={pr:.2f} done")
    data["fig4_pr"] = pr_list
    data["fig4_g"] = gammas4
    data["fig4_f"] = fg4
    data["fig4_f_sem"] = fg4_sem

    gammas6 = np.linspace(0.02, 0.88, 13)
    labels6 = np.array(["WS k=6", "BA m=2", "BA m=3", "BA m=5"])
    f6 = np.zeros((len(labels6), len(gammas6)))
    f6_sem = np.zeros_like(f6)
    networks = [
        build_ring(N_MC, 6),
        build_ba(N_MC, 2, seed=BASE_SEED + 62),
        build_ba(N_MC, 3, seed=BASE_SEED + 63),
        build_ba(N_MC, 5, seed=BASE_SEED + 65),
    ]
    for i, nbrs in enumerate(networks):
        for j, gamma in enumerate(gammas6):
            out = mc_stationary(nbrs, gamma, reps=20, seed=BASE_SEED + 60000 + i * 100 + j)
            f6[i, j] = out["fitness_mean"]
            f6_sem[i, j] = out["fitness_sem"]
        print(f"  Fig. 6 {labels6[i]} done")
    data["fig6_g"] = gammas6
    data["fig6_labels"] = labels6
    data["fig6_f"] = f6
    data["fig6_f_sem"] = f6_sem

    np.savez(CACHE_PATH, **data)
    print(f"Saved cache to {CACHE_PATH}")
    return data


def arrow(ax, p1, p2, color, rad=0.0, lw=1.7, ms=13):
    patch = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="->",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    return patch


def rounded_box(ax, xy, w, h, fc, ec, lw=1.4, r=0.10):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.025,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def _make_fig1_legacy():
    fig = plt.figure(figsize=(10.2, 5.35))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.15, 1.35, 1.15],
        height_ratios=[1.0, 1.0],
        wspace=0.28,
        hspace=0.25,
    )
    ax_net = fig.add_subplot(gs[:, 0])
    ax_rules = fig.add_subplot(gs[0, 1:])
    ax_trans = fig.add_subplot(gs[1, 1])
    ax_cond = fig.add_subplot(gs[1, 2])

    for ax in [ax_net, ax_rules, ax_trans, ax_cond]:
        ax.set_axis_off()

    panel_label(ax_net, "(a)", x=-0.02, y=1.01)
    ax_net.set_xlim(-1.65, 1.65)
    ax_net.set_ylim(-2.25, 1.95)
    ax_net.set_aspect("equal")
    n = 10
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    xs, ys = np.cos(angles), np.sin(angles)
    states = [1, 2, 1, 0, 0, 1, 2, 1, 0, 2]
    for i in range(n):
        ax_net.plot(
            [xs[i], xs[(i + 1) % n]],
            [ys[i], ys[(i + 1) % n]],
            color="#BFC5CC",
            lw=1.7,
            zorder=1,
        )
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax_net.add_patch(Circle((x, y), 0.16, color=STATE[states[i]], ec="white", lw=1.4, zorder=3))
    ax_net.add_patch(Circle((xs[3], ys[3]), 0.30, fill=False, ec=COL["purple"], lw=1.8, ls="--", zorder=4))
    arrow(ax_net, (xs[2] * 0.86, ys[2] * 0.86), (xs[3] * 0.86, ys[3] * 0.86), COL["orange"], rad=-0.15, lw=1.8)
    ax_net.text(0, 1.45, "Network state", ha="center", fontsize=13.2, fontweight="bold", color=COL["black"])
    ax_net.text(
        0.82,
        0.82,
        "Game A\nneighbor contest",
        ha="left",
        va="center",
        fontsize=9.2,
        color=COL["orange"],
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5),
        zorder=8,
    )
    ax_net.text(
        -1.58,
        -1.03,
        "Game B\nlocal state",
        ha="left",
        va="center",
        fontsize=9.2,
        color=COL["purple"],
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5),
        zorder=8,
    )
    ax_net.text(0, -1.45, r"state $s_i=c_i\,\mathrm{mod}\,3$", ha="center", fontsize=10.6, color=COL["gray"])
    legend_x = [-0.78, 0.0, 0.78]
    legend_labels = [r"$s=0$ trap", r"$s=1$", r"$s=2$"]
    for x, state, label in zip(legend_x, [0, 1, 2], legend_labels):
        ax_net.add_patch(Circle((x, -1.78), 0.105, color=STATE[state], ec="white", lw=1.0))
        ax_net.text(x, -2.02, label, ha="center", fontsize=9.6, color=STATE[state], fontweight="bold")

    panel_label(ax_rules, "(b)", x=-0.03, y=1.01)
    ax_rules.set_xlim(0, 10)
    ax_rules.set_ylim(0, 4.2)
    rounded_box(ax_rules, (0.35, 0.40), 4.45, 3.20, COL["light_orange"], COL["orange"])
    rounded_box(ax_rules, (5.20, 0.40), 4.45, 3.20, COL["light_purple"], COL["purple"])
    ax_rules.text(2.58, 3.27, "Game A", ha="center", fontsize=13.2, fontweight="bold", color=COL["orange"])
    ax_rules.text(2.58, 3.00, "competition + penalty", ha="center", fontsize=10.2, color=COL["orange"])
    ax_rules.add_patch(Circle((1.55, 2.20), 0.34, color=COL["orange"], ec="white", lw=1.3))
    ax_rules.add_patch(Circle((3.55, 2.20), 0.34, color=COL["orange"], ec="white", lw=1.3))
    ax_rules.text(1.55, 2.20, r"$i$", ha="center", va="center", fontsize=14.2, color="white", fontweight="bold")
    ax_rules.text(3.55, 2.20, r"$j$", ha="center", va="center", fontsize=14.2, color="white", fontweight="bold")
    arrow(ax_rules, (1.96, 2.20), (3.14, 2.20), COL["red"], rad=0.05, lw=1.6)
    arrow(ax_rules, (3.14, 2.00), (1.96, 2.00), COL["red"], rad=0.05, lw=1.6)
    ax_rules.text(2.55, 2.66, r"$\pm 1$ with prob. $1/2$", ha="center", fontsize=10.5, color=COL["red"])
    arrow(ax_rules, (1.55, 1.75), (1.55, 1.20), COL["purple"], lw=1.6)
    ax_rules.text(2.78, 1.48, r"penalty $-1$ with prob. $\tau$", ha="center", fontsize=10.0, color=COL["purple"])
    ax_rules.text(2.58, 0.82, r"$\langle\Delta c_i\rangle_A=-\tau$", ha="center", fontsize=11.6, color=COL["red"], fontweight="bold")

    ax_rules.text(7.42, 3.27, "Game B", ha="center", fontsize=13.2, fontweight="bold", color=COL["purple"])
    ax_rules.text(7.42, 3.00, "capital-dependent local game", ha="center", fontsize=10.2, color=COL["purple"])
    ax_rules.add_patch(Circle((6.35, 2.12), 0.34, color=STATE[0], ec="white", lw=1.3))
    ax_rules.text(6.35, 2.12, r"$s=0$", ha="center", va="center", fontsize=10.4, color="white", fontweight="bold")
    ax_rules.text(6.35, 1.55, r"win $p_1=0.095$", ha="center", fontsize=10.0, color=STATE[0])
    ax_rules.text(6.35, 1.17, "low-win trap", ha="center", fontsize=9.5, color=COL["gray"])
    ax_rules.add_patch(Circle((8.46, 2.12), 0.42, color=STATE[2], ec="white", lw=1.3))
    ax_rules.text(8.46, 2.12, r"$s=1,2$", ha="center", va="center", fontsize=9.2, color="white", fontweight="bold")
    ax_rules.text(8.46, 1.55, r"win $p_2=0.745$", ha="center", fontsize=9.8, color=STATE[2])
    ax_rules.text(8.46, 1.17, "high-win states", ha="center", fontsize=9.4, color=COL["gray"])
    arrow(ax_rules, (7.05, 2.12), (7.75, 2.12), COL["gray"], rad=0.0, lw=1.2, ms=11)
    ax_rules.text(7.42, 0.82, "mixed play combines A and B", ha="center", fontsize=10.0, color=COL["black"])

    panel_label(ax_trans, "(c)", x=-0.04, y=1.01)
    ax_trans.set_xlim(-0.25, 3.25)
    ax_trans.set_ylim(-0.10, 3.05)
    ax_trans.set_aspect("equal")
    pos = {0: (1.5, 2.35), 1: (2.55, 0.65), 2: (0.45, 0.65)}
    for state, (x, y) in pos.items():
        ax_trans.add_patch(Circle((x, y), 0.34, color=STATE[state], ec="white", lw=1.5, zorder=3))
        text = r"$s=0$" + "\ntrap" if state == 0 else rf"$s={state}$"
        ax_trans.text(x, y, text, ha="center", va="center", fontsize=10.0, color="white", fontweight="bold", zorder=4)
    arrow(ax_trans, (1.72, 2.08), (2.35, 0.96), STATE[0], rad=-0.10)
    ax_trans.text(2.30, 1.75, r"$p_1$", fontsize=10.7, color=STATE[0], fontweight="bold")
    arrow(ax_trans, (2.28, 0.50), (0.72, 0.50), COL["gray"], rad=-0.10)
    ax_trans.text(1.50, 0.15, r"$1-p_2$", fontsize=9.9, color=COL["gray"], ha="center")
    arrow(ax_trans, (0.66, 0.95), (1.28, 2.08), STATE[2], rad=-0.10)
    ax_trans.text(0.60, 1.78, r"$p_2$", fontsize=10.7, color=STATE[2], fontweight="bold")
    arrow(ax_trans, (2.34, 0.94), (1.72, 2.07), COL["gray"], rad=0.15)
    ax_trans.text(2.75, 1.58, r"$1-p_2$", fontsize=9.9, color=COL["gray"])
    arrow(ax_trans, (0.70, 0.49), (2.30, 0.49), STATE[1], rad=0.12)
    ax_trans.text(1.50, 0.78, r"$p_2$", fontsize=10.7, color=STATE[1], fontweight="bold", ha="center")
    arrow(ax_trans, (1.26, 2.06), (0.66, 0.95), COL["gray"], rad=0.15)
    ax_trans.text(0.05, 1.58, r"$1-p_1$", fontsize=9.9, color=COL["gray"])
    ax_trans.text(1.5, 2.90, "Game B transition rates", ha="center", fontsize=11.3, color=COL["black"], fontweight="bold")

    ax_cond.set_xlim(0, 1)
    ax_cond.set_ylim(0, 1)
    rounded_box(ax_cond, (0.04, 0.12), 0.92, 0.76, COL["light_green"], COL["green"], lw=1.8, r=0.06)
    ax_cond.text(0.50, 0.78, "Necessary condition", ha="center", fontsize=12.2, fontweight="bold", color=COL["green"])
    ax_cond.text(
        0.50,
        0.56,
        r"$\bar{\pi}_0^{\rm mix}<\pi_0^*$" + "\n" + r"$\pi_0^*=\frac{2p_2-1}{2(p_2-p_1)}$",
        ha="center",
        va="center",
        fontsize=11.9,
        color="#14532D",
        linespacing=1.35,
    )
    ax_cond.text(0.50, 0.27, r"trap occupancy below threshold", ha="center", fontsize=9.5, color=COL["gray"])
    ax_cond.text(0.50, 0.18, r"$\pi_0^*$ independent of $\gamma$, $\tau$, topology", ha="center", fontsize=8.7, color=COL["gray"])

    savefig(fig, "Fig1_model_schematic.png")


def _make_fig1_v14():
    fig = plt.figure(figsize=(10.4, 5.45))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.08, 1.45, 1.18],
        height_ratios=[1.0, 1.05],
        wspace=0.34,
        hspace=0.36,
    )
    ax_net = fig.add_subplot(gs[:, 0])
    ax_rules = fig.add_subplot(gs[0, 1:])
    ax_trans = fig.add_subplot(gs[1, 1])
    ax_cond = fig.add_subplot(gs[1, 2])

    for ax in [ax_net, ax_rules, ax_trans, ax_cond]:
        ax.set_axis_off()

    # (a) Network state: highlight the two mechanisms without text-node overlap.
    panel_label(ax_net, "(a)", x=-0.02, y=1.01)
    ax_net.set_xlim(-1.72, 1.80)
    ax_net.set_ylim(-2.25, 1.92)
    ax_net.set_aspect("equal")
    n = 10
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    xs, ys = np.cos(angles), np.sin(angles)
    states = [1, 2, 1, 0, 0, 1, 2, 1, 0, 2]
    for i in range(n):
        ax_net.plot([xs[i], xs[(i + 1) % n]], [ys[i], ys[(i + 1) % n]], color="#BFC5CC", lw=1.7, zorder=1)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax_net.add_patch(Circle((x, y), 0.16, color=STATE[states[i]], ec="white", lw=1.4, zorder=3))
    ax_net.plot([xs[2], xs[3]], [ys[2], ys[3]], color=COL["orange"], lw=3.0, alpha=0.85, zorder=2)
    ax_net.add_patch(Circle((xs[3], ys[3]), 0.30, fill=False, ec=COL["purple"], lw=1.8, ls="--", zorder=4))
    ax_net.text(0, 1.45, "Network state", ha="center", fontsize=13.2, fontweight="bold", color=COL["black"])
    note_box = dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.5)
    ax_net.text(
        0.95,
        0.78,
        "Game A:\nneighbor contest",
        ha="left",
        va="center",
        fontsize=9.0,
        color=COL["orange"],
        fontweight="bold",
        linespacing=1.15,
        bbox=note_box,
        zorder=8,
    )
    ax_net.text(
        -1.58,
        -1.10,
        "Game B:\nlocal state",
        ha="left",
        va="center",
        fontsize=9.0,
        color=COL["purple"],
        fontweight="bold",
        linespacing=1.15,
        bbox=note_box,
        zorder=8,
    )
    ax_net.text(0, -1.45, r"state $s_i=c_i\,\mathrm{mod}\,3$", ha="center", fontsize=10.6, color=COL["gray"])
    for x, state, label in zip([-0.78, 0.0, 0.78], [0, 1, 2], [r"$s=0$ trap", r"$s=1$", r"$s=2$"]):
        ax_net.add_patch(Circle((x, -1.78), 0.105, color=STATE[state], ec="white", lw=1.0))
        ax_net.text(x, -2.02, label, ha="center", fontsize=9.6, color=STATE[state], fontweight="bold")

    # (b) Game rules: use one double-headed interaction arrow and separated penalty text.
    panel_label(ax_rules, "(b)", x=-0.03, y=1.01)
    ax_rules.set_xlim(0, 10)
    ax_rules.set_ylim(0, 4.45)
    rounded_box(ax_rules, (0.28, 0.36), 4.55, 3.55, COL["light_orange"], COL["orange"])
    rounded_box(ax_rules, (5.16, 0.36), 4.55, 3.55, COL["light_purple"], COL["purple"])

    ax_rules.text(2.55, 3.58, "Game A", ha="center", fontsize=13.0, fontweight="bold", color=COL["orange"])
    ax_rules.text(2.55, 3.22, "competition + penalty", ha="center", fontsize=10.1, color=COL["orange"])
    ax_rules.add_patch(Circle((1.42, 2.52), 0.32, color=COL["orange"], ec="white", lw=1.3))
    ax_rules.add_patch(Circle((3.68, 2.52), 0.32, color=COL["orange"], ec="white", lw=1.3))
    ax_rules.text(1.42, 2.52, r"$i$", ha="center", va="center", fontsize=13.6, color="white", fontweight="bold")
    ax_rules.text(3.68, 2.52, r"$j$", ha="center", va="center", fontsize=13.6, color="white", fontweight="bold")
    ax_rules.add_patch(
        FancyArrowPatch(
            (1.82, 2.52),
            (3.28, 2.52),
            arrowstyle="<->",
            mutation_scale=14,
            linewidth=1.7,
            color=COL["red"],
            connectionstyle="arc3,rad=0",
        )
    )
    ax_rules.text(2.55, 2.02, r"$\pm 1$ with prob. $1/2$", ha="center", fontsize=10.2, color=COL["red"])
    arrow(ax_rules, (1.42, 1.90), (1.42, 1.34), COL["purple"], lw=1.5, ms=12)
    ax_rules.text(2.75, 1.62, r"penalty $-1$ with prob. $\tau$", ha="center", va="center", fontsize=9.9, color=COL["purple"])
    ax_rules.text(2.55, 0.82, r"$\langle\Delta c_i\rangle_A=-\tau$", ha="center", fontsize=11.4, color=COL["red"], fontweight="bold")

    ax_rules.text(7.43, 3.58, "Game B", ha="center", fontsize=13.0, fontweight="bold", color=COL["purple"])
    ax_rules.text(7.43, 3.22, "capital-dependent local game", ha="center", fontsize=10.1, color=COL["purple"])
    ax_rules.add_patch(Circle((6.38, 2.50), 0.35, color=STATE[0], ec="white", lw=1.3))
    ax_rules.add_patch(Circle((8.48, 2.50), 0.40, color=STATE[2], ec="white", lw=1.3))
    ax_rules.text(6.38, 2.50, r"$s=0$", ha="center", va="center", fontsize=10.1, color="white", fontweight="bold")
    ax_rules.text(8.48, 2.50, r"$s=1,2$", ha="center", va="center", fontsize=9.0, color="white", fontweight="bold")
    arrow(ax_rules, (6.88, 2.50), (7.95, 2.50), COL["gray"], rad=0.0, lw=1.3, ms=12)
    ax_rules.text(6.38, 1.78, r"win $p_1=0.095$", ha="center", fontsize=9.7, color=STATE[0])
    ax_rules.text(6.38, 1.42, "low-win trap", ha="center", fontsize=9.0, color=COL["gray"])
    ax_rules.text(8.48, 1.78, r"win $p_2=0.745$", ha="center", fontsize=9.7, color=STATE[2])
    ax_rules.text(8.48, 1.42, "high-win states", ha="center", fontsize=9.0, color=COL["gray"])
    ax_rules.text(7.43, 0.82, "mixed play combines A and B", ha="center", fontsize=9.8, color=COL["black"])

    # (c) Transition diagram: show the ratchet directions and list reverse loss rates.
    panel_label(ax_trans, "(c)", x=-0.04, y=1.01)
    ax_trans.set_xlim(-0.20, 3.25)
    ax_trans.set_ylim(-0.10, 3.22)
    ax_trans.set_aspect("equal")
    pos = {0: (1.52, 2.50), 1: (2.72, 0.74), 2: (0.32, 0.74)}
    for state, (x, y) in pos.items():
        ax_trans.add_patch(Circle((x, y), 0.34, color=STATE[state], ec="white", lw=1.5, zorder=5))
        text = r"$s=0$" + "\ntrap" if state == 0 else rf"$s={state}$"
        ax_trans.text(x, y, text, ha="center", va="center", fontsize=9.9, color="white", fontweight="bold", zorder=6)

    def rate_arrow(start, end, color, rad, lw=1.5, ms=12):
        ax_trans.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                mutation_scale=ms,
                linewidth=lw,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=18,
                shrinkB=18,
                zorder=2,
            )
        )

    label_box = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.25)
    rate_arrow(pos[0], pos[1], STATE[0], -0.06, lw=1.7, ms=13)
    rate_arrow(pos[1], pos[2], STATE[1], 0.08, lw=1.7, ms=13)
    rate_arrow(pos[2], pos[0], STATE[2], -0.06, lw=1.7, ms=13)
    ax_trans.text(2.22, 1.72, r"$p_1$", fontsize=10.8, color=STATE[0], fontweight="bold", bbox=label_box)
    ax_trans.text(1.52, 0.98, r"$p_2$", fontsize=10.8, color=STATE[1], fontweight="bold", ha="center", bbox=label_box)
    ax_trans.text(0.68, 1.72, r"$p_2$", fontsize=10.8, color=STATE[2], fontweight="bold", bbox=label_box)
    rounded_box(ax_trans, (0.18, 0.03), 2.68, 0.28, "#F8FAFC", "#CBD5E1", lw=0.8, r=0.04)
    ax_trans.text(
        1.52,
        0.17,
        r"reverse loss rates: $1-p_1,\;1-p_2,\;1-p_2$",
        ha="center",
        va="center",
        fontsize=8.7,
        color=COL["gray"],
    )
    ax_trans.text(1.52, 3.05, "Game B transition rates", ha="center", fontsize=11.3, color=COL["black"], fontweight="bold")

    ax_cond.set_xlim(0, 1)
    ax_cond.set_ylim(0, 1)
    rounded_box(ax_cond, (0.05, 0.12), 0.90, 0.76, COL["light_green"], COL["green"], lw=1.8, r=0.06)
    ax_cond.text(0.50, 0.76, "Necessary condition", ha="center", fontsize=12.0, fontweight="bold", color=COL["green"])
    ax_cond.text(0.50, 0.57, r"$\bar{\pi}_0^{\mathrm{mix}} < \pi_0^\ast$", ha="center", fontsize=15.0, color="#14532D")
    ax_cond.text(0.50, 0.42, r"$\pi_0^\ast=\frac{2p_2-1}{2(p_2-p_1)}$", ha="center", fontsize=12.3, color="#14532D")
    ax_cond.text(0.50, 0.27, "trap occupancy below threshold", ha="center", fontsize=9.4, color=COL["gray"])
    ax_cond.text(0.50, 0.18, r"$\pi_0^\ast$ independent of $\gamma$, $\tau$, topology", ha="center", fontsize=8.6, color=COL["gray"])

    savefig(fig, "Fig1_model_schematic.png")


def make_fig1():
    fig = plt.figure(figsize=(7.5, 4.55), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.035,
        right=0.985,
        top=0.965,
        bottom=0.045,
        wspace=0.14,
        hspace=0.18,
    )
    ax_net = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_cond = fig.add_subplot(gs[1, 1])

    for ax in [ax_net, ax_a, ax_b, ax_cond]:
        ax.set_axis_off()

    def tag(ax, label):
        ax.text(
            -0.08,
            1.03,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11.5,
            fontweight="bold",
            color=COL["black"],
            clip_on=False,
        )

    # (a) Network state: show mechanisms by marks, not explanatory text.
    tag(ax_net, "(a)")
    ax_net.set_xlim(-1.28, 1.48)
    ax_net.set_ylim(-1.23, 1.19)
    ax_net.set_aspect("equal")
    n = 10
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    xs, ys = np.cos(angles), np.sin(angles)
    states = [1, 2, 1, 0, 0, 1, 2, 1, 0, 2]
    for i in range(n):
        ax_net.plot([xs[i], xs[(i + 1) % n]], [ys[i], ys[(i + 1) % n]], color="#C7CCD1", lw=1.7, zorder=1)
    ax_net.plot([xs[2], xs[3]], [ys[2], ys[3]], color=COL["orange"], lw=3.2, solid_capstyle="round", zorder=2)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax_net.add_patch(Circle((x, y), 0.155, color=STATE[states[i]], ec="white", lw=1.2, zorder=4))
    ax_net.add_patch(Circle((xs[3], ys[3]), 0.30, fill=False, ec=COL["purple"], lw=1.7, ls="--", zorder=5))
    badge = dict(boxstyle="round,pad=0.18,rounding_size=0.04", ec="none")
    ax_net.text(1.33, 0.39, "A", ha="center", va="center", fontsize=10.0, fontweight="bold",
                color="white", bbox={**badge, "fc": COL["orange"]})
    ax_net.text(1.33, -0.43, "B", ha="center", va="center", fontsize=10.0, fontweight="bold",
                color="white", bbox={**badge, "fc": COL["purple"]})

    # (b) Game A: only the contest, penalty, and mean drift.
    tag(ax_a, "(b)")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    rounded_box(ax_a, (0.07, 0.12), 0.86, 0.76, "#FFF8EC", COL["orange"], lw=1.1, r=0.032)
    ax_a.text(0.50, 0.79, "Game A", ha="center", fontsize=11.4, fontweight="bold", color=COL["orange"])
    ax_a.add_patch(Circle((0.32, 0.56), 0.092, color=COL["orange"], ec="white", lw=1.0))
    ax_a.add_patch(Circle((0.68, 0.56), 0.092, color=COL["orange"], ec="white", lw=1.0))
    ax_a.text(0.32, 0.56, r"$i$", ha="center", va="center", fontsize=11.0, color="white", fontweight="bold")
    ax_a.text(0.68, 0.56, r"$j$", ha="center", va="center", fontsize=11.0, color="white", fontweight="bold")
    ax_a.add_patch(
        FancyArrowPatch(
            (0.42, 0.56),
            (0.58, 0.56),
            arrowstyle="<->",
            mutation_scale=12,
            linewidth=1.5,
            color=COL["red"],
        )
    )
    ax_a.text(0.50, 0.39, r"$\pm 1$  $(1/2)$", ha="center", fontsize=9.8, color=COL["red"])
    ax_a.text(0.50, 0.26, r"$-1$  $(\tau)$", ha="center", fontsize=9.6, color=COL["purple"])
    ax_a.text(0.50, 0.145, r"$\langle\Delta c_i\rangle_A=-\tau$", ha="center", fontsize=10.3,
              color=COL["black"])

    # (c) Game B: the ratchet cycle with win probabilities only.
    tag(ax_b, "(c)")
    ax_b.set_xlim(-1.12, 1.12)
    ax_b.set_ylim(-0.84, 0.92)
    ax_b.set_aspect("equal")
    ax_b.text(0, 0.79, "Game B", ha="center", fontsize=11.4, fontweight="bold", color=COL["purple"])
    pos = {0: (0.0, 0.46), 1: (0.78, -0.50), 2: (-0.78, -0.50)}
    for state, (x, y) in pos.items():
        ax_b.add_patch(Circle((x, y), 0.18, color=STATE[state], ec="white", lw=1.1, zorder=4))
        ax_b.text(x, y, rf"${state}$", ha="center", va="center", fontsize=11.0, color="white",
                  fontweight="bold", zorder=5)

    def cycle_arrow(start, end, color, rad):
        ax_b.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.55,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=16,
                shrinkB=16,
                zorder=2,
            )
        )

    cycle_arrow(pos[0], pos[1], STATE[0], -0.06)
    cycle_arrow(pos[1], pos[2], STATE[1], 0.06)
    cycle_arrow(pos[2], pos[0], STATE[2], -0.06)
    label_box = dict(facecolor="white", edgecolor="none", alpha=0.90, pad=0.2)
    ax_b.text(0.52, 0.03, r"$p_1$", fontsize=10.0, color=STATE[0], fontweight="bold", bbox=label_box)
    ax_b.text(0.00, -0.73, r"$p_2$", fontsize=10.0, color=STATE[1], fontweight="bold", ha="center", bbox=label_box)
    ax_b.text(-0.55, 0.03, r"$p_2$", fontsize=10.0, color=STATE[2], fontweight="bold", bbox=label_box)

    # (d) Keep the theorem-level message separate from the transition diagram.
    tag(ax_cond, "(d)")
    ax_cond.set_xlim(0, 1)
    ax_cond.set_ylim(0, 1)
    rounded_box(ax_cond, (0.10, 0.22), 0.80, 0.56, "white", COL["green"], lw=1.2, r=0.035)
    ax_cond.text(0.50, 0.58, r"$\bar{\pi}_{0}^{\mathrm{mix}}<\pi_{0}^{\ast}$", ha="center", fontsize=16.0, color="#14532D")
    ax_cond.text(0.50, 0.40, r"$\pi_{0}^{\ast}=\frac{2p_2-1}{2(p_2-p_1)}$", ha="center", fontsize=12.2, color="#14532D")

    savefig(fig, "Fig1_model_schematic.png")


def make_fig2(data):
    gammas = data["fig4_validation_g"]
    roots, _ = window_roots(P1, P2, TAU)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9), constrained_layout=True)

    ax = axes[0]
    panel_label(ax, "(a)")
    ax.axhline(PI0_STAR, color=COL["black"], ls=":", lw=1.4, label=rf"$\pi_0^*={PI0_STAR:.3f}$")
    g_fine = np.linspace(0.0, 1.0, 360)
    ax.plot(g_fine, [solve_mf(g)[0] for g in g_fine], color=COL["blue"], lw=2.0, label="complete theory")
    ax.errorbar(
        gammas,
        data["fig4_validation_pi0_mean"],
        yerr=data["fig4_validation_pi0_sem"],
        fmt="o",
        ms=4.0,
        color=COL["orange"],
        capsize=2.0,
        label="stationary MC",
    )
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Trap occupancy $\pi_0$")
    ax.set_ylim(0.325, 0.392)
    finish_axes(ax)

    ax = axes[1]
    panel_label(ax, "(b)")
    ax.axhline(0, color=COL["black"], ls="--", lw=1.0)
    if len(roots) >= 2:
        ax.axvspan(roots[0], roots[1], color=COL["light_green"], zorder=0)
        ax.axvline(roots[0], color=COL["gray"], ls=":", lw=1.1)
        ax.axvline(roots[1], color=COL["gray"], ls=":", lw=1.1)
    ax.plot(g_fine, [f_mix(g) / N_MC * 1e4 for g in g_fine], color=COL["blue"], lw=2.0, label=r"theory $f_{\rm mix}/N$")
    ax.errorbar(
        gammas,
        data["fig4_validation_fit_mean"] * 1e4,
        yerr=data["fig4_validation_fit_sem"] * 1e4,
        fmt="o",
        ms=4.0,
        color=COL["orange"],
        capsize=2.0,
        label="stationary MC",
    )
    ax.legend(loc="upper right", framealpha=0.92)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Average fitness $(\times10^{-4})$")
    finish_axes(ax)

    ax = axes[2]
    panel_label(ax, "(c)")
    ba = nx.barabasi_albert_graph(N_MC, 3, seed=BASE_SEED + 909)
    nbrs_ba = [list(ba.neighbors(i)) for i in range(N_MC)]
    deg = np.array([len(nb) for nb in nbrs_ba])
    lam = lambda_values(nbrs_ba)
    gamma_demo = 0.25
    base = solve_mf(gamma_demo)[0]
    unique_deg = np.array(sorted(set(deg)))
    shown_deg = []
    shifts = []
    for k in unique_deg:
        if np.sum(deg == k) >= 3:
            shown_deg.append(k)
            shifts.append(np.mean([solve_mf(gamma_demo, lam=l)[0] - base for l in lam[deg == k]]))
    ax.axhline(0, color=COL["black"], ls="--", lw=1.0)
    ax.plot(shown_deg, np.array(shifts) * 1e3, color=COL["green"], marker="o", ms=4.0, lw=1.8)
    ax.set_xlabel(r"Degree $k$ in one BA network")
    ax.set_ylabel(r"$\langle\pi_0(\lambda_i)-\pi_0(1)\rangle_k$ $(\times10^{-3})$")
    finish_axes(ax)
    savefig(fig, "Fig4_MF_PA_MC_comparison.png")


def make_fig3():
    tc, gc = find_tauc(P1, P2)
    tau_arr = np.linspace(0.002, tc * 1.04, 180)
    gl = np.full_like(tau_arr, np.nan)
    gh = np.full_like(tau_arr, np.nan)
    peaks = np.full_like(tau_arr, np.nan)
    for i, tau in enumerate(tau_arr):
        gl[i], gh[i], peaks[i] = compute_window_peak(tau)
    ok = ~np.isnan(peaks) & (peaks > 0)
    width = np.where(ok, gh - gl, 0)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9), constrained_layout=True)
    ax = axes[0]
    panel_label(ax, "(a)")
    ax.fill_between(tau_arr[ok], gl[ok], gh[ok], color=COL["light_green"], label="Paradox window")
    ax.plot(tau_arr[ok], gh[ok], color=COL["red"], lw=1.9, label=r"$\gamma_{\rm high}$")
    ax.plot(tau_arr[ok], gl[ok], color=COL["blue"], lw=1.9, label=r"$\gamma_{\rm low}$")
    ax.scatter([tc], [gc], color=COL["black"], s=90, marker="*", zorder=5, label=rf"$\tau_c={tc:.3f}$")
    ax.axvline(tc, color=COL["black"], ls=":", lw=1.1)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_xlim(0, tc * 1.06)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", framealpha=0.92)
    finish_axes(ax)

    ax = axes[1]
    panel_label(ax, "(b)")
    ax.fill_between(tau_arr[ok], 0, width[ok], color=COL["light_green"])
    ax.plot(tau_arr[ok], width[ok], color=COL["green"], lw=1.9, label=r"$\Delta\gamma$")
    ax.scatter([tc], [0], color=COL["black"], s=45, zorder=5, clip_on=False)
    ax.axvline(tc, color=COL["black"], ls="--", lw=1.3, label=rf"$\tau_c={tc:.3f}$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"Window width $\Delta\gamma$")
    ax.set_xlim(0, tc * 1.06)
    ax.set_ylim(0, 1.0)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.55, 1.01),
        ncol=2,
        framealpha=0.92,
        fontsize=9.4,
        handlelength=1.4,
        columnspacing=0.9,
        borderpad=0.35,
    )
    finish_axes(ax)

    ax = axes[2]
    panel_label(ax, "(c)")
    taus = np.linspace(0.005, tc, 7)
    cmap = plt.cm.YlGnBu
    ymax = 1.0
    for i, tau in enumerate(taus):
        gli, ghi, _ = compute_window_peak(tau, n=400)
        if np.isnan(gli) or np.isnan(ghi) or ghi <= gli:
            continue
        gs = np.linspace(gli + 0.003, ghi - 0.003, 300)
        vals = uniform_filter1d(np.array([f_mix(g, P1, P2, tau) for g in gs]), size=8)
        vals_per_node = vals / N_MC
        ymax = max(ymax, np.nanmax(vals_per_node * 1e4))
        is_crit = abs(tau - tc) < 0.003
        ax.plot(
            gs,
            vals_per_node * 1e4,
            color=COL["red"] if is_crit else cmap(0.28 + 0.62 * i / (len(taus) - 1)),
            lw=2.4 if is_crit else 1.8,
            label=rf"$\tau_c$ {tau:.3f}" if is_crit else rf"{tau:.3f}",
        )
    ax.axhline(0, color=COL["black"], ls="--", lw=1.0)
    ax.scatter([gc], [0], color=COL["black"], s=45, zorder=5, clip_on=False, label=rf"$\gamma_c$ {gc:.3f}")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"MF average fitness $f_{\rm mix}/N$ $(\times10^{-4})$")
    ax.set_xlim(0.0, 0.85)
    ax.set_ylim(0, ymax * 1.45)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.52, 1.01),
        ncol=4,
        framealpha=0.92,
        fontsize=8.8,
        title=r"$\tau$",
        title_fontsize=9.0,
        handlelength=1.1,
        columnspacing=0.75,
        borderpad=0.35,
    )
    finish_axes(ax)
    savefig(fig, "Fig2_critical_point.png")


def make_fig4(data):
    pr = data["fig4_pr"]
    gammas = data["fig4_g"]
    fg = data["fig4_f"]
    roots, _ = window_roots(P1, P2, TAU)
    pr_grid, g_grid = np.meshgrid(pr, gammas, indexing="ij")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9), constrained_layout=True)

    ax = axes[0]
    panel_label(ax, "(a)")
    vm = float(np.abs(fg).max() * 1e4)
    cf = ax.contourf(g_grid, pr_grid, fg * 1e4, levels=17, cmap="RdYlGn", vmin=-vm, vmax=vm, alpha=0.88)
    try:
        cs = ax.contour(g_grid, pr_grid, fg, levels=[0], colors=COL["black"], linewidths=1.5)
        ax.clabel(cs, fmt=r"$f=0$", fontsize=10)
    except Exception:
        pass
    cb = fig.colorbar(cf, ax=ax, fraction=0.040, pad=0.012)
    cb.set_label("Fitness", labelpad=4)
    cb.ax.set_title(r"$\times10^{-4}$", fontsize=8.8, pad=4)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Rewiring probability $p_r$")
    finish_axes(ax, grid=False)

    ax = axes[1]
    panel_label(ax, "(b)")
    pd = (fg > 0).astype(float)
    ax.contourf(g_grid, pr_grid, pd, levels=[-0.5, 0.5, 1.5], colors=[COL["light_red"], COL["light_green"]])
    try:
        ax.contour(g_grid, pr_grid, pd, levels=[0.5], colors=COL["black"], linewidths=1.2)
    except Exception:
        pass
    for i, p_r in enumerate(pr):
        for j, gamma in enumerate(gammas):
            ax.scatter(gamma, p_r, color=COL["green"] if fg[i, j] > 0 else COL["red"], s=24, zorder=5)
    ax.legend(
        handles=[
            Patch(fc=COL["light_green"], ec=COL["black"], label="Paradox"),
            Patch(fc=COL["light_red"], ec=COL["black"], label="No paradox"),
        ],
        loc="upper right",
        framealpha=0.92,
    )
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Rewiring probability $p_r$")
    ax.set_xlim(gammas.min() - 0.02, gammas.max() + 0.02)
    ax.set_ylim(pr.min() - 0.03, pr.max() + 0.03)
    finish_axes(ax)

    ax = axes[2]
    panel_label(ax, "(c)")
    ax.axhline(0, color=COL["black"], ls="--", lw=1.0)
    if len(roots) >= 2:
        ax.axvspan(roots[0], roots[1], color=COL["light_green"], zorder=0)
        ax.axvline(roots[0], color=COL["gray"], ls=":", lw=1.1)
        ax.axvline(roots[1], color=COL["gray"], ls=":", lw=1.1)
    colors = [COL["blue"], COL["purple"], "#CC79A7", COL["red"], COL["orange"], COL["green"]]
    labels = {
        0.0: r"$0$",
        0.05: r"$0.05$",
        0.15: r"$0.15$",
        0.30: r"$0.30$",
        0.60: r"$0.60$",
        1.0: r"$1$",
    }
    for i, p_r in enumerate(pr):
        ax.plot(gammas, fg[i] * 1e4, color=colors[i], lw=1.55, label=labels[float(p_r)])
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Average fitness $(\times10^{-4})$")
    ax.legend(
        title=r"$p_r$",
        loc="upper right",
        ncol=2,
        framealpha=0.90,
        fontsize=7.0,
        title_fontsize=7.4,
        handlelength=1.2,
        columnspacing=0.75,
        borderpad=0.30,
    )
    finish_axes(ax)
    savefig(fig, "Fig5_WS_robustness.png")


def peak_at_tau(tau, p1=P1, p2=P2):
    res = minimize_scalar(lambda g: -f_mix(g, p1, p2, tau), bounds=(0.0, 1.0), method="bounded")
    return float(res.x), float(-res.fun)


def find_tauc(p1, p2, tau_max_cap=0.30):
    lo = 0.0
    if peak_at_tau(lo, p1, p2)[1] <= 0:
        return np.nan, np.nan
    hi = min(max(0.02, 2 * p2 - 1 + 0.05), tau_max_cap)
    while peak_at_tau(hi, p1, p2)[1] > 0 and hi < tau_max_cap:
        hi = min(hi * 1.35, tau_max_cap)
        if np.isclose(hi, tau_max_cap):
            break
    if peak_at_tau(hi, p1, p2)[1] > 0:
        return np.nan, np.nan
    tc = brentq(lambda t: peak_at_tau(t, p1, p2)[1], lo, hi, xtol=1e-11)
    gc = peak_at_tau(tc, p1, p2)[0]
    return float(tc), float(gc)


def make_fig5():
    p2s = np.linspace(0.68, 0.82, 25)
    p1s = np.linspace(0.01, 0.14, 25)
    t1, g1, pv1 = [], [], []
    t2, g2, pv2 = [], [], []
    for p2 in p2s:
        tc, gc = find_tauc(P1, p2)
        if not np.isnan(tc) and tc > 1e-4:
            t1.append(tc)
            g1.append(gc)
            pv1.append(p2)
    for p1 in p1s:
        tc, gc = find_tauc(p1, P2)
        if not np.isnan(tc) and tc > 1e-4:
            t2.append(tc)
            g2.append(gc)
            pv2.append(p1)
    tco, gco = find_tauc(P1, P2)
    pv1, t1, g1 = np.asarray(pv1), np.asarray(t1), np.asarray(g1)
    pv2, t2, g2 = np.asarray(pv2), np.asarray(t2), np.asarray(g2)

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), constrained_layout=True)
    ax = axes[0, 0]
    panel_label(ax, "(a)")
    ax.plot(pv1, t1, color=COL["blue"], lw=2.1, zorder=3)
    ax.scatter([P2], [tco], color=COL["black"], s=90, marker="*", zorder=6, label="default", clip_on=False)
    ax.set_xlabel(r"$p_2$ at fixed $p_1=0.095$")
    ax.set_ylabel(r"Critical penalty $\tau_c$")
    ax.set_xlim(float(pv1.min()) - 0.01, float(pv1.max()) + 0.01)
    ax.set_ylim(-0.010, max(max(t1), max(t2)) * 1.14)
    finish_axes(ax)

    ax = axes[0, 1]
    panel_label(ax, "(b)")
    ax.plot(pv1, g1, color=COL["blue"], lw=2.1, zorder=3)
    ax.scatter([P2], [gco], color=COL["black"], s=90, marker="*", zorder=6, label="default", clip_on=False)
    ax.set_xlabel(r"$p_2$ at fixed $p_1=0.095$")
    ax.set_ylabel(r"Critical mixing $\gamma_c$")
    ax.set_xlim(float(pv1.min()) - 0.01, float(pv1.max()) + 0.01)
    ax.set_ylim(-0.06, max(max(g1), max(g2)) * 1.10)
    finish_axes(ax)

    ax = axes[1, 0]
    panel_label(ax, "(c)")
    ax.plot(pv2, t2, color=COL["orange"], lw=2.1, zorder=3)
    ax.scatter([P1], [tco], color=COL["black"], s=90, marker="*", zorder=6, clip_on=False)
    ax.set_xlabel(r"$p_1$ at fixed $p_2=0.745$")
    ax.set_ylabel(r"Critical penalty $\tau_c$")
    ax.set_xlim(float(pv2.min()) - 0.01, float(pv2.max()) + 0.01)
    ax.set_ylim(-0.010, max(max(t1), max(t2)) * 1.14)
    finish_axes(ax)

    ax = axes[1, 1]
    panel_label(ax, "(d)")
    ax.plot(pv2, g2, color=COL["orange"], lw=2.1, zorder=3)
    ax.scatter([P1], [gco], color=COL["black"], s=90, marker="*", zorder=6, clip_on=False)
    ax.set_xlabel(r"$p_1$ at fixed $p_2=0.745$")
    ax.set_ylabel(r"Critical mixing $\gamma_c$")
    ax.set_xlim(float(pv2.min()) - 0.01, float(pv2.max()) + 0.01)
    ax.set_ylim(-0.06, max(max(g1), max(g2)) * 1.10)
    finish_axes(ax)
    savefig(fig, "Fig3_parameter_dependence.png")


def make_fig6(data):
    gammas = data["fig6_g"]
    labels = [str(x) for x in data["fig6_labels"]]
    f = data["fig6_f"]
    roots, _ = window_roots(P1, P2, TAU)
    colors = [COL["black"], COL["red"], COL["orange"], COL["blue"]]
    pretty = {
        "WS k=6": r"WS $k=6$ (baseline)",
        "BA m=2": r"BA $m=2$",
        "BA m=3": r"BA $m=3$",
        "BA m=5": r"BA $m=5$",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8), constrained_layout=True)

    ax = axes[0]
    panel_label(ax, "(a)", x=-0.11, y=1.10)
    ax.axhline(0, color=COL["black"], ls="--", lw=1.0)
    if len(roots) >= 2:
        ax.axvspan(roots[0], roots[1], color=COL["light_green"], zorder=0)
        ax.axvline(roots[0], color=COL["gray"], ls=":", lw=1.1)
        ax.axvline(roots[1], color=COL["gray"], ls=":", lw=1.1)
    for i, label in enumerate(labels):
        ax.plot(
            gammas,
            f[i] * 1e4,
            color=colors[i],
            lw=2.0 if i == 0 else 1.7,
            ls="-" if i == 0 else "--",
            marker="o",
            ms=3.5,
            label=pretty[label],
        )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.40), ncol=2, framealpha=0.92)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Average fitness $(\times10^{-4})$")
    finish_axes(ax)

    ax = axes[1]
    panel_label(ax, "(b)", x=-0.11, y=1.10)
    ax.axhline(0, color=COL["gray"], ls="--", lw=0.9)
    ax.axhline(1, color=COL["gray"], ls="--", lw=0.9)
    if len(roots) >= 2:
        ax.axvspan(roots[0], roots[1], color=COL["light_green"], zorder=0, label="Theory window")
    for i, label in enumerate(labels):
        ax.plot(
            gammas,
            (f[i] > 0).astype(float),
            color=colors[i],
            lw=2.0 if i == 0 else 1.7,
            ls="-" if i == 0 else "--",
            marker="o",
            ms=3.5,
            label=pretty[label],
        )
    ax.set_ylim(-0.12, 1.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.46), ncol=2, framealpha=0.92)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Paradox indicator")
    finish_axes(ax)
    savefig(fig, "Fig6_BA_scalefree.png")


def mf_window(p1, p2, tau):
    gammas = np.linspace(0.0, 1.0, 1001)
    vals = np.array([f_mix(g, p1, p2, tau) for g in gammas])
    roots = []
    for i in range(len(gammas) - 1):
        if vals[i] == 0:
            roots.append(gammas[i])
        elif vals[i] * vals[i + 1] < 0:
            roots.append(brentq(lambda x: f_mix(x, p1, p2, tau), gammas[i], gammas[i + 1]))
    return roots, vals.max()


def make_fig7_from_csv():
    csv_path = DATA_DIR / "additional_robustness_curves.csv"
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["gamma"] = float(row["gamma"])
            row["fitness"] = float(row["fitness"])
            row["p1"] = float(row["p1"])
            row["p2"] = float(row["p2"])
            row["tau"] = float(row["tau"])
            rows.append(row)

    panels = [
        ("low_p1", dict(p1=0.020, p2=0.850, tau=0.010), np.linspace(0.02, 0.98, 17), r"$p_1=0.020,\ p_2=0.850,\ \tau=0.010$"),
        ("higher_penalty", dict(p1=0.095, p2=0.745, tau=0.050), np.linspace(0.08, 0.48, 17), r"$p_1=0.095,\ p_2=0.745,\ \tau=0.050$"),
        ("finite_size", dict(p1=P1, p2=P2, tau=TAU), np.linspace(0.05, 0.90, 14), r"default parameters, BA $m=5$"),
    ]
    label_order = {
        "low_p1": ["WS k=6, p_r=1", "BA m=3", "BA m=5"],
        "higher_penalty": ["WS k=6, p_r=1", "BA m=3", "BA m=5"],
        "finite_size": ["BA m=5, N=500", "BA m=5, N=1000", "BA m=5, N=2000"],
    }
    pretty_label = {
        "WS k=6, p_r=1": r"WS $k=6$, $p_r=1$",
        "BA m=3": r"BA $m=3$",
        "BA m=5": r"BA $m=5$",
        "BA m=5, N=500": r"BA $m=5$, $N=500$",
        "BA m=5, N=1000": r"BA $m=5$, $N=1000$",
        "BA m=5, N=2000": r"BA $m=5$, $N=2000$",
    }
    colors = [COL["blue"], COL["orange"], COL["green"]]
    markers = ["o", "s", "^"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9), constrained_layout=True)

    for pi, (ax, (panel, params, gammas, title)) in enumerate(zip(axes, panels)):
        panel_label(ax, f"({chr(97 + pi)})", x=-0.09, y=1.18)
        roots, _ = mf_window(**params)
        if len(roots) >= 2:
            ax.axvspan(roots[0], roots[1], color=COL["light_green"], zorder=0)
            ax.axvline(roots[0], color=COL["gray"], ls=":", lw=1.1)
            ax.axvline(roots[1], color=COL["gray"], ls=":", lw=1.1)
        ax.axhline(0, color=COL["black"], ls="--", lw=1.0)
        for li, label in enumerate(label_order[panel]):
            means = []
            sems = []
            for gamma in gammas:
                vals = [
                    r["fitness"]
                    for r in rows
                    if r["label"] == label
                    and abs(r["gamma"] - float(gamma)) < 1e-10
                    and abs(r["p1"] - params["p1"]) < 1e-12
                    and abs(r["p2"] - params["p2"]) < 1e-12
                    and abs(r["tau"] - params["tau"]) < 1e-12
                ]
                vals = np.array(vals, dtype=float)
                means.append(vals.mean())
                sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            means = np.array(means)
            sems = np.array(sems)
            if panel == "finite_size":
                ax.plot(
                    gammas,
                    (means > 0).astype(float),
                    color=colors[li],
                    marker=markers[li],
                    ms=3.6,
                    lw=1.7,
                    label=pretty_label[label],
                )
            else:
                ax.errorbar(
                    gammas,
                    means * 1e4,
                    yerr=sems * 1e4,
                    color=colors[li],
                    marker=markers[li],
                    ms=3.6,
                    lw=1.7,
                    capsize=2.0,
                    label=pretty_label[label],
                )
        ax.set_title(title, fontsize=12.6, pad=4)
        ax.set_xlabel(r"$\gamma$")
        if panel == "finite_size":
            ax.axhline(1, color=COL["gray"], ls="--", lw=0.9)
            ax.set_ylim(-0.12, 1.18)
            ax.set_ylabel("Paradox indicator")
        else:
            ax.set_ylabel(r"Average fitness $(\times10^{-4})$")
        ax.legend(loc="best", framealpha=0.92)
        finish_axes(ax)
    savefig(fig, "Fig7_additional_robustness.png")


def main():
    set_style()
    data = load_or_compute_data()
    make_fig1()
    make_fig2(data)
    make_fig3()
    make_fig4(data)
    make_fig5()
    make_fig6(data)
    make_fig7_from_csv()
    print("Figures written to figures/")


if __name__ == "__main__":
    main()
