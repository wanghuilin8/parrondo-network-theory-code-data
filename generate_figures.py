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
from scipy.optimize import brentq, fsolve


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
            "axes.labelsize": 11,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "grid.color": "#B8B8B8",
            "grid.alpha": 0.25,
            "savefig.dpi": 220,
        }
    )


def panel_label(ax, label):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=COL["black"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
    )


def finish_axes(ax, grid=True):
    if grid:
        ax.grid(True, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, name):
    fig.savefig(FIG_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def solve_mf(gamma, p1=P1, p2=P2, tau=TAU):
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


def f_mix(gamma, p1=P1, p2=P2, tau=TAU):
    pi = solve_mf(gamma, p1, p2, tau)
    return gamma * (-tau) + (1 - gamma) * (
        (2 * p2 - 1) - 2 * (p2 - p1) * pi[0]
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


def mc_run_fast(nbrs, gamma, p1=P1, p2=P2, tau=TAU, t_steps=T_MC, reps=40, seed=0):
    rng = np.random.default_rng(seed)
    n = len(nbrs)
    max_deg = max(len(nb) for nb in nbrs)
    nb_arr = np.zeros((n, max_deg), dtype=np.int32)
    deg_arr = np.array([len(nb) for nb in nbrs], dtype=np.int32)
    for i, nb in enumerate(nbrs):
        nb_arr[i, : len(nb)] = nb

    cap_sum = 0.0
    for _ in range(reps):
        cap = np.zeros(n, dtype=np.float64)
        nodes = rng.integers(0, n, size=t_steps)
        game_rands = rng.random(t_steps)
        contest_rands = rng.random(t_steps)
        penalty_rands = rng.random(t_steps)
        play_rands = rng.random(t_steps)
        nb_rands = rng.integers(0, max_deg, size=t_steps)

        for t in range(t_steps):
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
    return cap_sum / (n * reps * t_steps)


def pair_equations(x, k, gamma, p1=P1, p2=P2, tau=TAU):
    p = np.zeros((3, 3), dtype=float)
    p[0, 0], p[1, 1], p[2, 2] = x[0], x[1], x[2]
    p[0, 1] = p[1, 0] = x[3]
    p[0, 2] = p[2, 0] = x[4]
    p[1, 2] = p[2, 1] = x[5]
    pi = p.sum(axis=1)

    def w_b(a, b):
        if a == 0:
            return p1 if b == 1 else (1 - p1 if b == 2 else 0.0)
        if a == 1:
            return (1 - p2) if b == 0 else (p2 if b == 2 else 0.0)
        return p2 if b == 0 else (1 - p2 if b == 1 else 0.0)

    eqs = np.zeros(6, dtype=float)
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    for ei, (a, b) in enumerate(pairs):
        pab = p[a, b]
        flux = 0.0
        au, ad = (a + 1) % 3, (a - 1) % 3
        bu, bd = (b + 1) % 3, (b - 1) % 3
        for a2 in range(3):
            if a2 != a:
                flux += (1 - gamma) * w_b(a2, a) * p[a2, b]
        flux -= (1 - gamma) * sum(w_b(a, a2) for a2 in range(3) if a2 != a) * pab
        for b2 in range(3):
            if b2 != b:
                flux += (1 - gamma) * w_b(b2, b) * p[a, b2]
        flux -= (1 - gamma) * sum(w_b(b, b2) for b2 in range(3) if b2 != b) * pab
        flux += gamma / k * (0.5 * p[ad, bu] + 0.5 * (1 - tau) * p[au, bd])
        flux -= gamma / k * pab
        if pi[ad] > 1e-10 and pi[au] > 1e-10:
            flux += gamma * (k - 1) / k * (
                0.5 * p[ad, b] + 0.5 * (1 - tau) * p[au, b] - pab
            )
        if pi[bd] > 1e-10 and pi[bu] > 1e-10:
            flux += gamma * (k - 1) / k * (
                0.5 * p[a, bd] + 0.5 * (1 - tau) * p[a, bu] - pab
            )
        eqs[ei] = flux
    eqs[5] = x[0] + x[1] + x[2] + 2 * x[3] + 2 * x[4] + 2 * x[5] - 1.0
    return eqs


def solve_pa(k, gamma, p1=P1, p2=P2, tau=TAU, seed=0):
    rng = np.random.default_rng(seed)
    pi_mf = solve_mf(gamma, p1, p2, tau)
    x_mf = np.array([pi_mf[i] * pi_mf[j] for i in range(3) for j in range(i, 3)])
    inits = [x_mf]
    for _ in range(14):
        r = np.abs(rng.normal(size=6)) + 0.01
        r /= r[0] + r[1] + r[2] + 2 * r[3] + 2 * r[4] + 2 * r[5]
        inits.append(r)

    best_x = None
    best_err = np.inf
    for x0 in inits:
        try:
            sol, info, ier, _ = fsolve(
                pair_equations,
                x0,
                args=(k, gamma, p1, p2, tau),
                full_output=True,
                xtol=1e-12,
            )
            err = float(np.max(np.abs(info["fvec"])))
            if ier == 1 and err < best_err and np.all(sol >= -1e-8):
                best_x = sol.copy()
                best_err = err
        except Exception:
            continue
    if best_x is None or best_err > 1e-5:
        return np.nan
    return best_x[0] + best_x[3] + best_x[4]


def compute_window_peak(tau, p1=P1, p2=P2, n=600):
    gammas = np.linspace(0.01, 0.98, n)
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


def load_or_compute_data():
    if CACHE_PATH.exists():
        print(f"Loading cached data from {CACHE_PATH}")
        return dict(np.load(CACHE_PATH, allow_pickle=True))

    print("Computing Monte Carlo data for figure generation...")
    data = {}

    k_list = np.array([2, 6, 14])
    gammas2 = np.linspace(0.02, 0.95, 18)
    data["fig2_k"] = k_list
    data["fig2_g"] = gammas2
    for k in k_list:
        nbrs = build_ring(N_MC, int(k))
        mf = []
        pa = []
        mc = []
        for gi, gamma in enumerate(gammas2):
            mf.append(solve_mf(gamma)[0])
            pa.append(solve_pa(int(k), gamma, seed=BASE_SEED + int(k) * 100 + gi))
            mc.append(
                mc_run_fast(
                    nbrs,
                    gamma,
                    reps=40,
                    seed=BASE_SEED + int(k) * 1000 + gi,
                )
            )
        data[f"fig2_mf_{k}"] = np.array(mf)
        data[f"fig2_pa_{k}"] = np.array(pa)
        data[f"fig2_mc_{k}"] = np.array(mc)
        print(f"  Fig. 2 k={k} done")

    pr_list = np.array([0.0, 0.05, 0.15, 0.30, 0.60, 1.0])
    gammas4 = np.linspace(0.05, 0.90, 12)
    fg4 = np.zeros((len(pr_list), len(gammas4)))
    for i, pr in enumerate(pr_list):
        nbrs = build_ws(N_MC, 6, float(pr), seed=BASE_SEED + i)
        for j, gamma in enumerate(gammas4):
            fg4[i, j] = mc_run_fast(
                nbrs,
                gamma,
                reps=40,
                seed=BASE_SEED + 40000 + i * 100 + j,
            )
        print(f"  Fig. 4 p_r={pr:.2f} done")
    data["fig4_pr"] = pr_list
    data["fig4_g"] = gammas4
    data["fig4_f"] = fg4

    gammas6 = np.linspace(0.05, 0.90, 14)
    labels6 = np.array(["WS k=6", "BA m=2", "BA m=3", "BA m=5"])
    f6 = np.zeros((len(labels6), len(gammas6)))
    networks = [
        build_ring(N_MC, 6),
        build_ba(N_MC, 2, seed=BASE_SEED + 62),
        build_ba(N_MC, 3, seed=BASE_SEED + 63),
        build_ba(N_MC, 5, seed=BASE_SEED + 65),
    ]
    for i, nbrs in enumerate(networks):
        for j, gamma in enumerate(gammas6):
            f6[i, j] = mc_run_fast(
                nbrs,
                gamma,
                reps=50,
                seed=BASE_SEED + 60000 + i * 100 + j,
            )
        print(f"  Fig. 6 {labels6[i]} done")
    data["fig6_g"] = gammas6
    data["fig6_labels"] = labels6
    data["fig6_f"] = f6

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


def make_fig1():
    fig = plt.figure(figsize=(11.6, 5.8))
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

    panel_label(ax_net, "(a)")
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
    ax_net.text(0, 1.45, "Network state", ha="center", fontsize=12, fontweight="bold", color=COL["black"])
    ax_net.text(1.02, 0.10, "Game A\nneighbor contest", ha="left", va="center", fontsize=8.5, color=COL["orange"], fontweight="bold")
    ax_net.text(-1.48, -0.64, "Game B\nlocal state", ha="left", va="center", fontsize=8.5, color=COL["purple"], fontweight="bold")
    ax_net.text(0, -1.45, r"state $s_i=c_i\,\mathrm{mod}\,3$", ha="center", fontsize=9.5, color=COL["gray"])
    legend_x = [-0.78, 0.0, 0.78]
    legend_labels = [r"$s=0$ trap", r"$s=1$", r"$s=2$"]
    for x, state, label in zip(legend_x, [0, 1, 2], legend_labels):
        ax_net.add_patch(Circle((x, -1.78), 0.105, color=STATE[state], ec="white", lw=1.0))
        ax_net.text(x, -2.02, label, ha="center", fontsize=8.5, color=STATE[state], fontweight="bold")

    panel_label(ax_rules, "(b)")
    ax_rules.set_xlim(0, 10)
    ax_rules.set_ylim(0, 4.2)
    rounded_box(ax_rules, (0.35, 0.40), 4.45, 3.20, COL["light_orange"], COL["orange"])
    rounded_box(ax_rules, (5.20, 0.40), 4.45, 3.20, COL["light_purple"], COL["purple"])
    ax_rules.text(2.58, 3.27, "Game A", ha="center", fontsize=12, fontweight="bold", color=COL["orange"])
    ax_rules.text(2.58, 3.00, "competition + penalty", ha="center", fontsize=8.8, color=COL["orange"])
    ax_rules.add_patch(Circle((1.55, 2.20), 0.34, color=COL["orange"], ec="white", lw=1.3))
    ax_rules.add_patch(Circle((3.55, 2.20), 0.34, color=COL["orange"], ec="white", lw=1.3))
    ax_rules.text(1.55, 2.20, r"$i$", ha="center", va="center", fontsize=13, color="white", fontweight="bold")
    ax_rules.text(3.55, 2.20, r"$j$", ha="center", va="center", fontsize=13, color="white", fontweight="bold")
    arrow(ax_rules, (1.96, 2.20), (3.14, 2.20), COL["red"], rad=0.05, lw=1.6)
    arrow(ax_rules, (3.14, 2.00), (1.96, 2.00), COL["red"], rad=0.05, lw=1.6)
    ax_rules.text(2.55, 2.60, r"$\pm 1$ with prob. $1/2$", ha="center", fontsize=9.4, color=COL["red"])
    arrow(ax_rules, (1.55, 1.75), (1.55, 1.20), COL["purple"], lw=1.6)
    ax_rules.text(2.70, 1.43, r"penalty $-1$ with prob. $\tau$", ha="center", fontsize=9.0, color=COL["purple"])
    ax_rules.text(2.58, 0.78, r"$\langle\Delta c_i\rangle_A=-\tau$", ha="center", fontsize=10.5, color=COL["red"], fontweight="bold")

    ax_rules.text(7.42, 3.27, "Game B", ha="center", fontsize=12, fontweight="bold", color=COL["purple"])
    ax_rules.text(7.42, 3.00, "capital-dependent local game", ha="center", fontsize=8.8, color=COL["purple"])
    ax_rules.add_patch(Circle((6.35, 2.12), 0.34, color=STATE[0], ec="white", lw=1.3))
    ax_rules.text(6.35, 2.12, r"$s=0$", ha="center", va="center", fontsize=9.2, color="white", fontweight="bold")
    ax_rules.text(6.35, 1.55, r"win $p_1=0.095$", ha="center", fontsize=8.8, color=STATE[0])
    ax_rules.text(6.35, 1.17, "low-win trap", ha="center", fontsize=8.2, color=COL["gray"])
    ax_rules.add_patch(Circle((8.38, 2.12), 0.38, color=STATE[2], ec="white", lw=1.3))
    ax_rules.text(8.38, 2.12, r"$s=1,2$", ha="center", va="center", fontsize=8.6, color="white", fontweight="bold")
    ax_rules.text(8.38, 1.55, r"win $p_2=0.745$", ha="center", fontsize=8.8, color=STATE[2])
    ax_rules.text(8.38, 1.17, "high-win states", ha="center", fontsize=8.2, color=COL["gray"])
    arrow(ax_rules, (7.05, 2.12), (7.72, 2.12), COL["gray"], rad=0.0, lw=1.2, ms=11)
    ax_rules.text(7.42, 0.82, "mixed play combines A and B", ha="center", fontsize=8.8, color=COL["black"])

    panel_label(ax_trans, "(c)")
    ax_trans.set_xlim(-0.25, 3.25)
    ax_trans.set_ylim(-0.10, 3.05)
    ax_trans.set_aspect("equal")
    pos = {0: (1.5, 2.35), 1: (2.55, 0.65), 2: (0.45, 0.65)}
    for state, (x, y) in pos.items():
        ax_trans.add_patch(Circle((x, y), 0.34, color=STATE[state], ec="white", lw=1.5, zorder=3))
        text = r"$s=0$" + "\ntrap" if state == 0 else rf"$s={state}$"
        ax_trans.text(x, y, text, ha="center", va="center", fontsize=9.0, color="white", fontweight="bold", zorder=4)
    arrow(ax_trans, (1.72, 2.08), (2.35, 0.96), STATE[0], rad=-0.10)
    ax_trans.text(2.30, 1.75, r"$p_1$", fontsize=9.5, color=STATE[0], fontweight="bold")
    arrow(ax_trans, (2.28, 0.50), (0.72, 0.50), COL["gray"], rad=-0.10)
    ax_trans.text(1.50, 0.15, r"$1-p_2$", fontsize=8.8, color=COL["gray"], ha="center")
    arrow(ax_trans, (0.66, 0.95), (1.28, 2.08), STATE[2], rad=-0.10)
    ax_trans.text(0.60, 1.78, r"$p_2$", fontsize=9.5, color=STATE[2], fontweight="bold")
    arrow(ax_trans, (2.34, 0.94), (1.72, 2.07), COL["gray"], rad=0.15)
    ax_trans.text(2.75, 1.58, r"$1-p_2$", fontsize=8.8, color=COL["gray"])
    arrow(ax_trans, (0.70, 0.49), (2.30, 0.49), STATE[1], rad=0.12)
    ax_trans.text(1.50, 0.78, r"$p_2$", fontsize=9.5, color=STATE[1], fontweight="bold", ha="center")
    arrow(ax_trans, (1.26, 2.06), (0.66, 0.95), COL["gray"], rad=0.15)
    ax_trans.text(0.05, 1.58, r"$1-p_1$", fontsize=8.8, color=COL["gray"])
    ax_trans.text(1.5, 2.95, "Game B transition rates", ha="center", fontsize=10.5, color=COL["black"], fontweight="bold")

    ax_cond.set_xlim(0, 1)
    ax_cond.set_ylim(0, 1)
    rounded_box(ax_cond, (0.04, 0.18), 0.92, 0.64, COL["light_green"], COL["green"], lw=1.8, r=0.06)
    ax_cond.text(0.50, 0.72, "Necessary condition", ha="center", fontsize=11.5, fontweight="bold", color=COL["green"])
    ax_cond.text(
        0.50,
        0.50,
        r"$\pi_0^{\rm mix}<\pi_0^*=\dfrac{2p_2-1}{2(p_2-p_1)}$",
        ha="center",
        va="center",
        fontsize=13.0,
        color="#14532D",
    )
    ax_cond.text(0.50, 0.33, r"trap occupancy below threshold", ha="center", fontsize=8.8, color=COL["gray"])
    ax_cond.text(0.50, 0.23, r"$\pi_0^*$ independent of $\gamma$, $\tau$, topology", ha="center", fontsize=8.0, color=COL["gray"])

    savefig(fig, "Fig1_model_schematic.png")


def make_fig2(data):
    k_list = data["fig2_k"]
    gammas = data["fig2_g"]
    colors = {2: COL["blue"], 6: COL["orange"], 14: COL["green"]}
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), constrained_layout=True)

    ax = axes[0]
    panel_label(ax, "(a)")
    ax.axhline(PI0_STAR, color=COL["black"], ls=":", lw=1.5)
    g_fine = np.linspace(0.01, 0.98, 220)
    for k in k_list:
        k = int(k)
        ax.plot(g_fine, [solve_mf(g)[0] for g in g_fine], color=colors[k], lw=1.8, alpha=0.95)
        pa = data[f"fig2_pa_{k}"]
        valid = ~np.isnan(pa)
        ax.plot(gammas[valid], pa[valid], "--", color=colors[k], lw=1.5)
    handles = [
        Line2D([0], [0], color=COL["black"], ls=":", lw=1.5, label=rf"$\pi_0^*={PI0_STAR:.3f}$"),
        Line2D([0], [0], color=COL["gray"], lw=1.8, label="MF"),
        Line2D([0], [0], color=COL["gray"], ls="--", lw=1.5, label="PA"),
        *[Line2D([0], [0], color=colors[int(k)], lw=1.8, label=rf"$k={int(k)}$") for k in k_list],
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.92)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Trap occupancy $\pi_0$")
    ax.set_ylim(0.325, 0.395)
    finish_axes(ax)

    ax = axes[1]
    panel_label(ax, "(b)")
    ax.axhline(0, color=COL["black"], ls="--", lw=1.0)
    for k in k_list:
        k = int(k)
        delta = data[f"fig2_pa_{k}"] - data[f"fig2_mf_{k}"]
        valid = ~np.isnan(delta)
        ax.plot(gammas[valid], delta[valid] * 1e3, color=colors[k], lw=1.7, marker="s", ms=3.6, label=rf"$k={k}$")
    ax.legend(loc="upper right", framealpha=0.92)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"PA correction $\Delta\pi_0$ $(\times10^{-3})$")
    finish_axes(ax)

    ax = axes[2]
    panel_label(ax, "(c)")
    ax.axhline(0, color=COL["black"], ls="--", lw=1.0)
    ax.axvspan(0.10, 0.60, color=COL["light_green"], zorder=0)
    ax.axvline(0.10, color=COL["gray"], ls=":", lw=1.1)
    ax.axvline(0.60, color=COL["gray"], ls=":", lw=1.1)
    for k in k_list:
        k = int(k)
        ax.plot(gammas, data[f"fig2_mc_{k}"] * 1e4, color=colors[k], lw=1.7, marker="o", ms=3.6, label=rf"$k={k}$")
    ax.legend(loc="upper right", framealpha=0.92)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Average fitness $(\times10^{-4})$")
    finish_axes(ax)
    savefig(fig, "Fig2_MF_PA_MC_comparison.png")


def make_fig3():
    tau_arr = np.linspace(0.002, 0.085, 150)
    gl = np.full_like(tau_arr, np.nan)
    gh = np.full_like(tau_arr, np.nan)
    peaks = np.full_like(tau_arr, np.nan)
    for i, tau in enumerate(tau_arr):
        gl[i], gh[i], peaks[i] = compute_window_peak(tau)
    ok = ~np.isnan(peaks) & (peaks > 0)
    pos = np.where(ok)[0]
    tc = tau_arr[pos[-1]]
    gc = (gl[pos[-1]] + gh[pos[-1]]) / 2
    width = np.where(ok, gh - gl, 0)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), constrained_layout=True)
    ax = axes[0]
    panel_label(ax, "(a)")
    ax.fill_between(tau_arr[ok], gl[ok], gh[ok], color=COL["light_green"], label="Paradox window")
    ax.plot(tau_arr[ok], gh[ok], color=COL["red"], lw=1.9, label=r"$\gamma_{\rm high}$")
    ax.plot(tau_arr[ok], gl[ok], color=COL["blue"], lw=1.9, label=r"$\gamma_{\rm low}$")
    ax.scatter([tc], [gc], color=COL["black"], s=90, marker="*", zorder=5, label=rf"$\tau_c={tc:.3f}$")
    ax.axvline(tc, color=COL["black"], ls=":", lw=1.1)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_xlim(0, 0.085)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", framealpha=0.92)
    finish_axes(ax)

    ax = axes[1]
    panel_label(ax, "(b)")
    ax.fill_between(tau_arr[ok], 0, width[ok], color=COL["light_green"])
    ax.plot(tau_arr[ok], width[ok], color=COL["green"], lw=1.9, label=r"$\Delta\gamma(\tau)$")
    ax.scatter([tc], [0], color=COL["black"], s=45, zorder=5)
    ax.axvline(tc, color=COL["black"], ls="--", lw=1.3, label=rf"$\tau_c={tc:.3f}$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"Window width $\Delta\gamma$")
    ax.set_xlim(0, 0.085)
    ax.set_ylim(-0.04, 1.0)
    ax.legend(loc="upper right", framealpha=0.92)
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
        ymax = max(ymax, np.nanmax(vals * 1e4))
        is_crit = abs(tau - tc) < 0.003
        ax.plot(
            gs,
            vals * 1e4,
            color=COL["red"] if is_crit else cmap(0.28 + 0.62 * i / (len(taus) - 1)),
            lw=2.4 if is_crit else 1.8,
            label=rf"$\tau_c={tau:.3f}$" if is_crit else rf"$\tau={tau:.3f}$",
        )
    ax.axhline(0, color=COL["black"], ls="--", lw=1.0)
    ax.scatter([gc], [0], color=COL["black"], s=45, zorder=5, label=rf"$\gamma_c={gc:.3f}$")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Mixed fitness $(\times10^{-4})$")
    ax.set_xlim(0.05, 0.90)
    ax.set_ylim(-0.5, ymax * 1.45)
    ax.legend(loc="upper right", framealpha=0.92, ncol=2)
    finish_axes(ax)
    savefig(fig, "Fig3_critical_point.png")


def make_fig4(data):
    pr = data["fig4_pr"]
    gammas = data["fig4_g"]
    fg = data["fig4_f"]
    pr_grid, g_grid = np.meshgrid(pr, gammas, indexing="ij")
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), constrained_layout=True)

    ax = axes[0]
    panel_label(ax, "(a)")
    vm = float(np.abs(fg).max() * 1e4)
    cf = ax.contourf(g_grid, pr_grid, fg * 1e4, levels=17, cmap="RdYlGn", vmin=-vm, vmax=vm, alpha=0.88)
    try:
        cs = ax.contour(g_grid, pr_grid, fg, levels=[0], colors=COL["black"], linewidths=1.5)
        ax.clabel(cs, fmt=r"$f=0$", fontsize=8)
    except Exception:
        pass
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"Fitness $(\times10^{-4})$")
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
    finish_axes(ax)

    ax = axes[2]
    panel_label(ax, "(c)")
    ax.axhline(0, color=COL["black"], ls="--", lw=1.0)
    ax.axvspan(0.10, 0.60, color=COL["light_green"], zorder=0)
    ax.axvline(0.10, color=COL["gray"], ls=":", lw=1.1)
    ax.axvline(0.60, color=COL["gray"], ls=":", lw=1.1)
    colors = [COL["blue"], COL["purple"], "#CC79A7", COL["red"], COL["orange"], COL["green"]]
    labels = {
        0.0: r"$p_r=0$ (ring)",
        0.05: r"$p_r=0.05$",
        0.15: r"$p_r=0.15$",
        0.30: r"$p_r=0.30$",
        0.60: r"$p_r=0.60$",
        1.0: r"$p_r=1$ (random)",
    }
    for i, p_r in enumerate(pr):
        ax.plot(gammas, fg[i] * 1e4, color=colors[i], lw=1.6, marker="o", ms=3.3, label=labels[float(p_r)])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, framealpha=0.92)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Average fitness $(\times10^{-4})$")
    finish_axes(ax)
    savefig(fig, "Fig4_WS_robustness.png")


def find_tauc(p1, p2, tau_max_cap=0.15, n_tau=80):
    tmax = min(2 * p2 - 1 - 0.005, tau_max_cap)
    if tmax <= 0.001:
        return np.nan, np.nan
    taus = np.linspace(0.001, tmax, n_tau)
    peaks = np.array([compute_window_peak(t, p1, p2, n=300)[2] for t in taus])
    pos = np.where(peaks > 1e-8)[0]
    if len(pos) == 0 or taus[pos[-1]] > tmax * 0.93:
        return np.nan, np.nan
    tc = taus[pos[-1]]
    gs = np.linspace(0.01, 0.98, 300)
    vals = np.array([f_mix(g, p1, p2, tc) for g in gs])
    valid = ~np.isnan(vals)
    gc = gs[valid][np.argmax(vals[valid])] if valid.sum() else np.nan
    return tc, gc


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

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), constrained_layout=True)
    ax = axes[0]
    panel_label(ax, "(a)")
    ax.plot(pv1, t1, color=COL["blue"], lw=1.8, marker="o", ms=4, label=rf"$p_1={P1:.3f}$ fixed")
    ax.plot(pv2, t2, color=COL["red"], lw=1.8, marker="s", ms=4, label=rf"$p_2={P2:.3f}$ fixed")
    ax.scatter([P2], [tco], color=COL["black"], s=90, marker="*", zorder=6, label="This system")
    ax.set_xlabel(r"$p_2$ (blue) or $p_1$ (red)")
    ax.set_ylabel(r"Critical penalty $\tau_c$")
    ax.set_xlim(0, 0.82)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower center", framealpha=0.92)
    finish_axes(ax)

    ax = axes[1]
    panel_label(ax, "(b)")
    ax.plot(pv1, g1, color=COL["blue"], lw=1.8, marker="o", ms=4, label=rf"$p_1={P1:.3f}$ fixed")
    ax.plot(pv2, g2, color=COL["red"], lw=1.8, marker="s", ms=4, label=rf"$p_2={P2:.3f}$ fixed")
    ax.scatter([P2], [gco], color=COL["black"], s=90, marker="*", zorder=6, label="This system")
    ax.set_xlabel(r"$p_2$ (blue) or $p_1$ (red)")
    ax.set_ylabel(r"Critical mixing $\gamma_c$")
    ax.set_xlim(0, 0.82)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", framealpha=0.92)
    finish_axes(ax)
    savefig(fig, "Fig5_tauc_scaling.png")


def make_fig6(data):
    gammas = data["fig6_g"]
    labels = [str(x) for x in data["fig6_labels"]]
    f = data["fig6_f"]
    colors = [COL["black"], COL["red"], COL["orange"], COL["blue"]]
    pretty = {
        "WS k=6": r"WS $k=6$ (baseline)",
        "BA m=2": r"BA $m=2$",
        "BA m=3": r"BA $m=3$",
        "BA m=5": r"BA $m=5$",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), constrained_layout=True)

    ax = axes[0]
    panel_label(ax, "(a)")
    ax.axhline(0, color=COL["black"], ls="--", lw=1.0)
    ax.axvspan(0.10, 0.60, color=COL["light_green"], zorder=0)
    ax.axvline(0.10, color=COL["gray"], ls=":", lw=1.1)
    ax.axvline(0.60, color=COL["gray"], ls=":", lw=1.1)
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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, framealpha=0.92)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Average fitness $(\times10^{-4})$")
    finish_axes(ax)

    ax = axes[1]
    panel_label(ax, "(b)")
    ax.axhline(0, color=COL["gray"], ls="--", lw=0.9)
    ax.axhline(1, color=COL["gray"], ls="--", lw=0.9)
    ax.axvspan(0.10, 0.60, color=COL["light_green"], zorder=0, label="Paradox window")
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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2, framealpha=0.92)
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
        ("near_critical", dict(p1=0.095, p2=0.745, tau=0.050), np.linspace(0.08, 0.48, 17), r"$p_1=0.095,\ p_2=0.745,\ \tau=0.050$"),
        ("finite_size", dict(p1=P1, p2=P2, tau=TAU), np.linspace(0.05, 0.90, 14), r"default parameters, BA $m=5$"),
    ]
    label_order = {
        "low_p1": ["WS k=6, p_r=1", "BA m=3", "BA m=5"],
        "near_critical": ["WS k=6, p_r=1", "BA m=3", "BA m=5"],
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
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), constrained_layout=True)

    for pi, (ax, (panel, params, gammas, title)) in enumerate(zip(axes, panels)):
        panel_label(ax, f"({chr(97 + pi)})")
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
        ax.set_title(title, fontsize=9.3)
        ax.set_xlabel(r"$\gamma$")
        if panel == "finite_size":
            ax.axhline(1, color=COL["gray"], ls="--", lw=0.9)
            ax.set_ylim(-0.12, 1.18)
            ax.set_ylabel("Paradox indicator")
        elif pi == 0:
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
