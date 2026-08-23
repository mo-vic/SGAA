"""
sgaa.plotting
=============
Reads the saved experiment results (results/) and produces the figures for the
SGAA research paper.  All figures are written to figures/.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import paths

RES = str(paths.RESULTS_DIR)
FIG = str(paths.FIGURES_DIR)
os.makedirs(FIG, exist_ok=True)

SCHEMES = ["sigmoid_all", "sg_alt", "gs_alt", "relu", "tanh"]
SHORT = {
    "sigmoid_all": "all-$\\sigma$",
    "sg_alt": "$\\sigma$/G alt",
    "gs_alt": "G/$\\sigma$ alt",
    "relu": "ReLU",
    "tanh": "tanh",
}
COLORS = {
    "sigmoid_all": "#1f77b4",
    "sg_alt": "#d62728",
    "gs_alt": "#8c564b",
    "relu": "#2ca02c",
    "tanh": "#9467bd",
}
LS = {"sigmoid_all": "-", "sg_alt": "-", "gs_alt": "--", "relu": "-", "tanh": ":"}


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.autolayout": True,
})


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def save(fig, fname):
    p = os.path.join(FIG, fname)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


# ----------------------------------------------------------------------
def fig1_activations():
    d = load("activation_surfaces.json")
    x = np.array(d["x"])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax[0].plot(x, d["sigma"], color=COLORS["sigmoid_all"], lw=1.8, label=r"$\sigma(x)$")
    ax[0].plot(x, d["G"], color=COLORS["sg_alt"], lw=1.8, label=r"$G(x)$")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].axvline(0, color="k", lw=0.6)
    ax[0].set_xlabel("$x$")
    ax[0].set_title(r"$\sigma(x)$ and $G(x)=2\sinh x+2x$", fontsize=9)
    ax[0].legend(loc="best")
    ax[0].set_ylim(-40, 40)

    ax[1].plot(x, d["sigma_prime"], color=COLORS["sigmoid_all"], lw=1.8,
               label=r"$\sigma'(x)$")
    ax[1].plot(x, d["G_prime"], color=COLORS["sg_alt"], lw=1.8, label=r"$G'(x)=1/\sigma'(x)$")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("$x$")
    ax[1].set_title(r"$\sigma'(x)$ and $G'(x)$ (log)", fontsize=9)
    ax[1].legend(loc="best")
    ax[1].set_ylim(1e-4, 1e4)
    save(fig, "fig1_activations.png")


def fig2_identity():
    d = load("activation_surfaces.json")
    x = np.array(d["x"])
    sp = np.array(d["sigma_prime"]); gp = np.array(d["G_prime"])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    prod = sp * gp
    ax[0].plot(x, prod, color="#444444", lw=1.8)
    ax[0].set_ylim(-0.05, 1.15)
    ax[0].set_xlabel("$z$")
    ax[0].set_ylabel(r"$\sigma'(z)\cdot G'(z)$")
    ax[0].set_title(r"Exact identity $\sigma'(z)G'(z)=1$", fontsize=9)

    # product as a function of the mismatch Delta = z_i - z_{i+1}
    zc = np.array([0.0, 1.0, 2.0, 4.0])
    z_ = np.linspace(-6, 6, 401)
    for z0 in zc:
        s = 1.0 / (1.0 + np.exp(-z0))
        # product = sigma'(z0+u)/sigma'(z0) with u = z_i - z_{i+1}
        f = (1.0 / (1.0 + np.exp(-(z0 + z_)))) * (1.0 / (1 + np.exp(z0 + z_)))
        denom = s * (1 - s)
        ratio = f / denom
        ax[1].plot(z_, ratio, lw=1.6,
                   label=fr"$z_{{\rm ref}}$={z0:g}", color=COLORS["sg_alt"])
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel(r"$\Delta z = z_i - z_{i+1}$")
    ax[1].set_ylabel(r"$\sigma'(z_i)G'(z_{i+1})$")
    ax[1].set_title(r"Product deviates from 1 only when $z_i\neq z_{i+1}$", fontsize=9)
    ax[1].legend(loc="best")
    save(fig, "fig2_derivative_identity.png")


def fig3_training():
    d = load("main_comparison.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for s in SCHEMES:
        r = d[s]
        curves = np.array(r["test_acc_curves"])   # (seeds, epochs)
        e = np.arange(1, curves.shape[1] + 1)
        m = curves.mean(0)
        sd = curves.std(0)
        ax[0].plot(e, m, color=COLORS[s], ls=LS[s], lw=1.6, label=SHORT[s])
        ax[0].fill_between(e, m - sd, m + sd, color=COLORS[s], alpha=0.15)
        lo = np.array(r["train_loss_curves"])
        m2 = lo.mean(0); sd2 = lo.std(0)
        e2 = np.arange(1, lo.shape[1] + 1)
        ax[1].plot(e2, m2, color=COLORS[s], ls=LS[s], lw=1.6, label=SHORT[s])
        ax[1].fill_between(e2, m2 - sd2, m2 + sd2, color=COLORS[s], alpha=0.15)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("test accuracy")
    ax[0].set_title("Test accuracy vs epoch", fontsize=9); ax[0].legend(loc="lower right")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("train loss")
    ax[1].set_title("Training loss vs epoch", fontsize=9)
    save(fig, "fig3_training.png")


def fig4_accuracy_bar():
    d = load("main_comparison.json")
    good = ["sigmoid_all", "sg_alt", "relu", "tanh"]
    names = [SHORT[s] for s in good]
    means = [d[s]["best_acc_mean"] for s in good]
    stds = [d[s]["best_acc_std"] for s in good]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9),
                            gridspec_kw={"width_ratios": [2.0, 1.0]})
    ax = axs[0]
    bars = ax.bar(names, means, yerr=stds, capsize=3,
                  color=[COLORS[s] for s in good], alpha=0.85, edgecolor="k", lw=0.5)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.0015, f"{m:.3f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("best test accuracy")
    ax.set_ylim(0.962, 0.985)
    ax.set_title("Working schemes (10 epochs, 3 seeds)", fontsize=9)

    ax = axs[1]
    s = "gs_alt"
    ax.bar([SHORT[s]], [d[s]["best_acc_mean"]],
           yerr=[d[s]["best_acc_std"]], capsize=3, color=COLORS[s],
           alpha=0.85, edgecolor="k", lw=0.5)
    ax.text(0, d[s]["best_acc_mean"] + 0.02, f"{d[s]['best_acc_mean']:.3f}",
            ha="center", va="bottom", fontsize=7)
    ax.set_ylim(-0.05, 0.42)
    ax.set_title("$G/\\sigma$ alt collapses", fontsize=9)
    ax.set_xlabel("")
    save(fig, "fig4_accuracy_bar.png")


def fig5_depth_grad():
    d = load("depth_sweep.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for s in ["sigmoid_all", "sg_alt", "relu", "tanh"]:
        depths = [int(k) for k in d[s].keys()]
        depths.sort()
        ratio = [d[s][str(dd)]["final"]["z_grad_ratio_last_first"] for dd in depths]
        acc = [d[s][str(dd)]["test_acc"] for dd in depths]
        ratio = np.array(ratio, dtype=float)
        ratio = np.nan_to_num(ratio, nan=1.0, posinf=1e18, neginf=1e-18)
        ratio = np.clip(ratio, 1e-12, 1e18)
        ax[0].plot(depths, np.maximum(ratio, 1e-12), marker="o", ms=3,
                   color=COLORS[s], ls=LS[s], lw=1.4, label=SHORT[s])
        ax[1].plot(depths, np.nan_to_num(acc, nan=0.0), marker="o", ms=3,
                   color=COLORS[s], ls=LS[s], lw=1.4, label=SHORT[s])
    ax[0].set_yscale("log")
    ax[0].axhline(1, color="k", lw=0.9, ls="--")
    ax[0].set_xlabel("number of hidden layers"); ax[0].set_ylabel("$\\|\\nabla_{z_N}\\|/\\|\\nabla_{z_1}\\|$")
    ax[0].set_title("Gradient norm ratio (output/input side)", fontsize=9)
    ax[0].legend(loc="upper left", fontsize=7)
    ax[1].set_xlabel("number of hidden layers"); ax[1].set_ylabel("test accuracy")
    ax[1].set_title("Test accuracy vs depth", fontsize=9)
    save(fig, "fig5_depth_grad.png")


def fig6_prod_evolution():
    d = load("main_comparison.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    from collections import defaultdict
    for s in ["sigmoid_all", "sg_alt", "gs_alt"]:
        diag = d[s]["diag"]
        by_ep = defaultdict(list)
        for seed, ep, dd in diag:
            by_ep[ep].append(dd)
        eps = sorted(by_ep.keys())
        md = [np.mean([by_ep[e][i]["mean_abs_deviation_of_prod_from_1"]
                       for i in range(len(by_ep[e]))]) for e in eps]
        ax[0].plot(eps, md, marker="o", ms=4, color=COLORS[s], ls=LS[s], lw=1.6, label=SHORT[s])
    for s in ["sigmoid_all", "sg_alt"]:
        diag = d[s]["diag"]
        by_ep = defaultdict(list)
        for seed, ep, dd in diag:
            by_ep[ep].append(dd)
        eps = sorted(by_ep.keys())
        gr = [np.nanmean([by_ep[e][i]["z_grad_ratio_last_first"]
                          for i in range(len(by_ep[e]))]) for e in eps]
        ax[1].plot(eps, gr, marker="s", ms=4, color=COLORS[s], ls=LS[s], lw=1.6, label=SHORT[s])
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel(r"mean $|\,\sigma'(z_i)G'(z_{i+1})-1\,|$")
    ax[0].set_title("Deviation of adjacent derivative-product from 1", fontsize=9)
    ax[0].legend(loc="best")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel(r"$\|\nabla_{z_N}\|/\|\nabla_{z_1}\|$")
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].set_yscale("log")
    ax[1].set_title("Gradient-norm ratio during training", fontsize=9)
    ax[1].legend(loc="best")
    save(fig, "fig6_prod_evolution.png")


def fig7_per_layer_grad():
    d = load("depth_sweep.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
    depth = 6
    for ax, stage in zip(axs, ["init", "final"]):
        for s in ["sigmoid_all", "sg_alt", "relu", "tanh"]:
            z = np.array(d[s][str(depth)][stage]["z_grad_norms"], dtype=float)
            z = np.nan_to_num(z, nan=1e-12, posinf=1e18, neginf=1e-18)
            ax.plot(range(len(z)), np.maximum(z, 1e-12), marker="o", ms=3,
                    color=COLORS[s], ls=LS[s], lw=1.4, label=SHORT[s])
        ax.set_yscale("log")
        ax.set_xlabel("layer index (input->output)")
        ax.set_ylabel(r"$\|\partial L/\partial z_\ell\|$")
        ax.set_title(f"Per-layer pre-activation grad norm ({stage})", fontsize=9)
    axs[0].legend(loc="best", fontsize=7)
    save(fig, "fig7_per_layer_grad.png")


def make_all():
    fig1_activations()
    fig2_identity()
    fig3_training()
    fig4_accuracy_bar()
    fig5_depth_grad()
    fig6_prod_evolution()
    fig7_per_layer_grad()
    print("All figures generated.")


if __name__ == "__main__":
    make_all()
