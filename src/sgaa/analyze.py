"""
sgaa.analyze
============
Reads the saved experiment results and prints a summary of the headline
numbers used in the paper, in a copy-paste-friendly form.
"""

import json
import os

import numpy as np

from . import paths

RES = str(paths.RESULTS_DIR)


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def main():
    main = load("main_comparison.json")
    depth = load("depth_sweep.json")
    schemes = ["sigmoid_all", "sg_alt", "gs_alt", "relu", "tanh"]

    print("=" * 90)
    print("TABLE 1  Main comparison (W=256, L=4, 10 epochs, 3 seeds)")
    print("=" * 90)
    print(f"{'scheme':12s} {'best acc':>10s} {'final acc':>10s} "
          f"{'grad ratio init':>16s} {'grad ratio final':>16s} "
          f"{'act_max final':>14s} {'prod_dev@init':>14s}")
    for s in schemes:
        r = main[s]
        diag = r["diag"]
        from collections import defaultdict
        by = defaultdict(list)
        for seed, ep, dd in diag:
            by[ep].append(dd)
        eps = sorted(by.keys())
        ini, fin = eps[0], eps[-1]
        gr_i = np.mean([by[ini][i]["z_grad_ratio_last_first"] for i in range(len(by[ini]))])
        gr_f = np.mean([by[fin][i]["z_grad_ratio_last_first"] for i in range(len(by[fin]))])
        act_f = np.mean([by[fin][i]["act_max"] for i in range(len(by[fin]))])
        pd_i = np.mean([by[ini][i]["mean_abs_deviation_of_prod_from_1"] for i in range(len(by[ini]))])
        final_acc = np.array(r["test_acc_curves"])[:, -1].mean()
        print(f"{s:12s} {r['best_acc_mean']:10.4f} {final_acc:10.4f} "
              f"{gr_i:16.3e} {gr_f:16.3e} {act_f:14.3e} {pd_i:14.4f}")

    print()
    print("=" * 90)
    print("TABLE 2  Depth sweep (W=128)  test accuracy / grad-ratio / act_max")
    print("=" * 90)
    for dk in ["sigmoid_all", "sg_alt", "relu", "tanh"]:
        depths = sorted(int(k) for k in depth[dk].keys())
        print(f"-- {dk} --")
        for dd in depths:
            e = depth[dk][str(dd)]
            f = e["final"]
            print(f"   L={dd:2d}  acc={e['test_acc']:.4f}  "
                  f"grad_ratio={f['z_grad_ratio_last_first']:.3e}  "
                  f"act_max={f['act_max']:.3e}  act_nan={f['act_has_nan']}  "
                  f"loss_hist_len={e['epochs']}")


if __name__ == "__main__":
    main()
