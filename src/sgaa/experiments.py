"""
sgaa.experiments
================
Experiment driver for the SGAA study:
  * main comparison   (all-σ vs alternating σ/G vs ReLU vs tanh)
  * depth sweep       (vanishing / exploding gradients vs depth)
Each run records losses, accuracies, per-layer pre-activation grads and the
derivative products that bear on the SGAA hypothesis.
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from . import paths
from .activations import G_derivative, sigmoid_prime, StableG
from .nets import FeedforwardNet, forward_and_backward, derivative_product_stats

RESULTS_DIR = str(paths.RESULTS_DIR)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def get_loaders(batch_size=256, num_workers=0, seed=0):
    """Precompute normalized MNIST tensors once and wrap in fast TensorDatasets."""
    torch.manual_seed(seed)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))])
    tr = datasets.MNIST(root=str(paths.MNIST_ROOT), train=True, download=False,
                        transform=tf)
    te = datasets.MNIST(root=str(paths.MNIST_ROOT), train=False, download=False,
                        transform=tf)
    # materialise to a single tensor (fast, no per-batch transform cost)
    trx = torch.stack([t[0] for t in tr]).clone()
    trlab = torch.tensor([t[1] for t in tr], dtype=torch.long)
    tex = torch.stack([t[0] for t in te]).clone()
    telab = torch.tensor([t[1] for t in te], dtype=torch.long)
    trd = DataLoader(torch.utils.data.TensorDataset(trx, trlab), batch_size=batch_size,
                     shuffle=True, num_workers=num_workers, drop_last=True)
    ted = DataLoader(torch.utils.data.TensorDataset(tex, telab), batch_size=1024,
                     shuffle=False, num_workers=num_workers)
    return trd, ted


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits, y):
    return (logits.argmax(dim=-1) == y).float().mean().item()


def eval_net(net, loader, criterion, device):
    net.eval()
    loss, acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z, _ = net.forward_pre_acts(x)
            loss += criterion(z, y).item() * x.shape[0]
            acc += accuracy(z, y) * x.shape[0]
            n += x.shape[0]
    net.train()
    return loss / n, acc / n


def train_epoch(net, loader, optimizer, criterion, device):
    net.train()
    tot_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        z, _ = net.forward_pre_acts(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * x.shape[0]
        n += x.shape[0]
    return tot_loss / n


# ----------------------------------------------------------------------
# Gradient / derivative diagnostics on a diagnostic batch
# ----------------------------------------------------------------------
def diagnostics(net, xb, yb, criterion, device):
    """Return dict of gradient-norm and derivative-product diagnostics."""
    net.zero_grad()
    xb = xb.to(device)
    yb = yb.to(device)
    net.eval()
    res = forward_and_backward(net, xb, yb, criterion)
    net.train()

    z_grad = np.array(res["z_grad_norms"])

    prods, prod_stats_list, kinds = derivative_product_stats(res["pre"], net.pattern)
    prod_stats = {st["kind"]: {k: v for k, v in st.items() if k != "kind"}
                  for st in prod_stats_list}
    mean_prod = float(np.mean([st["mean_abs_dev_from_1"] for st in prod_stats_list])
                      ) if prod_stats_list else float("nan")
    mean_sigma_similarity = float(np.mean([st["sigma_prime_similarity"]
                                           for st in prod_stats_list])
                                  ) if prod_stats_list else float("nan")

    act_max = float(max([a.abs().max().item() for a in res["act"]])
                    if res["act"] else 0.0)
    act_has_nan = any(bool(np.isnan(a.cpu().numpy()).any() or np.isinf(a.cpu().numpy()).any())
                      for a in res["act"])

    out = {
        "loss": res["loss"],
        "z_grad_norm_first": float(z_grad[0]) if len(z_grad) else float("nan"),
        "z_grad_norm_last": float(z_grad[-1]) if len(z_grad) else float("nan"),
        "z_grad_ratio_last_first": float(z_grad[-1] / z_grad[0]) if len(z_grad) else float("nan"),
        "z_grad_norms": z_grad.tolist(),
        "w_grad_norms": res["w_grad_norms"],
        "act_max": act_max,
        "act_has_nan": act_has_nan,
        "prod_stats": prod_stats,
        "mean_abs_deviation_of_prod_from_1": mean_prod,
        "mean_sigma_prime_similarity": mean_sigma_similarity,
    }
    return out


# ----------------------------------------------------------------------
# Main comparison
# ----------------------------------------------------------------------
def run_main_comparison(epochs=10, batch_size=256, hidden_sizes=(256, 256, 256, 256),
                        lr=1e-3, seeds=(0, 1, 2), device="cuda"):
    criterion = nn.CrossEntropyLoss()
    schemes = ["sigmoid_all", "sg_alt", "gs_alt", "relu", "tanh"]
    all_res = {s: {} for s in schemes}

    trl, tel = get_loaders(batch_size=batch_size)
    dxb, dyb = next(iter(trl))   # fixed diagnostic batch

    for scheme in schemes:
        accs_best, tr_losses, te_accs, diag = [], [], [], []
        for seed in seeds:
            set_seed(seed)
            net = FeedforwardNet(list(hidden_sizes), scheme).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            tr_losses.append([])
            te_accs.append([])
            best = 0.0
            start = time.time()
            for ep in range(epochs):
                l = train_epoch(net, trl, opt, criterion, device)
                tr_losses[-1].append(l)
                if not np.isfinite(l):
                    break
                _, a = eval_net(net, tel, criterion, device)
                te_accs[-1].append(a)
                best = max(best, a)
                if ep in (0, epochs // 2, epochs - 1):
                    d = diagnostics(net, dxb, dyb, criterion, device)
                    d["epoch"] = ep
                    diag.append((seed, ep, d))
            accs_best.append(best)
        all_res[scheme] = {
            "best_acc_mean": float(np.mean(accs_best)),
            "best_acc_std": float(np.std(accs_best)),
            "train_loss_curves": tr_losses,
            "test_acc_curves": te_accs,
            "diag": diag,
            "epochs": epochs,
            "hidden_sizes": list(hidden_sizes),
            "lr": lr,
        }
        print(f"[main] {scheme:12s} best test acc = "
              f"{np.mean(accs_best):.4f} ± {np.std(accs_best):.4f}")

    path = f"{RESULTS_DIR}/main_comparison.json"
    with open(path, "w") as f:
        json.dump(all_res, f)
    print(f"saved -> {path}")
    return all_res


# ----------------------------------------------------------------------
# Depth sweep
# ----------------------------------------------------------------------
def run_depth_sweep(depths=(2, 4, 6, 10, 16, 24, 32), width=128, epochs=6,
                    batch_size=256, lr=1e-3, seed=0, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    schemes = ["sigmoid_all", "sg_alt", "relu", "tanh"]
    trl, tel = get_loaders(batch_size=batch_size)
    dxb, dyb = next(iter(trl))
    out = {s: {} for s in schemes}

    for scheme in schemes:
        for depth in depths:
            set_seed(seed)
            net = FeedforwardNet([width] * depth, scheme).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            d0 = diagnostics(net, dxb, dyb, criterion, device)
            d0["epoch"] = -1
            d0["depth"] = depth
            loss_hist = []
            for ep in range(epochs):
                l = train_epoch(net, trl, opt, criterion, device)
                loss_hist.append(l)
                if not np.isfinite(l):
                    break
            _, test_acc = eval_net(net, tel, criterion, device)
            d1 = diagnostics(net, dxb, dyb, criterion, device)
            d1["epoch"] = len(loss_hist) - 1
            d1["depth"] = depth
            out[scheme][str(depth)] = {
                "depth": depth,
                "init": d0,
                "final": d1,
                "test_acc": test_acc,
                "loss_history": loss_hist,
                "epochs": len(loss_hist),
            }
            f = d1["z_grad_ratio_last_first"]
            print(f"[depth] {scheme:12s} L={depth:2d}  test_acc={test_acc:.4f}  "
                  f"grad ratio(last/first)@final={f:.3e}  "
                  f"act_max={d1['act_max']:.3e}  act_nan={d1['act_has_nan']}")

    path = f"{RESULTS_DIR}/depth_sweep.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"saved -> {path}")
    return out


# ----------------------------------------------------------------------
# Activation & derivative surface study (theory figure data)
# ----------------------------------------------------------------------
def activation_surfaces():
    xs = np.linspace(-8, 8, 4001).astype(np.float32)
    xt = torch.from_numpy(xs)
    data = {
        "x": xs.tolist(),
        "sigma": torch.sigmoid(xt).numpy().tolist(),
        "sigma_prime": sigmoid_prime(xt).numpy().tolist(),
        "G": StableG()(xt).detach().numpy().tolist(),
        "G_prime": G_derivative(xt).numpy().tolist(),
    }
    path = f"{RESULTS_DIR}/activation_surfaces.json"
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"saved -> {path}")
    return data
