"""
sgaa.nets
=========
Feed-forward network and the activation schemes compared in the SGAA study,
together with tools for recording per-layer pre-activations, activations and
gradient norms.
"""

import numpy as np
import torch
import torch.nn as nn

from .activations import StableG, G_derivative, sigmoid_prime


# ----------------------------------------------------------------------
# Activation schemes
# ----------------------------------------------------------------------
# Each scheme returns a list of activation modules, one per hidden layer.
SCHEMES = ["sigmoid_all", "sg_alt", "gs_alt", "relu", "tanh"]

SHORT = {
    "sigmoid_all": "all-σ",
    "sg_alt": "σ/G alt",
    "gs_alt": "G/σ alt",
    "relu": "ReLU",
    "tanh": "tanh",
}


def activation_list(pattern, n_layers):
    """Return ``n_layers`` activation modules for the given scheme."""
    if pattern == "sigmoid_all":
        return [nn.Sigmoid() for _ in range(n_layers)]
    if pattern == "relu":
        return [nn.ReLU() for _ in range(n_layers)]
    if pattern == "tanh":
        return [nn.Tanh() for _ in range(n_layers)]
    if pattern == "sg_alt":       # start with sigma, alternate sigma / G
        return [nn.Sigmoid() if i % 2 == 0 else StableG()
                for i in range(n_layers)]
    if pattern == "gs_alt":       # start with G, alternate G / sigma
        return [StableG() if i % 2 == 0 else nn.Sigmoid()
                for i in range(n_layers)]
    raise ValueError(f"unknown scheme {pattern!r}")


# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------
class FeedforwardNet(nn.Module):
    """MLP with one activation per hidden layer, all layers Xavier-initialised.

    Parameters
    ----------
    hidden_sizes : list[int]
        Width of each hidden layer.
    pattern : str
        Activation scheme (see :func:`activation_list`).
    in_dim, out_dim : int
        Input dimension (784 for MNIST) and output dimension (10).
    init_scale : float
        Multiplier applied to the Xavier std for every linear layer.  This is
        the single knob used to keep the comparison activation-only.
    """

    def __init__(self, hidden_sizes, pattern, in_dim=784, out_dim=10,
                 init_scale=1.0):
        super().__init__()
        self.pattern = pattern
        dims = [in_dim] + list(hidden_sizes) + [out_dim]
        self.linears = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.acts = nn.ModuleList(activation_list(pattern, len(hidden_sizes)))
        self._init_weights(init_scale)

    def _init_weights(self, scale):
        for lin in self.linears:
            nn.init.xavier_uniform_(lin.weight, gain=scale)
            nn.init.zeros_(lin.bias)

    def forward(self, x, record=False):
        x = x.reshape(x.size(0), -1)
        pre, act = [], []
        for i, (lin, a) in enumerate(zip(self.linears, self.acts)):
            z = lin(x)
            if record:
                pre.append(z.detach())
            x = a(z)
            if record:
                act.append(x.detach())
        z = self.linears[-1](x)               # final output layer (logits)
        if record:
            pre.append(z.detach())
        return z, (pre, act)

    def forward_pre_acts(self, x):
        """Only collect pre-activations z_l and outputs a_l for every layer."""
        return self.forward(x, record=True)


# ----------------------------------------------------------------------
# Simple, explicit pre-activation + gradient capture (no hooks needed)
# ----------------------------------------------------------------------
def per_layer_weight_grad_norms(net):
    """Return ||dL/dW_l||_F for each linear layer (call after backward)."""
    norms = []
    for lin in net.linears:
        if lin.weight.grad is not None:
            norms.append(float(lin.weight.grad.norm(2).item()))
        else:
            norms.append(float("nan"))
    return norms


def forward_and_backward(net, x, y, criterion):
    """Forward/backward capturing z_l, a_l and dL/dz_l for every layer.

    Returns a dict with tensors.
    """
    x = x.reshape(x.size(0), -1)
    xin = [x]
    pre = []      # z_l for each linear layer (incl. output layer)
    act = []      # a_l after each hidden layer (before next linear)
    for i, (lin, a) in enumerate(zip(net.linears, net.acts)):
        z = lin(xin[-1])
        pre.append(z)
        xin.append(a(z))
        act.append(xin[-1])
    z_out = net.linears[-1](xin[-1])
    pre.append(z_out)
    loss = criterion(z_out, y)

    grads = torch.autograd.grad(loss, pre, retain_graph=True,
                                allow_unused=True, create_graph=False)
    z_norms = [float(g.norm(2).item()) if g is not None else float("nan")
               for g in grads]
    for p in net.parameters():
        if p.grad is not None:
            p.grad = None
    loss.backward()
    w_norms = per_layer_weight_grad_norms(net)
    return {
        "loss": float(loss.item()),
        "pre": [z.detach() for z in pre],
        "act": [a.detach() for a in act],
        "z_grad_norms": z_norms,
        "w_grad_norms": w_norms,
    }


# ----------------------------------------------------------------------
# Derivative-product statistics for the alternating hypothesis
# ----------------------------------------------------------------------
def derivative_product_stats(pre, pattern):
    """For adjacent hidden layers, compute the backprop-relevant statistics.

    The chain rule multiplies the derivatives of two adjacent activations:
        sigma'(z_i) * G'(z_{i+1})  when the pair is (sigma, G) or (G, sigma).
    Because G'(z) = 1/sigma'(z), this product equals sigma'(z_i)/sigma'(z_{i+1}),
    which is ~1 precisely when the two pre-activations sit at similar sigma'
    levels (a meaningful notion of "adjacent pre-activations being similar").

    Returns
    -------
    products : list of np.ndarray  (the raw per-sample product per pair)
    stats    : list of dict        (mean/std/percentiles, plus sigma'-similarity)
    kinds    : list of str         ("sigma(i)->g(i+1)", ...)
    """
    n = len(pre) - 1  # number of hidden layers (last pre is the output layer)
    deriv = {"sigma": sigmoid_prime, "g": G_derivative}

    def layer_kind(i):
        if pattern == "sg_alt":
            return "sigma" if i % 2 == 0 else "g"
        if pattern == "gs_alt":
            return "g" if i % 2 == 0 else "sigma"
        if pattern == "sigmoid_all":
            return "sigma"
        return None   # relu / tanh not expressible via the sigma<->G identity

    products, stats, kinds = [], [], []
    for i in range(n - 1):
        k1, k2 = layer_kind(i), layer_kind(i + 1)
        if k1 is None or k2 is None:
            continue
        z1, z2 = pre[i], pre[i + 1]
        d1, d2 = deriv[k1](z1), deriv[k2](z2)
        prod = (d1 * d2).detach()
        s1 = torch.sigmoid(z1).detach()
        s2 = torch.sigmoid(z2).detach()
        sp1 = s1 * torch.sigmoid(-z1)
        sp2 = s2 * torch.sigmoid(-z2)
        # "sigma'-similarity": relative gap between adjacent sigma' levels
        sim = ((sp1 - sp2).abs() / (0.5 * (sp1 + sp2) + 1e-12)).mean().item()
        p = prod.cpu().numpy()
        products.append(p)
        stats.append({
            "kind": f"{k1}({i})->{k2}({i+1})",
            "prod_mean": float(p.mean()),
            "prod_std": float(p.std()),
            "prod_p05": float(np.percentile(p, 5)),
            "prod_p95": float(np.percentile(p, 95)),
            "mean_abs_dev_from_1": float(np.abs(p - 1.0).mean()),
            "sigma_prime_similarity": sim,
        })
        kinds.append(stats[-1]["kind"])
    return products, stats, kinds
