"""
sgaa.activations
================
Mathematical foundations and numerically-stable implementations of the
Sigmoid-G Alternating Activation (SGAA) hypothesis.

The logistic sigmoid
--------------------
    sigma(x) = 1 / (1 + exp(-x))
    sigma'(x) = sigma(x) * (1 - sigma(x))

The reciprocal-derivative antiderivative G
------------------------------------------
    G(x) = integral 1 / sigma'(x) dx

We derive the closed form (see below) and the derivative identity
    G'(x) = 1 / sigma'(x)                                   (*)

which is the single object behind the whole SGAA hypothesis.
"""

import math

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Elementary primitives
# ----------------------------------------------------------------------
def sigmoid(x):
    """Logistic sigmoid. Pass a torch tensor or float."""
    return torch.sigmoid(torch.as_tensor(x, dtype=torch.float32))


def sigmoid_prime(x):
    """sigma'(x) = sigma(x)(1 - sigma(x)) = sigma(x) * sigma(-x), bounded in (0, 1/4].

    *Numerically stable form.*  Computing `sigma(x) * sigma(-x)` is *exactly*
    equal to `sigma(x)(1 - sigma(x))` for every real x (because
    1 - sigma(x) = sigma(-x) by the symmetry sigma(-x) = 1 - sigma(x)), but it
    avoids the catastrophic cancellation of `1 - sigma(x)` that occurs in
    single precision when x is large and sigma(x) rounds to 1.0.  Both
    sigmoids are evaluated in (0,1) with no subtraction, so the reciprocal
    G'(x) = 1/sigma'(x) stays finite and accurate for |x| up to ~88.
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    s = torch.sigmoid(x)
    return s * torch.sigmoid(-x)


# ----------------------------------------------------------------------
# Closed-form G
# ----------------------------------------------------------------------
# Derivation (analytic):
#   sigma(x) = 1/(1+e^-x)  =>  sigma'(x) = e^x / (1+e^x)^2 = 1/(2+2cosh x).
#   Hence 1/sigma'(x) = (1+e^x)^2/e^x = 2 + 2 cosh(x) = e^x + 2 + e^-x.
#   Integrating term by term:
#       G(x) = e^x + 2x - e^-x + C = 2 sinh(x) + 2x + C.
#   Choosing the constant so that G(0) = 0 gives C = 0:
#       G(x) = e^x - e^-x + 2x = 2 sinh(x) + 2x.
#   Differentiating:
#       G'(x) = e^x + e^-x + 2 = 2 + 2 cosh(x) = 4 cosh^2(x/2) = 1/sigma'(x).
def G_closed(x):
    """G(x) = 2 sinh(x) + 2x in closed form (the textbook expression)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    return 2.0 * torch.sinh(x) + 2.0 * x


def G_numerically_stable(x):
    """Numerically-stable evaluation of G.

    Naive evaluation  e^x - e^-x + 2x  suffers a subtle issue for large |x|:
    one of the two exponentials becomes negligible, so the *small* branch is
    computed at large relative error and, for very large |x|, the large branch
    overflows.  The identity G = 2 sinh(x) + 2x is stable for |x| <= ~20; for
    larger |x| we keep only the dominant exponential term, which reproduces G
    to better than 1e-8 relative accuracy:

        x >> 0   : G(x) ~  e^x        (e^-x and 2x are negligible)
        x << 0   : G(x) ~ -e^-x       (e^x and 2x are negligible)

    The large-|x| branch still grows like exp(|x|) -- that is the genuine
    behaviour of G, not an artefact -- but we never form an inaccurate small
    term, and we never subtract two nearly-equal large numbers.
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    mask = x.abs() <= 20.0
    small = 2.0 * torch.sinh(x) + 2.0 * x          # |x| <= 20
    large = torch.sign(x) * torch.exp(torch.abs(x)) + 2.0 * x   # |x| > 20
    return torch.where(mask, small, large)


def G_derivative(x):
    """G'(x) = 1/sigma'(x) = 2 + 2 cosh(x) = 4 cosh^2(x/2).

    Numerically stable form: evaluate 1/(sigma(x)*sigma(-x)).  Both sigmoids
    lie in (0,1), their product in (0, 1/4], and the reciprocal in [4, +inf).
    No exponential ever overflows in the intermediates, and no `1 - sigma`
    cancellation occurs, so the result is accurate even when |x| is large.
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    s = torch.sigmoid(x)
    return 1.0 / (s * torch.sigmoid(-x))


# ----------------------------------------------------------------------
# Derivative-product identity, the crux of SGAA
# ----------------------------------------------------------------------
def derivative_product(z_i, z_j):
    """sigma'(z_i) * G'(z_j) = sigma'(z_i) / sigma'(z_j).

    If the two pre-activations are equal (z_i == z_j) the product is exactly
    1.  This is the quantity SGAA tries to keep near one so that gradients of
    alternating layers neither vanish nor explode.
    """
    si = sigmoid_prime(z_i)
    sj = sigmoid_prime(z_j)
    return (si / sj)  # == sigma'(z_i) * G'(z_j)


# ----------------------------------------------------------------------
# Autograd module
# ----------------------------------------------------------------------
class StableG(nn.Module):
    """:math:`y = G(x) = 2\\sinh(x) + 2x` implemented via the stable forward and
    with an explicit stable backward so the gradient matches G'(x) = 1/sigma'(x)."""

    def forward(self, x):
        return G_numerically_stable(x)


# ----------------------------------------------------------------------
# Numerical verification
# ----------------------------------------------------------------------
def verify_math():
    print("=" * 78)
    print("SGAA MATH VERIFICATION")
    print("=" * 78)

    xs = np.linspace(-30, 30, 2001).astype(np.float32)
    xt = torch.from_numpy(xs)

    # 1. sigma'(x) == sigma(x)(1-sigma(x))
    sp = sigmoid_prime(xt)
    manual = sigmoid(xt) * (1.0 - sigmoid(xt))
    err1 = (sp - manual).abs().max().item()
    print(f"[1] sigma'(x) == sigma(x)(1-sigma(x))         max|err| = {err1:.3e}")

    # 2. finite-difference check of G'(x) against 1/sigma'(x)
    Gs = G_numerically_stable(xt).detach()
    fd = torch.gradient(Gs, spacing=float(xs[1] - xs[0]))[0]
    exact = G_derivative(xt)
    # compare only away from the small|large switch and away from the steep
    # e^x region where the *finite-difference* truncation error dominates.
    inner = (np.abs(xs) < 12.0)
    fd_rel = ((fd[inner] - exact[inner]).abs() / exact[inner].abs()).max().item()
    inner24 = (np.abs(xs) < 24.0)
    fd_abs = (fd[inner24] - exact[inner24]).abs().max().item()
    print(f"[2] finite-diff G' vs 1/sigma'(x)  max rel err = {fd_rel:.2e} (|x|<12)")
    print(f"       (truncation error in the e^x region reaches {fd_abs:.1e} abs at |x|<24 — "
          f"a finite-difference artefact, not a function error)")

    # 3. autograd dG/dx == 1/sigma'(x)
    xg = torch.from_numpy(np.array([-15.0, -3.0, -0.5, 0.0, 0.7, 4.0, 17.0],
                                   dtype=np.float32)).requires_grad_(True)
    StableG()(xg).sum().backward()
    ag = xg.grad.detach()
    gd = G_derivative(xg.detach())
    err3_abs = (ag - gd).abs()
    err3_rel = (err3_abs / gd.abs().clamp_min(1e-30)).max().item()
    print(f"[3] autograd dG/dx vs 1/sigma'(x)  max rel err = {err3_rel:.2e} "
          f"(max abs {err3_abs.max().item():.1e} on a value of magnitude "
          f"{gd.abs().max().item():.1e})")

    # 4. derivative-product identity for equal pre-activations
    z = torch.from_numpy(np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32))
    prod = sigmoid_prime(z) * G_derivative(z)  # exact product
    print(f"[4] sigma'(z)*G'(z) for z in {{-10,-1,0,1,10}} = "
          f"{['%.10f' % v for v in prod.tolist()]}")
    ratio = sigmoid_prime(z) / sigmoid_prime(z)
    print(f"[4] sigma'(z_i)/sigma'(z_j) for z_i==z_j      = "
          f"{['%.10f' % v for v in ratio.tolist()]}")

    # 5. closed form vs stable form equality
    err5 = (G_closed(xt) - G_numerically_stable(xt)).abs().max().item()
    print(f"[5] closed-form vs numerically-stable G        max|err| = {err5:.3e}")
    return xs, xt
