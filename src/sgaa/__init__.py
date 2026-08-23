"""
sgaa
====
Sigmoid-G Alternating Activation (SGAA): a parameter-free activation scheme
that pairs the logistic sigmoid with the antiderivative of its reciprocal
derivative so that adjacent-layer derivative products stay near 1.
"""

from .activations import (
    sigmoid,
    sigmoid_prime,
    G_closed,
    G_numerically_stable,
    G_derivative,
    derivative_product,
    StableG,
    verify_math,
)
from .nets import (
    FeedforwardNet,
    activation_list,
    forward_and_backward,
    derivative_product_stats,
    SCHEMES,
    SHORT,
)

__all__ = [
    "sigmoid",
    "sigmoid_prime",
    "G_closed",
    "G_numerically_stable",
    "G_derivative",
    "derivative_product",
    "StableG",
    "verify_math",
    "FeedforwardNet",
    "activation_list",
    "forward_and_backward",
    "derivative_product_stats",
    "SCHEMES",
    "SHORT",
]

__version__ = "0.1.0"
