"""
sgaa.__main__
=============
Command-line entry point for the SGAA study.

Examples
--------
    python -m sgaa verify            # verify the math
    python -m sgaa experiments       # run main comparison + depth sweep
    python -m sgaa figures           # regenerate all figures
    python -m sgaa analyze           # print the summary tables
    python -m sgaa run-all           # the full pipeline
"""

import argparse
import sys

import torch


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def cmd_verify(args):
    from .activations import verify_math
    verify_math()
    return 0


def cmd_experiments(args):
    from . import experiments as E
    E.activation_surfaces()
    if args.main:
        hidden = tuple([args.width] * 4)
        E.run_main_comparison(epochs=args.epochs, hidden_sizes=hidden,
                              device=args.device)
    if args.depth:
        E.run_depth_sweep(epochs=args.depth_epochs, width=args.depth_width,
                          device=args.device)
    return 0


def cmd_figures(args):
    from . import plotting as P
    P.make_all()
    return 0


def cmd_analyze(args):
    from . import analyze as A
    A.main()
    return 0


def cmd_run_all(args):
    cmds = ["verify", "experiments", "figures", "analyze"]
    for c in cmds:
        print(f"\n{'=' * 78}\n>>> {c}\n{'=' * 78}")
        if c == "verify":
            cmd_verify(args)
        elif c == "experiments":
            cmd_experiments(args)
        elif c == "figures":
            cmd_figures(args)
        else:
            cmd_analyze(args)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="python -m sgaa",
                                description="Sigmoid-G Alternating Activation")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify", help="verify the math / identities")

    pe = sub.add_parser("experiments", help="run the experiments")
    pe.add_argument("--epochs", type=int, default=10)
    pe.add_argument("--width", type=int, default=256,
                    help="hidden width for the main comparison")
    pe.add_argument("--depth-width", type=int, default=128,
                    help="hidden width for the depth sweep")
    pe.add_argument("--depth-epochs", type=int, default=6)
    pe.add_argument("--no-main", dest="main", action="store_false",
                    help="skip the main comparison")
    pe.add_argument("--no-depth", dest="depth", action="store_false",
                    help="skip the depth sweep")
    pe.add_argument("--device", default=None)
    pe.set_defaults(main=True, depth=True)

    sub.add_parser("figures", help="regenerate all figures")

    sub.add_parser("analyze", help="print the summary tables")

    sub.add_parser("run-all", help="verify + experiments + figures + analyze")

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "device", None) is None:
        args.device = _device()
    for attr in ("width", "depth_width", "depth_epochs"):
        if not hasattr(args, attr):
            setattr(args, attr, 256 if attr == "width" else 128)
    return {"verify": cmd_verify,
            "experiments": cmd_experiments,
            "figures": cmd_figures,
            "analyze": cmd_analyze,
            "run-all": cmd_run_all}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
