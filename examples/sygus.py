#!/usr/bin/env python3
"""
Syntax-Guided Synthesis (SyGuS).
"""

import argparse
import logging

import z3

from hstar.grammars import comb_grammar
from hstar.language import APP, COMP, KI, LEQ, I, K
from hstar.logging import setup_color_logging
from hstar.synthesis import sygus

logger = logging.getLogger(__name__)


def k_constraint(x: z3.ExprRef) -> z3.ExprRef:
    return x == K


def i_constraint(x: z3.ExprRef) -> z3.ExprRef:
    return x == I


def sr_constraint(x: z3.ExprRef) -> z3.ExprRef:
    s = APP(x, K)
    r = APP(x, KI)
    return LEQ(COMP(r, s), I)


def main(args: argparse.Namespace) -> None:
    constraint = globals()[f"{args.example}_constraint"]
    result = sygus(comb_grammar, constraint)
    logger.info(f"Synthesized: {result}")


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--example",
    choices=["k", "i", "sr"],
    default="k",
    help="The example to run.",
)
if __name__ == "__main__":
    setup_color_logging(level=logging.DEBUG)
    main(parser.parse_args())
