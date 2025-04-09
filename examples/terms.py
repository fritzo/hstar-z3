#!/usr/bin/env python3
"""
Term enumeration example.

This script prints λ-join-calculus terms in order of increasing complexity.
"""

import argparse
import random

from hstar.enumeration import enumerator
from hstar.normal import complexity, is_affine, is_closed


def main(args: argparse.Namespace) -> None:
    """Print the first args.number-many enumerated terms."""
    print("Complexity Term")
    print("-" * 40)

    count = 0
    for term in enumerator:
        # Filter to subsets of terms
        if args.closed and not is_closed(term):
            continue
        if args.affine and not is_affine(term):
            continue
        if args.nonaffine and is_affine(term):
            continue

        if args.prob >= 1 or random.random() < args.prob:
            print(f"{complexity(term)}\t{term}")
        count += 1
        if count >= args.number:
            break


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--number", "-n", type=int, default=100, help="Number of terms to enumerate"
)
parser.add_argument(
    "--closed",
    action="store_true",
    help="Only show closed terms (without free variables)",
)
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--affine",
    action="store_true",
    help="Only show affine terms",
)
group.add_argument(
    "--nonaffine",
    action="store_true",
    help="Only show non-affine terms",
)
parser.add_argument(
    "--prob",
    "-p",
    type=float,
    default=1.0,
    help="Probability of printing a term",
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
