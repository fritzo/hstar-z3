#!/usr/bin/env python3
"""
Term enumeration example.

This script prints λ-join-calculus terms in order of increasing complexity.
"""

import argparse

from hstar.enumeration import enumerator
from hstar.normal import complexity, is_closed, is_linear


def main(args: argparse.Namespace) -> None:
    """Print the first args.number-many enumerated terms."""
    print("Complexity Term")
    print("-" * 40)

    count = 0
    for term in enumerator:
        # Filter to subsets of terms
        if args.closed and not is_closed(term):
            continue
        if args.linear and not is_linear(term):
            continue
        if args.nonlinear and is_linear(term):
            continue

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
    "--linear",
    action="store_true",
    help="Only show linear terms",
)
group.add_argument(
    "--nonlinear",
    action="store_true",
    help="Only show non-linear terms",
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
