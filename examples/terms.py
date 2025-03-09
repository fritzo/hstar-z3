#!/usr/bin/env python3
"""
Term enumeration example.

This script prints λ-join-calculus terms in order of increasing complexity.
"""

import argparse

from hstar.enumeration import enumerator
from hstar.grammar import complexity


def main(args: argparse.Namespace) -> None:
    """Print the first args.number-many enumerated terms."""
    print("Complexity Term")
    print("-" * 40)

    count = 0
    for term in enumerator:
        # Skip terms with free variables if --closed is specified
        if args.closed and term.free_vars:
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

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
