#!/usr/bin/env python3
"""
Z3 Theory Profiler - Runs Z3 with theory from solvers.py and collects statistics
"""

import argparse
import os
import time

import z3

from hstar.solvers import add_theory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, "tmp")


def main(args: argparse.Namespace) -> None:
    # Create directory for trace file if needed
    if args.verbose and not os.path.exists(os.path.dirname(args.trace_file)):
        os.makedirs(os.path.dirname(args.trace_file), exist_ok=True)

    # Set up solver
    solver = z3.Solver()
    solver.set("timeout", args.timeout_ms)
    solver.set("unsat_core", True)

    if args.verbose:
        solver.set("trace", True)
        solver.set("trace_file_name", args.trace_file)

    # Add theory and measure performance
    print(f"Running Z3 theory check (timeout: {args.timeout_ms}ms)")

    start = time.time()
    add_theory(solver)
    theory_time = time.time() - start
    print(f"Theory added in {theory_time:.2f}s")

    start = time.time()
    result = solver.check()
    solve_time = time.time() - start
    print(f"Result: {result} in {solve_time:.2f}s")

    # Print statistics
    print("\n==== Z3 Statistics ====")
    print(solver.statistics())

    # Print formulas if requested
    if args.dump_formulas:
        print("\n==== Theory Formulas ====")
        for i, formula in enumerate(solver.assertions()):
            print(f"Formula {i+1}:\n{formula}\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--timeout-ms", type=int, default=1000, help="Solver timeout in milliseconds"
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose Z3 tracing",
)
parser.add_argument(
    "--trace-file",
    default=os.path.join(TMP, "z3_trace.log"),
    help="File for Z3 trace output",
)
parser.add_argument(
    "--dump-formulas",
    action="store_true",
    help="Print all formulas added to solver",
)

if __name__ == "__main__":
    main(parser.parse_args())
