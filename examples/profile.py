#!/usr/bin/env python3
"""
Z3 Theory Profiler - Runs Z3 with theory from solvers.py and collects statistics
"""

import argparse
import logging
import os
import subprocess
import time

import z3

from hstar.logging import setup_color_logging
from hstar.profile import summarize
from hstar.theory import add_theory

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, "tmp")


def main(args: argparse.Namespace) -> None:
    # Create directory for trace file if needed
    if not os.path.exists(os.path.dirname(args.trace_file)):
        os.makedirs(os.path.dirname(args.trace_file), exist_ok=True)

    # Enable tracing
    if args.trace:
        z3.set_param("proof", True)
        z3.set_param("mbqi.trace", True)
        z3.set_option(trace=True)
        z3.set_option(trace_file_name=args.trace_file)

    # Set up solver
    solver = z3.Solver()
    solver.set("timeout", args.timeout_ms)
    solver.set("unsat_core", True)

    # Set default solver parameters for better profiling
    solver.set("relevancy", 2)  # More precise relevancy propagation
    solver.set("mbqi", True)  # Enable model-based quantifier instantiation
    solver.set("qi.max_multi_patterns", 1000)  # Allow more multi-patterns

    # Enhanced profiling options with more aggressive settings
    solver.set("qi.profile", True)  # Profile quantifier instantiations
    solver.set("qi.profile_freq", args.qi_profile_freq)  # Frequency for profile output

    # Enhanced profiling options
    solver.set("mbqi.trace", True)  # Trace model-based quantifier instantiation
    solver.set("proof", True)  # Enable proof generation

    # Options to see more detailed quantifier instantiation info
    solver.set("qi.eager_threshold", 5.0)  # Lower threshold for eager instantiation
    solver.set("qi.lazy_threshold", 10.0)  # Lower threshold for lazy instantiation

    # Output options
    if args.show_instantiations:
        solver.set("instantiations2console", True)  # Show instantiation details

    if args.show_lemmas:
        solver.set("lemmas2console", True)  # Show generated lemmas

    # Add theory and measure performance
    logger.info(f"Running Z3 theory check (timeout: {args.timeout_ms}ms)")

    start = time.time()
    add_theory(solver)
    theory_time = time.time() - start
    logger.info(f"Theory added in {theory_time:.2f}s")

    # Log assertions if requested
    if args.dump_formulas:
        formulas = "\n".join(str(formula) for formula in solver.assertions())
        logger.info(f"Theory Formulas:\n{formulas}")

    if args.subprocess:
        smt2_path = os.path.join(TMP, "theory.smt2")
        with open(smt2_path, "w") as f:
            f.write(solver.to_smt2())

        # Build Z3 command with options
        cmd: list[str] = ["z3", "-st", f"-t:{args.timeout_ms}"]
        if args.trace:
            cmd.append("proof=true")
            cmd.append("trace=true")
            cmd.append(f"trace-file-name={args.trace_file}")
        cmd.append(smt2_path)
        logger.info(f"Running Z3 with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Process the output
        if result.returncode != 0:
            logger.warning(
                f"Z3 process returned non-zero exit code: {result.returncode}"
            )
        logger.info(f"Z3 stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Z3 stderr:\n{result.stderr}")
    else:
        result = solver.check()
        logger.info(f"Result: {result}")

    # Print statistics
    stats = solver.statistics()
    logger.info(f"Z3 Statistics:\n{stats}")
    summarize(args.trace_file)


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--subprocess",
    action="store_true",
    default=True,
    help="Run Z3 in a subprocess",
)
parser.add_argument(
    "--no-subprocess",
    action="store_false",
    dest="subprocess",
    help="Run Z3 in the current process",
)
parser.add_argument(
    "--timeout-ms", type=int, default=5000, help="Solver timeout in milliseconds"
)
parser.add_argument(
    "--trace",
    action="store_true",
    default=True,
    help="Enable Z3 tracing",
)
parser.add_argument(
    "--no-trace",
    action="store_false",
    dest="trace",
    help="Disable Z3 tracing",
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
parser.add_argument(
    "--qi-profile-freq",
    type=int,
    default=10000,
    help="Frequency for reporting quantifier instantiation profile "
    "(# of instantiations)",
)
parser.add_argument(
    "--show-instantiations",
    action="store_true",
    help="Show quantifier instantiations in console output",
)
parser.add_argument(
    "--show-lemmas",
    action="store_true",
    help="Show lemmas in console output",
)

if __name__ == "__main__":
    setup_color_logging(level=logging.INFO)
    try:
        main(parser.parse_args())
    except z3.Z3Exception as e:
        logger.error(e.value.decode() if isinstance(e.value, bytes) else str(e))
        raise
    except Exception as e:
        logger.exception(e)
        raise
