"""
# Profiling tools for optimizing Z3 performance.

See examples/profile.py to generate tmp/z3_trace.log.
"""

import argparse
import json
import logging
import os
import re
from collections import Counter, defaultdict
from typing import TypedDict

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, "tmp")
DEFAULT_TRACE = os.path.join(TMP, "z3_trace.log")


class TraceStats(TypedDict):
    command_counts: Counter[str]
    pattern_triggers: Counter[str]
    quantifier_stats: defaultdict[str, Counter[str]]
    axiom_instantiations: Counter[str]
    theory_solving_instances: Counter[str]


def process_trace(filename: str = DEFAULT_TRACE) -> TraceStats:
    """
    Process the Z3 trace file and extract statistics.

    Returns a dictionary with various statistics about the trace:
    - command_counts: Counter of command types
    - pattern_triggers: Counter of patterns used as triggers
    - quantifier_stats: Statistics about quantifier instantiations
    - axiom_instantiations: Count of axiom instantiations by name
    - theory_solving_instances: Count of theory solving instances by pattern
    """
    # Initialize statistics counters
    stats: TraceStats = {
        "command_counts": Counter(),
        "pattern_triggers": Counter(),
        "quantifier_stats": defaultdict(Counter),
        "axiom_instantiations": Counter(),
        "theory_solving_instances": Counter(),
    }

    # Track ID-to-name mappings for axioms and patterns
    id_to_name = {}
    id_to_pattern = {}
    id_to_axiom = {}

    # Variable to track the current quantifier being processed
    current_quant = None

    # Process the file line by line
    with open(filename) as f:
        for i, line in enumerate(f):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Extract command type (text inside square brackets)
            match = re.match(r"\[([\w-]+)\](.*)", line)
            if not match:
                continue

            cmd, args = match.groups()
            args = args.strip()

            # Count command types
            stats["command_counts"][cmd] += 1

            # Process different command types
            if cmd == "mk-quant":
                # Format: [mk-quant] #ID name num_vars pattern formula
                parts = args.split()
                if len(parts) >= 3:
                    quant_id = parts[0]
                    quant_name = parts[1]
                    current_quant = quant_id
                    id_to_name[quant_id] = quant_name

                    # Record quantifier patterns if present
                    if len(parts) > 3:
                        pattern_id = parts[3]
                        id_to_pattern[quant_id] = pattern_id
                        stats["pattern_triggers"][pattern_id] += 1

            elif cmd == "attach-var-names":
                # Format: [attach-var-names] #ID (var1) (var2) ...
                if current_quant:
                    parts = args.split()
                    quant_id = parts[0]
                    stats["quantifier_stats"][id_to_name.get(quant_id, quant_id)][
                        "attach"
                    ] += 1

            elif cmd == "inst-discovered":
                # Format: [inst-discovered] theory-solving ID pattern ; formula
                parts = args.split()
                if len(parts) >= 3:
                    pattern_type = parts[2]
                    stats["theory_solving_instances"][pattern_type] += 1

            elif cmd == "instance":
                # Format: [instance] ID formula
                parts = args.split()
                if len(parts) >= 1:
                    instance_id = parts[0]
                    stats["quantifier_stats"][id_to_name.get(instance_id, instance_id)][
                        "instantiations"
                    ] += 1

            elif cmd == "mk-app" and "=>" in args:
                # Axiom implications: [mk-app] #ID => axiom_name quantifier
                parts = args.split()
                if len(parts) >= 3 and parts[1] == "=>":
                    app_id = parts[0]
                    axiom_name = parts[2]
                    if len(parts) > 3:
                        quant_id = parts[3]
                        id_to_axiom[app_id] = axiom_name

            elif cmd == "mk-proof" and "mp" in args:
                # Modus ponens proofs might indicate axiom instantiations
                parts = args.split()
                if len(parts) >= 4 and parts[1] == "mp":
                    axiom_id = parts[2]
                    if axiom_id in id_to_axiom:
                        stats["axiom_instantiations"][id_to_axiom[axiom_id]] += 1

            # Add a basic progress indicator for large files
            if i % 1000000 == 0 and i > 0:
                logger.info(f"Processed {i:,} lines...")

    return stats


def truncate_stats(stats: TraceStats, top_n: int = 10) -> None:
    """
    Truncate the statistics to show only the top N items in each category.
    """

    def truncate(stats: Counter[str]) -> None:
        most_common = stats.most_common(top_n)
        stats.clear()
        stats.update(dict(most_common))

    truncate(stats["command_counts"])
    truncate(stats["pattern_triggers"])
    truncate(stats["axiom_instantiations"])
    truncate(stats["theory_solving_instances"])

    # TODO restrict to the top N quantifiers
    stats["quantifier_stats"].clear()


def main(args: argparse.Namespace) -> None:
    from .logging import setup_color_logging

    setup_color_logging()
    logger.info(f"Processing trace file: {args.file}")
    stats = process_trace(args.file)
    truncate_stats(stats, args.top_n)
    logger.info(f"Top {args.top_n} items:")
    logger.info(json.dumps(stats, indent=4, sort_keys=True))


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("-f", "--file", default=DEFAULT_TRACE, help="Path to Z3 trace file")
parser.add_argument(
    "-n",
    "--top-n",
    type=int,
    default=10,
    help="Number of top items to show in each category",
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
