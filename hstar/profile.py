"""
# Profiling tools for optimizing Z3 performance.

See examples/profile.py to generate tmp/z3_trace.log.
"""

import argparse
import json
import logging
import os
import re
from collections import Counter
from typing import TypedDict

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, "tmp")
DEFAULT_TRACE = os.path.join(TMP, "z3_trace.log")


class TraceStats(TypedDict):
    commands: Counter[str]
    quantifiers: Counter[str]


def process_trace(filename: str = DEFAULT_TRACE) -> TraceStats:
    """Process the Z3 trace file and extract statistics."""
    stats: TraceStats = {
        "commands": Counter(),
        "quantifiers": Counter(),
    }
    id_to_name: dict[str, str] = {}

    # Process the file line by line
    with open(filename) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Extract command type (text inside square brackets)
            match = re.match(r"\[([\w-]+)\](.*)", line)
            if not match:
                continue
            cmd, args = match.groups()

            # Count command types
            stats["commands"][cmd] += 1

            # Process different command types
            if cmd == "mk-quant":
                # Format: [mk-quant] #ID name num_vars pattern formula
                parts = args.strip().split()
                if len(parts) >= 3:
                    quant_id = parts[0]
                    quant_name = parts[1]
                    id_to_name[quant_id] = quant_name

            elif cmd == "instance":
                # Format: [instance] ID formula
                parts = args.strip().split()
                if len(parts) >= 1:
                    instance_id = parts[0]
                    name = id_to_name.get(instance_id, instance_id)
                    stats["quantifiers"][name] += 1

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

    truncate(stats["commands"])
    truncate(stats["quantifiers"])


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
