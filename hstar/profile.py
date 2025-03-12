"""
# Profiling tools for optimizing Z3 performance.

See examples/profile.py to generate tmp/z3_trace.log.
"""

import argparse
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
    commands: Counter[str]
    new_match: Counter[str]
    instance_count: int
    instantiation_chains: dict[str, list[str]]
    pattern_triggers: Counter[str]
    match_to_instance: Counter[str]
    proof_steps: Counter[str]


def process_trace(filename: str = DEFAULT_TRACE) -> TraceStats:
    """Process the Z3 trace file and extract statistics."""
    stats: TraceStats = {
        "commands": Counter(),
        "new_match": Counter(),
        "instance_count": 0,
        "instantiation_chains": defaultdict(list),
        "pattern_triggers": Counter(),
        "match_to_instance": Counter(),
        "proof_steps": Counter(),
    }
    id_to_name: dict[str, str] = {}
    id_to_pattern: dict[str, set[str]] = defaultdict(set)
    last_discovered_id: str | None = None
    match_to_instance_map: dict[str, int] = {}
    quant_instantiation_count: Counter[str] = Counter()
    quant_proof_steps: Counter[str] = Counter()

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
                # Note one named ForAll may have multiple ids.
                parts = args.strip().split()
                if len(parts) >= 3:
                    quant_id = parts[0]
                    quant_name = parts[1]
                    id_to_name[quant_id] = quant_name
                    # Track patterns/triggers for this quantifier
                    if len(parts) > 3 and parts[3] != "null":
                        pattern_id = parts[3]
                        id_to_pattern[quant_id].add(pattern_id)

            elif cmd == "new-match":
                # Format: [new-match] hash_id quant_id pattern_vars ; result_terms
                parts = args.strip().split()
                if len(parts) >= 2:
                    # The second element is typically the quantifier ID
                    match_quant_id = parts[1]
                    name = id_to_name.get(match_quant_id, match_quant_id)
                    stats["new_match"][name] += 1

                    # Record match hash for potential chain tracking
                    if parts[0] not in match_to_instance_map:
                        match_to_instance_map[parts[0]] = 0

                    # If this match came from a previous instance, track the chain
                    if last_discovered_id:
                        stats["instantiation_chains"][last_discovered_id].append(
                            parts[0]
                        )

            elif cmd == "inst-discovered":
                # Format: [inst-discovered] source hash_id other_info
                parts = args.strip().split()
                if len(parts) >= 2:
                    last_discovered_id = parts[1]
                    source = parts[0]
                    stats["pattern_triggers"][source] += 1

            elif cmd == "instance":
                # Format: [instance] hash_id proof_id ; depth
                parts = args.strip().split()
                if len(parts) >= 1:
                    stats["instance_count"] += 1
                    instance_hash = parts[0]

                    # Match this instance to its discovery
                    if last_discovered_id and last_discovered_id == instance_hash:
                        if last_discovered_id in match_to_instance_map:
                            match_to_instance_map[last_discovered_id] += 1

                    # Track which quantifiers are being instantiated most
                    if len(parts) >= 2 and parts[1] in id_to_name:
                        quant_name = id_to_name[parts[1]]
                        quant_instantiation_count[quant_name] += 1

            elif cmd == "mk-proof":
                # Track proof steps by type
                parts = args.strip().split()
                if len(parts) >= 2:
                    proof_type = parts[1]
                    stats["proof_steps"][proof_type] += 1

                    # If this proof is related to a quantifier, track that
                    if len(parts) >= 3 and parts[2] in id_to_name:
                        quant_name = id_to_name[parts[2]]
                        quant_proof_steps[quant_name] += 1

            # Add a basic progress indicator for large files
            if i % 1000000 == 0 and i > 0:
                logger.info(f"Processed {i:,} lines...")

    # Calculate efficiency of match-to-instance conversion
    for match_id, match_count in match_to_instance_map.items():
        if match_count > 0:
            stats["match_to_instance"][match_id] = match_count

    return stats


def truncate_stats(stats: TraceStats, top_n: int = 10) -> None:
    """
    Truncate the statistics to show only the top N items in each category.
    """

    def truncate(stats: Counter[str]) -> None:
        total = sum(stats.values())
        most_common = dict(stats.most_common(top_n))
        most_common["(total)"] = total
        stats.clear()
        stats.update(dict(most_common))

    truncate(stats["commands"])
    truncate(stats["new_match"])
    truncate(stats["pattern_triggers"])
    truncate(stats["proof_steps"])

    # Truncate the chains to the top N
    if isinstance(stats["instantiation_chains"], defaultdict):
        # Convert to regular dict first
        chains_dict = dict(stats["instantiation_chains"])
        # Sort by length of chain
        sorted_chains = sorted(
            chains_dict.items(), key=lambda x: len(x[1]), reverse=True
        )[:top_n]
        stats["instantiation_chains"] = dict(sorted_chains)


def main(args: argparse.Namespace) -> None:
    from .logging import setup_color_logging

    setup_color_logging()
    logger.info(f"Processing trace file: {args.file}")
    stats = process_trace(args.file)
    truncate_stats(stats, args.top_n)

    # Print a structured table
    lines: list[str] = []
    lines.append("Z3 Trace Analysis:")
    lines.append("-" * 40)
    lines.append("       Count  Command types")
    for cmd, count in stats["commands"].most_common(args.top_n):
        lines.append(f"{count:12,}  {cmd}")
    lines.append("-" * 40)
    lines.append("       Count  Quantifier pattern matches")
    for name, count in stats["new_match"].most_common(args.top_n):
        lines.append(f"{count:12,}  {name}")
    lines.append("-" * 40)
    lines.append("       Count  Proof step types")
    for proof_type, count in stats["proof_steps"].most_common(args.top_n):
        lines.append(f"{count:12,}  {proof_type}")
    lines.append("-" * 40)
    lines.append(f"Total instance count: {stats['instance_count']:,}")
    logger.info("\n".join(lines))


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
