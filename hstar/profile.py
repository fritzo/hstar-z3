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
    instance_source: Counter[str]
    instantiation_depth: Counter[int]


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
        "instance_source": Counter(),
        "instantiation_depth": Counter(),
    }
    id_to_name: dict[str, str] = {}
    id_to_pattern: dict[str, set[str]] = defaultdict(set)
    last_discovered_id: str | None = None
    match_to_instance_map: dict[str, int] = {}

    # Track depth of instantiation chains
    instance_depth: dict[str, int] = {}

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

                    # Track the source of instantiation
                    stats["instance_source"][source] += 1

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

                    # Track instantiation depth if available
                    if len(parts) >= 3 and parts[2].isdigit():
                        depth = int(parts[2])
                        instance_depth[instance_hash] = depth
                        stats["instantiation_depth"][depth] += 1

            elif cmd == "mk-proof":
                # Track proof steps by type
                parts = args.strip().split()
                if len(parts) >= 2:
                    proof_type = parts[1]
                    stats["proof_steps"][proof_type] += 1

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

    def truncate(stats_counter: Counter[str]) -> None:
        total = sum(stats_counter.values())
        most_common = dict(stats_counter.most_common(top_n))
        most_common["(total)"] = total
        stats_counter.clear()
        stats_counter.update(dict(most_common))

    truncate(stats["commands"])
    truncate(stats["new_match"])
    truncate(stats["pattern_triggers"])
    truncate(stats["proof_steps"])
    truncate(stats["instance_source"])

    # Convert instantiation depth to percentage
    total_instances = sum(stats["instantiation_depth"].values())
    if total_instances > 0:
        depth_percentage = {
            f"depth {k}": (v / total_instances) * 100
            for k, v in stats["instantiation_depth"].items()
        }
        # Sort by depth
        sorted_depths = sorted(
            depth_percentage.items(), key=lambda x: int(x[0].split()[1])
        )
        # Take top_n depths
        if len(sorted_depths) > top_n:
            sorted_depths = sorted_depths[:top_n]
        stats["instantiation_depth"].clear()
        stats["instantiation_depth"].update(
            {int(k.split()[1]): v for k, v in sorted_depths}
        )

    # Truncate the chains to the top N
    if isinstance(stats["instantiation_chains"], defaultdict):
        # Convert to regular dict first
        chains_dict = dict(stats["instantiation_chains"])
        # Sort by length of chain
        sorted_chains = sorted(
            chains_dict.items(), key=lambda x: len(x[1]), reverse=True
        )[:top_n]
        stats["instantiation_chains"] = dict(sorted_chains)


def summarize(infile: str, *, outfile: str | None = None, top_n: int = 10) -> None:
    """
    Process a Z3 trace file and generate a summary of statistics.

    Args:
        infile: Path to the Z3 trace file to analyze
        outfile: Optional path to write the analysis summary
        top_n: Number of top items to show in each category
    """
    from .logging import setup_color_logging

    if outfile is None:
        assert infile.endswith(".log"), infile
        outfile = infile[:-4] + ".txt"

    setup_color_logging()
    logger.info(f"Processing trace file: {infile}")
    stats = process_trace(infile)
    truncate_stats(stats, top_n)

    # Print a structured table
    lines: list[str] = []

    # Format functions for consistent output
    def format_section(title: str) -> None:
        lines.append("")
        lines.append(f"=== {title} ===")

    def format_counter(counter: Counter[str], title: str) -> None:
        lines.append(f"       Count  {title}")
        for name, count in counter.most_common(top_n):
            if name == "(total)":
                continue
            lines.append(f"{count:12,}  {name}")
        total = counter.get("(total)", 0)
        if total > 0:
            lines.append(f"{'':12}  -----")
            lines.append(f"{total:12,}  (total)")

    # Add title
    lines.append("Z3 Trace Analysis")

    # Command statistics
    format_section("Command Statistics")
    format_counter(stats["commands"], "Command")

    # Pattern matching statistics
    format_section("Pattern Matching Statistics")
    format_counter(stats["new_match"], "Quantifier pattern")

    # Instantiation statistics
    format_section("Instantiation Statistics")
    lines.append(f"Total instance count: {stats['instance_count']:,}")
    mean = stats["instance_count"] / sum(stats["new_match"].values())
    lines.append(f"Average instances per match: {mean:.2f}")

    percentage: float
    if stats["instantiation_depth"]:
        lines.append("")
        lines.append("Instantiation depth distribution:")
        for depth, percentage in stats["instantiation_depth"].items():
            lines.append(f"  {depth}: {percentage:.1f}%")

    lines.append("")
    lines.append("Instantiation source:")
    for source, count in stats["instance_source"].most_common(top_n):
        percentage = (count / stats["instance_count"]) * 100
        lines.append(f"  {source}: {count:,} ({percentage:.1f}%)")

    # Proof step statistics
    format_section("Proof Statistics")
    format_counter(stats["proof_steps"], "Proof step types")

    summary = "\n".join(lines)
    logger.info(summary)
    with open(outfile, "w") as f:
        f.write(summary)


def main(args: argparse.Namespace) -> None:
    """Process command line arguments and run the analysis."""
    summarize(args.infile, outfile=args.outfile, top_n=args.top_n)


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "-i", "--infile", default=DEFAULT_TRACE, help="Path to Z3 trace file"
)
parser.add_argument(
    "-o", "--outfile", default=None, help="Path to output file for analysis (optional)"
)
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
