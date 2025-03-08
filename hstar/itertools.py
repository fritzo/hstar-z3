from collections.abc import Iterator


def partitions(total: int, num_parts: int) -> Iterator[tuple[int, ...]]:
    """Generate all partitions of `total` into `num_parts`."""
    if num_parts == 1:
        yield (total,)
        return
    for count in range(total + 1):
        for part in partitions(total - count, num_parts - 1):
            yield (count,) + part


def weighted_partitions(
    total: int, part_weights: tuple[int, ...], lb: int = 1
) -> Iterator[tuple[int, ...]]:
    """
    Generate all partitions of `total` with the given part weights, where each
    part is at least `lb`. The constraints are:
    ```
    total == sum(weight * count for weight, count in zip(part_weights, part))
    all(count >= lb for count in part)
    ```
    """
    if not part_weights:
        if total == 0:
            yield ()
        return
    weight = part_weights[0]
    assert weight > 0
    if len(part_weights) == 1:
        if total % weight == 0 and total // weight >= lb:
            yield (total // weight,)
        return
    for count in range(lb, total // weight + 1):
        remaining = total - weight * count
        if remaining >= lb:
            for part in weighted_partitions(remaining, part_weights[1:], lb):
                yield (count,) + part
