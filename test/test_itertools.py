from hstar.itertools import iter_subsets, partitions, weighted_partitions


def test_iter_subsets() -> None:
    actual = set(map(frozenset, iter_subsets(range(3))))
    expected = set(
        map(
            frozenset,
            [
                [],
                [1],
                [2],
                [1, 2],
                [0],
                [0, 1],
                [0, 2],
                [0, 1, 2],
            ],
        )
    )
    assert actual == expected


def test_partitions() -> None:
    total = 3
    actual = list(partitions(total, num_parts=2))
    expected = [(0, 3), (1, 2), (2, 1), (3, 0)]
    assert actual == expected
    for counts in actual:
        assert sum(counts) == total


def test_weighted_partitions() -> None:
    total = 10
    weights = (1, 2, 3)
    actual = list(weighted_partitions(total, weights))
    expected = [(1, 3, 1), (2, 1, 2), (3, 2, 1), (5, 1, 1)]
    assert actual == expected
    for counts in actual:
        actual_total = sum(w * c for w, c in zip(weights, counts, strict=True))
        assert actual_total == total
