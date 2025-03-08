from hstar.util import weighted_partitions


def test_weighted_partitions() -> None:
    expected = [(1, 3, 1), (2, 1, 2), (3, 2, 1), (5, 1, 1)]
    actual = list(weighted_partitions(10, (1, 2, 3)))
    assert actual == expected
