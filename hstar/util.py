import sys
from abc import ABCMeta
from collections import Counter
from collections.abc import Hashable, Iterator
from typing import TypeVar
from weakref import WeakKeyDictionary, ref

counter: Counter[str] = Counter()

_V = TypeVar("_V", bound=Hashable)
_INTERN: WeakKeyDictionary[Hashable, ref[Hashable]] = WeakKeyDictionary()


def intern(x: _V) -> _V:
    """Return a canonical object for `x`, useful for hash consing."""
    counter["intern.hit"] += 1
    if x is None or x is False or x is True:
        return x  # type: ignore
    if isinstance(x, str):
        return sys.intern(x)  # type: ignore
    try:
        return _INTERN[x]()  # type: ignore
    except KeyError:
        counter["intern.hit"] -= 1
        counter["intern.miss"] += 1
        _INTERN[x] = ref(x)
        return x


class HashConsMeta(ABCMeta):
    """Metaclass to hash cons instances."""

    def __call__(self, *args, **kwargs):  # type: ignore
        # TODO intern args and kwargs values?
        return intern(super().__call__(*args, **kwargs))

    def __new__(mcs, name, bases, namespace):  # type: ignore
        # Support copy.deepcopy(-).
        def __deepcopy__(self, memo):  # type: ignore
            return self

        # Support pickle.loads(pickle.dumps(-)) for dataclasses.
        def __reduce__(self):  # type: ignore
            args = tuple(getattr(self, f) for f in self.__dataclass_fields__)
            return type(self), args

        namespace["__deepcopy__"] = __deepcopy__
        namespace["__reduce__"] = __reduce__
        return super().__new__(mcs, name, bases, namespace)

    def __getitem__(self, params):  # type: ignore
        """Binding a generic type has no runtime effect."""
        return self


def boool_and(lhs: bool | None, rhs: bool | None) -> bool | None:
    if lhs is True and rhs is True:
        return True
    if lhs is False or rhs is False:
        return False
    return None


def boool_or(lhs: bool | None, rhs: bool | None) -> bool | None:
    if lhs is True or rhs is True:
        return True
    if lhs is False and rhs is False:
        return False
    return None


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
