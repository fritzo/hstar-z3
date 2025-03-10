import logging

import pytest
import z3

from hstar import grammar, solvers
from hstar.bridge import py_to_z3, z3_to_py

logger = logging.getLogger(__name__)

# A list of examples with Python terms and their corresponding Z3 expressions
EXAMPLES: list[tuple[grammar.Term, z3.ExprRef]] = [
    # Basic Terms
    (grammar.TOP, solvers.TOP),  # TOP constant
    (grammar.BOT, solvers.BOT),  # BOT constant (empty join)
    (grammar.VAR(0), solvers.VAR(0)),  # Variable 0
    (grammar.VAR(1), solvers.VAR(1)),  # Variable 1
    # Lambda Abstractions
    # λx.x (identity)
    (grammar.ABS(grammar.VAR(0)), solvers.ABS(solvers.VAR(0))),
    # λx.y (constant function)
    (grammar.ABS(grammar.VAR(1)), solvers.ABS(solvers.VAR(1))),
    # Applications
    # x y
    (
        grammar.APP(grammar.VAR(0), grammar.VAR(1)),
        solvers.APP(solvers.VAR(0), solvers.VAR(1)),
    ),
    # Joins
    # x | y
    (
        grammar.JOIN(grammar.VAR(0), grammar.VAR(1)),
        solvers.JOIN(solvers.VAR(0), solvers.VAR(1)),
    ),
    # Complex nested terms
    # λx.x y
    (
        grammar.ABS(grammar.APP(grammar.VAR(0), grammar.VAR(1))),
        solvers.ABS(solvers.APP(solvers.VAR(0), solvers.VAR(1))),
    ),
    # λx.x y
    (
        grammar.ABS(grammar.APP(grammar.VAR(0), grammar.VAR(1))),
        solvers.ABS(solvers.APP(solvers.VAR(0), solvers.VAR(1))),
    ),
    # (λx.x) | (y z)
    (
        grammar.JOIN(
            grammar.ABS(grammar.VAR(0)), grammar.APP(grammar.VAR(1), grammar.VAR(2))
        ),
        solvers.JOIN(
            solvers.ABS(solvers.VAR(0)), solvers.APP(solvers.VAR(1), solvers.VAR(2))
        ),
    ),
    # Multiple joins
    # x | y | z
    (
        grammar.JOIN(grammar.VAR(0), grammar.VAR(1), grammar.VAR(2)),
        solvers.JOIN(solvers.VAR(0), solvers.JOIN(solvers.VAR(1), solvers.VAR(2))),
    ),
]
IDS = [repr(term) for term, _ in EXAMPLES]


@pytest.mark.parametrize("term, expr", EXAMPLES, ids=IDS)
def test_py_to_z3(term: grammar.Term, expr: z3.ExprRef) -> None:
    """Test conversion from Python term to Z3 expression."""
    result = py_to_z3(term)
    assert z3.eq(result, expr), f"Expected {expr}, but got {result}"


@pytest.mark.parametrize("term, expr", EXAMPLES, ids=IDS)
def test_z3_to_py(term: grammar.Term, expr: z3.ExprRef) -> None:
    """Test conversion from Z3 expression to Python term."""
    result = z3_to_py(expr)
    assert result == term, f"Expected {term}, but got {result}"
