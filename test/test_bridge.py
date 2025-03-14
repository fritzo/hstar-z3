import logging

import pytest
import z3

from hstar import ast, normal, solvers
from hstar.bridge import ast_to_nf, nf_to_z3, z3_to_nf

logger = logging.getLogger(__name__)

# A list of examples with Python terms and their corresponding Z3 expressions
EXAMPLES: list[tuple[normal.Term, z3.ExprRef]] = [
    # Basic Terms
    (normal.TOP, solvers.TOP),  # TOP constant
    (normal.BOT, solvers.BOT),  # BOT constant (empty join)
    (normal.VAR(0), solvers.VAR(0)),  # Variable 0
    (normal.VAR(1), solvers.VAR(1)),  # Variable 1
    # Lambda Abstractions
    # λx.x (identity)
    (normal.ABS(normal.VAR(0)), solvers.ABS(solvers.VAR(0))),
    # λx.y (constant function)
    (normal.ABS(normal.VAR(1)), solvers.ABS(solvers.VAR(1))),
    # Applications
    # x y
    (
        normal.APP(normal.VAR(0), normal.VAR(1)),
        solvers.APP(solvers.VAR(0), solvers.VAR(1)),
    ),
    # Joins
    # x | y
    (
        normal.JOIN(normal.VAR(0), normal.VAR(1)),
        solvers.JOIN(solvers.VAR(0), solvers.VAR(1)),
    ),
    # Complex nested terms
    # λx.x y
    (
        normal.ABS(normal.APP(normal.VAR(0), normal.VAR(1))),
        solvers.ABS(solvers.APP(solvers.VAR(0), solvers.VAR(1))),
    ),
    # λx.x y
    (
        normal.ABS(normal.APP(normal.VAR(0), normal.VAR(1))),
        solvers.ABS(solvers.APP(solvers.VAR(0), solvers.VAR(1))),
    ),
    # (λx.x) | (y z)
    (
        normal.JOIN(
            normal.ABS(normal.VAR(0)), normal.APP(normal.VAR(1), normal.VAR(2))
        ),
        solvers.JOIN(
            solvers.ABS(solvers.VAR(0)), solvers.APP(solvers.VAR(1), solvers.VAR(2))
        ),
    ),
    # Multiple joins
    # x | y | z
    (
        normal.JOIN(normal.VAR(0), normal.VAR(1), normal.VAR(2)),
        solvers.JOIN(solvers.VAR(0), solvers.JOIN(solvers.VAR(1), solvers.VAR(2))),
    ),
]
IDS = [repr(term) for term, _ in EXAMPLES]


@pytest.mark.parametrize("term, expr", EXAMPLES, ids=IDS)
def test_py_to_z3(term: normal.Term, expr: z3.ExprRef) -> None:
    """Test conversion from Python term to Z3 expression."""
    result = nf_to_z3(term)
    assert z3.eq(result, expr), f"Expected {expr}, but got {result}"


@pytest.mark.parametrize("term, expr", EXAMPLES, ids=IDS)
def test_z3_to_py(term: normal.Term, expr: z3.ExprRef) -> None:
    """Test conversion from Z3 expression to Python term."""
    result = z3_to_nf(expr)
    assert result == term, f"Expected {term}, but got {result}"


# Examples for testing ast_to_nf conversion
AST_TO_NF_EXAMPLES: list[tuple[ast.Term, normal.Term]] = [
    # Basic terms
    (ast.TOP, normal.TOP),  # TOP
    (ast.BOT, normal.BOT),  # BOT
    (ast.VAR(0), normal.VAR(0)),  # Variable 0
    (ast.VAR(1), normal.VAR(1)),  # Variable 1
    # Lambda abstractions
    (ast.ABS(ast.VAR(0)), normal.ABS(normal.VAR(0))),  # λx.x (identity)
    (ast.ABS(ast.VAR(1)), normal.ABS(normal.VAR(1))),  # λx.y (constant function)
    # Applications
    (ast.APP(ast.VAR(0), ast.VAR(1)), normal.APP(normal.VAR(0), normal.VAR(1))),  # x y
    # Joins
    (
        ast.JOIN(ast.VAR(0), ast.VAR(1)),
        normal.JOIN(normal.VAR(0), normal.VAR(1)),
    ),  # x | y
    # Composition (converts to λx.f(g(x)))
    (
        ast.COMP(ast.VAR(0), ast.VAR(1)),
        # λx.0(1(x))
        normal.ABS(normal.APP(normal.VAR(1), normal.APP(normal.VAR(2), normal.VAR(0)))),
    ),
    # Complex nested terms
    (
        ast.ABS(ast.APP(ast.VAR(0), ast.VAR(1))),
        normal.ABS(normal.APP(normal.VAR(0), normal.VAR(1))),
    ),  # λx.x y
    # (λx.x) | (y z)
    (
        ast.JOIN(ast.ABS(ast.VAR(0)), ast.APP(ast.VAR(1), ast.VAR(2))),
        normal.JOIN(
            normal.ABS(normal.VAR(0)), normal.APP(normal.VAR(1), normal.VAR(2))
        ),
    ),
    # Multiple joins - x | y | z
    (
        ast.JOIN(ast.JOIN(ast.VAR(0), ast.VAR(1)), ast.VAR(2)),
        normal.JOIN(normal.VAR(0), normal.VAR(1), normal.VAR(2)),
    ),
]

AST_NF_IDS = [repr(ast_term) for ast_term, _ in AST_TO_NF_EXAMPLES]


@pytest.mark.parametrize("ast_term, nf_term", AST_TO_NF_EXAMPLES, ids=AST_NF_IDS)
def test_ast_to_nf(ast_term: ast.Term, nf_term: normal.Term) -> None:
    """Test conversion from ast.Term to normal.Term."""
    result = ast_to_nf(ast_term)
    assert result == nf_term, f"Expected {nf_term}, but got {result}"
