import logging

import pytest
import z3

from hstar import ast, language, normal
from hstar.ast import to_ast
from hstar.bridge import (
    ast_to_nf,
    grid_to_nf,
    nf_to_ast,
    nf_to_grid,
    nf_to_int,
    nf_to_z3,
    z3_to_nf,
)

logger = logging.getLogger(__name__)

# A list of examples with Python terms and their corresponding Z3 expressions
EXAMPLES: list[tuple[normal.Term, z3.ExprRef]] = [
    # Basic Terms
    (normal.TOP, language.TOP),  # TOP constant
    (normal.BOT, language.BOT),  # BOT constant (empty join)
    (normal.VAR(0), language.VAR(0)),  # Variable 0
    (normal.VAR(1), language.VAR(1)),  # Variable 1
    # Lambda Abstractions
    # λx.x (identity)
    (normal.ABS(normal.VAR(0)), language.I),
    # λx.y (constant function)
    (normal.ABS(normal.VAR(1)), language.APP(language.K, language.VAR(0))),
    # Applications
    # x y
    (
        normal.APP(normal.VAR(0), normal.VAR(1)),
        language.APP(language.VAR(0), language.VAR(1)),
    ),
    # Joins
    # x | y
    (
        normal.JOIN(normal.VAR(0), normal.VAR(1)),
        language.JOIN(language.VAR(0), language.VAR(1)),
    ),
    # Complex nested terms
    # λx.x y
    (
        normal.ABS(normal.APP(normal.VAR(0), normal.VAR(1))),
        language.app(language.C, language.I, language.VAR(0)),
    ),
    # (λx.x) | (y z)
    (
        normal.JOIN(
            normal.ABS(normal.VAR(0)), normal.APP(normal.VAR(1), normal.VAR(2))
        ),
        language.JOIN(language.I, language.APP(language.VAR(1), language.VAR(2))),
    ),
    # Multiple joins
    # x | y | z
    (
        normal.JOIN(normal.VAR(0), normal.VAR(1), normal.VAR(2)),
        language.JOIN(language.VAR(0), language.JOIN(language.VAR(1), language.VAR(2))),
    ),
    # A pair
    # <BOT, λx. y>
    (
        normal.ABS(
            normal.APP(normal.APP(normal.VAR(0), normal.BOT), normal.ABS(normal.VAR(2)))
        ),
        language.app(
            language.C,
            language.app(language.C, language.I, language.BOT),
            language.app(language.K, language.VAR(0)),
        ),
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
    assert result == nf_term


@pytest.mark.parametrize("ast_term, nf_term", AST_TO_NF_EXAMPLES, ids=AST_NF_IDS)
def test_nf_to_ast(nf_term: normal.Term, ast_term: ast.Term) -> None:
    """Test conversion from normal.Term to ast.Term."""
    actual_ast = nf_to_ast(nf_term)
    actual_nf = ast_to_nf(actual_ast)
    assert actual_nf == nf_term  # weaker than assert actual_ast == ast_term


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_nf_to_int(n: int) -> None:
    """Test conversion from normal.Term to int."""
    nf = ast_to_nf(to_ast(n))
    assert nf_to_int(nf) == n


EXAMPLE_GRIDS = [
    [[0]],
    [[1]],
    [[2, 3]],
    [[4], [5], [6]],
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ],
]


@pytest.mark.parametrize("grid", EXAMPLE_GRIDS)
def test_grid_to_nf(grid: list[list[int]]) -> None:
    """Test conversion from grid to normal.Term."""
    nf = grid_to_nf(grid)
    assert not nf.free_vars
    grid2 = nf_to_grid(nf)
    assert grid2 == grid
