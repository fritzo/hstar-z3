"""
# Bridge between λ-join-calculus terms in various representations.

This module provides conversion functions between the term representations in
normal.py (Python objects) and the Z3 terms in solvers.py (symbolic expressions).
"""

from functools import lru_cache

import z3

from . import ast, language, normal
from .functools import weak_key_cache


@weak_key_cache
def nf_to_z3(term: normal.Term) -> z3.ExprRef:
    """
    Convert a normal.Term normal form to a Z3 term.

    Args:
        term: A normal.Term instance

    Returns:
        A Z3 term expression
    """
    # Special case for BOT (empty join)
    if not term.parts:
        return language.BOT

    # If there's just one part, convert it directly
    if len(term.parts) == 1:
        part = next(iter(term.parts))
        return _nf_to_z3(part)

    # Sort parts for deterministic ordering.
    # TODO sort by number of free variables, for efficient abstraction.
    sorted_parts = sorted(term.parts)

    # Build the JOIN tree in right-associative order (last two elements first)
    result = _nf_to_z3(sorted_parts[-1])
    for part in reversed(sorted_parts[1:-1]):
        result = language.JOIN(_nf_to_z3(part), result)
    return language.JOIN(_nf_to_z3(sorted_parts[0]), result)

    # This should never happen (length would be 0 or 1, handled above)
    raise ValueError("Unexpected term parts length")


def _z3_unshift(term: z3.ExprRef) -> z3.ExprRef:
    subs = [
        (v, language.VAR(z3.get_var_index(v) - 1)) for v in language.free_vars(term)
    ]
    if subs:
        term = z3.substitute(term, *subs)
    return term


def _nf_to_z3(term: normal._Term) -> z3.ExprRef:
    """
    Convert a normal._Term normal form to a Z3 term.

    Args:
        term: A normal._Term instance

    Returns:
        A Z3 term expression
    """
    if term.typ == normal.TermType.TOP:
        return language.TOP

    elif term.typ == normal.TermType.VAR:
        return language.VAR(term.varname)

    elif term.typ == normal.TermType.ABS:
        assert term.head is not None
        body_z3 = _nf_to_z3(term.head)
        result = language.lam(language.VAR(0), body_z3)
        return _z3_unshift(result)

    elif term.typ == normal.TermType.APP:
        assert term.head is not None
        assert term.body is not None
        head_z3 = _nf_to_z3(term.head)
        body_z3 = nf_to_z3(term.body)
        return language.APP(head_z3, body_z3)

    raise ValueError(f"Unexpected term type: {term.typ}")


class InvalidExpr(Exception):
    pass


def _z3_to_nf_const() -> dict[z3.ExprRef, normal.Term]:
    from hstar.normal import ABS, BOT, JOIN, TOP, VAR, app

    v0 = VAR(0)
    v1 = VAR(1)
    v2 = VAR(2)

    let: dict[z3.ExprRef, normal.Term] = {}
    let[language.TOP] = TOP
    let[language.BOT] = BOT
    let[language.I] = ABS(v0)
    let[language.K] = ABS(ABS(v1))
    let[language.KI] = ABS(ABS(v0))
    let[language.J] = ABS(ABS(JOIN(v0, v1)))
    let[language.B] = ABS(ABS(ABS(app(v2, app(v1, v0)))))
    let[language.C] = ABS(ABS(ABS(app(v2, v0, v1))))
    let[language.CI] = CI = ABS(ABS(app(v0, v1)))
    let[language.CB] = ABS(ABS(ABS(app(v1, app(v2, v0)))))
    let[language.W] = ABS(ABS(app(v1, v0, v0)))
    let[language.S] = ABS(ABS(ABS(app(v2, v0, app(v1, v0)))))
    lam_x_y_xx = ABS(app(v1, app(v0, v0)))
    let[language.Y] = Y = ABS(app(lam_x_y_xx, lam_x_y_xx))
    let[language.V] = V = ABS(app(Y, ABS(ABS(JOIN(v0, app(v2, app(v1, v0)))))))
    let[language.DIV] = app(V, app(CI, TOP))
    # TODO define A

    return let


_Z3_TO_NF_CONST = _z3_to_nf_const()


def z3_to_nf(term: z3.ExprRef) -> normal.Term:
    """
    Convert a ground Z3 term to a normal.Term normal form.

    Args:
        term: A Z3 term expression

    Returns:
        A normal.Term instance
    """
    if z3.is_const(term):
        # Handle constants
        const = _Z3_TO_NF_CONST.get(term)
        if const is not None:
            return const

    if z3.is_var(term):
        # Handle VAR constructor
        idx = z3.get_var_index(term)
        return normal.VAR(idx)

    if z3.is_app(term):
        decl = term.decl()
        decl_name = str(decl)

        if decl_name == "APP":
            # Handle APP constructor
            lhs = term.arg(0)
            rhs = term.arg(1)
            return normal.APP(z3_to_nf(lhs), z3_to_nf(rhs))

        elif decl_name == "JOIN":
            # Handle JOIN constructor
            lhs = term.arg(0)
            rhs = term.arg(1)
            return normal.JOIN(z3_to_nf(lhs), z3_to_nf(rhs))

        elif decl_name == "COMP":
            # Handle COMP constructor
            lhs = term.arg(0)
            rhs = term.arg(1)
            return normal.comp(z3_to_nf(lhs), z3_to_nf(rhs))

    raise InvalidExpr(f"Unexpected Z3 term: {term}")


@lru_cache
def ast_to_nf(term: ast.Term) -> normal.Term:
    """
    Convert an ast.Term to a normal.Term.

    Args:
        term: An ast.Term instance

    Returns:
        A normal.Term instance
    """
    if term.type == ast.TermType.TOP:
        return normal.TOP

    elif term.type == ast.TermType.BOT:
        return normal.BOT

    elif term.type == ast.TermType.VAR:
        assert term.varname is not None
        return normal.VAR(term.varname)

    elif term.type == ast.TermType.ABS:
        assert term.body is not None
        body = ast_to_nf(term.body)
        return normal.ABS(body)

    elif term.type == ast.TermType.APP:
        assert term.lhs is not None
        assert term.rhs is not None
        lhs = ast_to_nf(term.lhs)
        rhs = ast_to_nf(term.rhs)
        return normal.APP(lhs, rhs)

    elif term.type == ast.TermType.JOIN:
        assert term.lhs is not None
        assert term.rhs is not None
        lhs = ast_to_nf(term.lhs)
        rhs = ast_to_nf(term.rhs)
        return normal.JOIN(lhs, rhs)

    elif term.type == ast.TermType.COMP:
        # Function composition (f ∘ g) is represented as λx. f(g(x))
        assert term.lhs is not None
        assert term.rhs is not None
        f = normal.shift(ast_to_nf(term.lhs))
        g = normal.shift(ast_to_nf(term.rhs))
        x = normal.VAR(0)
        return normal.ABS(normal.APP(f, normal.APP(g, x)))

    elif term.type == ast.TermType._FRESH:
        # _FRESH is a temporary term used during conversion from Python functions
        # It shouldn't appear in final AST terms being converted to normal form
        raise ValueError(
            f"Cannot convert _FRESH term with ID {term.varname} to normal form"
        )

    raise ValueError(f"Unexpected term type: {term.type}")


def ast_to_z3(term: ast.Term) -> z3.ExprRef:
    """
    Convert an ast.Term to a Z3 term.

    Args:
        term: An ast.Term instance

    Returns:
        A Z3 term expression
    """
    return nf_to_z3(ast_to_nf(term))


@lru_cache
def nf_to_ast(term: normal.Term) -> ast.Term:
    """
    Convert a normal.Term to an ast.Term.

    Args:
        term: A normal.Term instance

    Returns:
        An ast.Term instance
    """
    # Handle BOT (empty parts)
    if not term.parts:
        return ast.BOT

    # Handle single-part terms
    if len(term.parts) == 1:
        part = next(iter(term.parts))
        return _nf_to_ast(part)

    # Handle multi-part terms (JOIN)
    parts = sorted(term.parts)
    result = ast.JOIN(_nf_to_ast(parts[0]), _nf_to_ast(parts[1]))
    for part in parts[2:]:
        result = ast.JOIN(result, _nf_to_ast(part))
    return result


def _nf_to_ast(part: normal._Term) -> ast.Term:
    """Convert a single normal._Term to an ast.Term."""
    if part.typ == normal.TermType.TOP:
        return ast.TOP

    elif part.typ == normal.TermType.VAR:
        return ast.VAR(part.varname)

    elif part.typ == normal.TermType.ABS:
        assert part.head is not None
        body = _nf_to_ast(part.head)
        return ast.ABS(body)

    elif part.typ == normal.TermType.APP:
        assert part.head is not None
        assert part.body is not None
        lhs = _nf_to_ast(part.head)
        rhs = nf_to_ast(part.body)  # part.body is a normal.Term not a _Term
        return ast.APP(lhs, rhs)

    raise ValueError(f"Unexpected term type: {part.typ}")


def grid_to_ast(grid: list[list[int]]) -> ast.Term:
    """Convert a rectangular grid of numbers to a Term (n_rows, n_cols, grid)."""
    n_rows = len(grid)
    n_cols = len(grid[0])
    assert all(len(row) == n_cols for row in grid)
    return ast.TUPLE(n_rows, n_cols, grid)


def nf_to_int(term: normal.Term) -> int:
    assert not term.free_vars
    TOP = normal.TOP
    BOT = normal.BOT
    I = normal.ABS(normal.VAR(0))
    app = normal.app
    if app(term, TOP, BOT) == TOP:
        return 0
    if app(term, BOT, TOP) == TOP:
        return 1 + nf_to_int(app(term, TOP, I))
    raise ValueError(f"Unexpected term: {term}")


def nf_to_list_int(size: int, term: normal.Term) -> list[int]:
    APP = normal.APP
    return [nf_to_int(APP(term, ast_to_nf(ast.select(size, i)))) for i in range(size)]


def nf_to_list_list_int(n_rows: int, n_cols: int, term: normal.Term) -> list[list[int]]:
    APP = normal.APP

    def select(i: int) -> normal.Term:
        return ast_to_nf(ast.select(n_rows, i))

    return [nf_to_list_int(n_cols, APP(term, select(i))) for i in range(n_rows)]


def nf_to_grid(term: normal.Term) -> list[list[int]]:
    """Convert a Term (n_rows, n_cols, grid) to a rectangular grid of numbers."""
    APP = normal.APP
    n_rows = nf_to_int(APP(term, ast_to_nf(ast.select(3, 0))))
    n_cols = nf_to_int(APP(term, ast_to_nf(ast.select(3, 1))))
    data = APP(term, ast_to_nf(ast.select(3, 2)))
    return nf_to_list_list_int(n_rows, n_cols, data)


def grid_to_nf(grid: list[list[int]]) -> normal.Term:
    """Convert a rectangular grid of numbers to a Term (n_rows, n_cols, grid)."""
    return ast_to_nf(grid_to_ast(grid))
