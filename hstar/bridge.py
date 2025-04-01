"""
# Bridge between λ-join-calculus terms and Z3 terms.

This module provides conversion functions between the term representations in
normal.py (Python objects) and the Z3 terms in solvers.py (symbolic expressions).
"""

import z3

from hstar import ast, language, normal

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
            # Handle COMP constructor - we'll convert to a lambda term
            # COMP(f, g) = λx. f(g(x))
            f = normal.shift(z3_to_nf(term.arg(0)))
            g = normal.shift(z3_to_nf(term.arg(1)))
            f_g_x = normal.APP(f, normal.APP(g, normal.VAR(0)))
            return normal.ABS(f_g_x)

    raise InvalidExpr(f"Unexpected Z3 term: {term}")


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
