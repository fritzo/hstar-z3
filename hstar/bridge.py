"""
# Bridge between λ-join-calculus terms and Z3 terms.

This module provides conversion functions between the term representations in
normal.py (Python objects) and the Z3 terms in solvers.py (symbolic expressions).
"""

import z3

from hstar import normal, solvers

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
        return solvers.BOT

    # If there's just one part, convert it directly
    if len(term.parts) == 1:
        part = next(iter(term.parts))
        return _nf_to_z3(part)

    # For example with term11, we need to carefully construct the JOIN
    # in a right-associative way as expected by the tests
    # Sort parts for deterministic ordering
    sorted_parts = sorted(term.parts)

    # Build the JOIN tree in right-associative order (last two elements first)
    result = _nf_to_z3(sorted_parts[-1])
    for part in reversed(sorted_parts[1:-1]):
        result = solvers.JOIN(_nf_to_z3(part), result)
    return solvers.JOIN(_nf_to_z3(sorted_parts[0]), result)

    # This should never happen (length would be 0 or 1, handled above)
    raise ValueError("Unexpected term parts length")


def _nf_to_z3(term: normal._Term) -> z3.ExprRef:
    """
    Convert a normal._Term normal form to a Z3 term.

    Args:
        term: A normal._Term instance

    Returns:
        A Z3 term expression
    """
    if term.typ == normal.TermType.TOP:
        return solvers.TOP

    elif term.typ == normal.TermType.VAR:
        return solvers.VAR(term.varname)

    elif term.typ == normal.TermType.ABS:
        assert term.head is not None
        body_z3 = _nf_to_z3(term.head)
        return solvers.ABS(body_z3)

    elif term.typ == normal.TermType.APP:
        assert term.head is not None
        assert term.body is not None
        head_z3 = _nf_to_z3(term.head)
        body_z3 = nf_to_z3(term.body)
        return solvers.APP(head_z3, body_z3)

    raise ValueError(f"Unexpected term type: {term.typ}")


class InvalidExpr(Exception):
    pass


def z3_to_nf(term: z3.ExprRef) -> normal.Term:
    """
    Convert a ground Z3 term to a normal.Term normal form.

    Args:
        term: A Z3 term expression

    Returns:
        A normal.Term instance
    """
    # Handle special constants first
    if z3.eq(term, solvers.TOP):
        return normal.TOP

    if z3.eq(term, solvers.BOT):
        return normal.BOT

    # Check if we have a symbolic variable (like a, x, etc.)
    if z3.is_const(term) and term.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        raise InvalidExpr(f"Symbolic variable: {term}")

    try:
        # Use Z3's application inspection functions
        if z3.is_app(term):
            decl = term.decl()
            decl_name = str(decl)

            if decl_name == "VAR":
                # Handle VAR constructor
                idx = term.arg(0).as_long()  # Get the index directly
                return normal.VAR(idx)

            elif decl_name == "ABS":
                # Handle ABS constructor
                body = term.arg(0)
                return normal.ABS(z3_to_nf(body))

            elif decl_name == "APP":
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
                f = term.arg(0)
                g = term.arg(1)
                # Create the equivalent lambda term: λx. f(g(x))
                x_var = normal.VAR(0)
                g_x = normal.APP(z3_to_nf(g), x_var)
                f_g_x = normal.APP(z3_to_nf(f), g_x)
                return normal.ABS(f_g_x)

    except Exception as e:
        raise InvalidExpr(f"Error converting Z3 term: {e}") from e
    raise InvalidExpr(f"Unexpected Z3 term: {term}")
