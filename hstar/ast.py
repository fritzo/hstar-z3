"""
# Simple abstract syntax trees for λ-join-calculus.
"""

import enum
import inspect
import types
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache, singledispatch
from typing import Any, Optional

from .hashcons import HashConsMeta

_next_fresh = 0


class TermType(enum.Enum):
    """Enum representing the different types of terms in the λ-join-calculus."""

    TOP = "TOP"  # Top element
    BOT = "BOT"  # Bottom element
    VAR = "VAR"  # de Bruijn variable
    ABS = "ABS"  # Abstraction
    APP = "APP"  # Application
    COMP = "COMP"  # Function composition
    JOIN = "JOIN"  # Join wrt the Scott order
    _FRESH = "_FRESH"  # temporary for use in py_to_ast()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Term(metaclass=HashConsMeta):
    """A term in the λ-join-calculus."""

    type: TermType
    varname: int | None = None  # For VAR and FRESH
    body: Optional["Term"] = None  # For ABS
    lhs: Optional["Term"] = None  # For APP, JOIN, and COMP
    rhs: Optional["Term"] = None  # For APP, JOIN, and COMP

    def __call__(self, *args: Any) -> "Term":
        result = self
        for arg in args:
            result = APP(result, arg)
        return result

    def __or__(self, other: Any) -> "Term":
        return JOIN(self, other)

    def __ror__(self, other: Any) -> "Term":
        return JOIN(other, self)

    def __mul__(self, other: Any) -> "Term":
        return COMP(self, other)

    def __rmul__(self, other: Any) -> "Term":
        return COMP(other, self)

    def __rshift__(self, other: Any) -> "Term":
        return CONJ(self, other)

    def __rrshift__(self, other: Any) -> "Term":
        return CONJ(other, self)

    def __repr__(self) -> str:
        if self.type == TermType.TOP:
            return "TOP"
        elif self.type == TermType.BOT:
            return "BOT"
        elif self.type == TermType.VAR:
            return f"VAR({self.varname})"
        elif self.type == TermType.ABS:
            return f"ABS({self.body})"
        elif self.type == TermType.APP:
            return f"APP({self.lhs}, {self.rhs})"
        elif self.type == TermType.JOIN:
            return f"JOIN({self.lhs}, {self.rhs})"
        elif self.type == TermType.COMP:
            return f"COMP({self.lhs}, {self.rhs})"
        elif self.type == TermType._FRESH:
            return f"_FRESH({self.varname})"
        else:
            raise TypeError(f"Unknown term type: {self.type}")


TOP = Term(type=TermType.TOP)
"""The top term."""
BOT = Term(type=TermType.BOT)
"""The bottom term."""


def VAR(varname: int) -> Term:
    """Create a variable term."""
    return Term(type=TermType.VAR, varname=varname)


def ABS(body: Any) -> Term:
    """Create an abstraction term."""
    return Term(type=TermType.ABS, body=to_ast(body))


def APP(lhs: Any, rhs: Any) -> Term:
    """Create an application term."""
    return Term(type=TermType.APP, lhs=to_ast(lhs), rhs=to_ast(rhs))


def JOIN(lhs: Any, rhs: Any) -> Term:
    """Create a join term."""
    return Term(type=TermType.JOIN, lhs=to_ast(lhs), rhs=to_ast(rhs))


def COMP(lhs: Any, rhs: Any) -> Term:
    """Create a composition term (f ∘ g)."""
    return Term(type=TermType.COMP, lhs=to_ast(lhs), rhs=to_ast(rhs))


def CONJ(lhs: Any, rhs: Any) -> Term:
    """Create a conjunction term \f. rhs o f o lhs."""
    a = to_ast(lhs)
    b = to_ast(rhs)
    return to_ast(lambda f, x: b(f(a(x))))


def _FRESH() -> Term:
    """Create a fresh variable term with a unique ID."""
    global _next_fresh
    fresh_id = _next_fresh
    _next_fresh += 1
    return Term(type=TermType._FRESH, varname=fresh_id)


def shift(term: Term, cutoff: int = 0, delta: int = 1) -> Term:
    """Shift De Bruijn indices in a Term.

    This adjusts variable indices when a term is embedded inside another term,
    particularly inside a lambda abstraction.

    Args:
        term: The term to shift variables in
        cutoff: Variables with indices less than this are considered bound and
            won't be shifted
        delta: The amount to shift by

    Returns:
        A new term with shifted variable indices
    """
    if term.type in (TermType.TOP, TermType.BOT, TermType._FRESH):
        return term
    elif term.type == TermType.VAR:
        if term.varname is not None and term.varname >= cutoff:
            return VAR(term.varname + delta)
        return term
    elif term.type == TermType.ABS:
        if term.body is not None:
            return ABS(shift(term.body, cutoff + 1, delta))
        return term
    elif term.type == TermType.APP:
        if term.lhs is not None and term.rhs is not None:
            return APP(shift(term.lhs, cutoff, delta), shift(term.rhs, cutoff, delta))
        return term
    elif term.type == TermType.JOIN:
        if term.lhs is not None and term.rhs is not None:
            return JOIN(shift(term.lhs, cutoff, delta), shift(term.rhs, cutoff, delta))
        return term
    elif term.type == TermType.COMP:
        if term.lhs is not None and term.rhs is not None:
            return COMP(shift(term.lhs, cutoff, delta), shift(term.rhs, cutoff, delta))
        return term
    else:
        raise TypeError(f"Unknown term type: {term.type}")


def _fresh_to_var(term: Term, fresh_varname: int, depth: int = 0) -> Term:
    """Convert fresh variables to appropriate De Bruijn indices.

    Traverses a term and replaces FRESH variables with the appropriate VAR based
    on the abstraction depth at which they appear.

    Args:
        term: The term to process
        fresh_varname: The ID of the fresh variable to replace
        depth: Current abstraction depth (number of lambdas deep)

    Returns:
        A new term with fresh variables replaced by appropriate De Bruijn indices
    """
    if term.type == TermType._FRESH:
        if term.varname == fresh_varname:
            return VAR(depth)
        else:
            return term
    if term.type in (TermType.TOP, TermType.BOT, TermType.VAR):
        return term
    if term.type == TermType.ABS:
        assert term.body is not None
        return ABS(_fresh_to_var(term.body, fresh_varname, depth + 1))
    if term.type == TermType.APP:
        assert term.lhs is not None
        assert term.rhs is not None
        lhs = _fresh_to_var(term.lhs, fresh_varname, depth)
        rhs = _fresh_to_var(term.rhs, fresh_varname, depth)
        return APP(lhs, rhs)
    if term.type == TermType.JOIN:
        assert term.lhs is not None
        assert term.rhs is not None
        lhs = _fresh_to_var(term.lhs, fresh_varname, depth)
        rhs = _fresh_to_var(term.rhs, fresh_varname, depth)
        return JOIN(lhs, rhs)
    if term.type == TermType.COMP:
        assert term.lhs is not None
        assert term.rhs is not None
        lhs = _fresh_to_var(term.lhs, fresh_varname, depth)
        rhs = _fresh_to_var(term.rhs, fresh_varname, depth)
        return COMP(lhs, rhs)
    raise TypeError(f"Unknown term type: {term.type}")


@singledispatch
def to_ast(pythonic: Any) -> Term:
    """Convert a Python object to a Term in our AST using HOAS approach.

    Handles:
    - Term objects directly
    - Lambda functions -> ABS using Higher-Order Abstract Syntax
    """
    raise TypeError(f"Unsupported Python object type: {type(pythonic)}")


@to_ast.register
def _(pythonic: Term) -> Term:
    """Handle Term objects directly."""
    return pythonic


@to_ast.register(types.FunctionType)
def _(pythonic: types.FunctionType) -> Term:
    """Handle lambdas using HOAS approach."""
    # Apply fresh variables to the function
    sig = inspect.signature(pythonic)
    num_args = len(sig.parameters)
    fresh_vars = [_FRESH() for _ in range(num_args)]
    result = to_ast(pythonic(*fresh_vars))
    # Create lambda abstractions for each argument, from right to left
    for fresh in reversed(fresh_vars):
        assert fresh.varname is not None
        result = shift(result)
        result = _fresh_to_var(result, fresh.varname)
        result = ABS(result)
    return result


# Tuples.


def TUPLE(*args: Any) -> Term:
    """Barendregt encoding of tuples."""
    result = VAR(0)
    for arg in args:
        result = APP(result, shift(to_ast(arg)))
    return ABS(result)


def select(size: int, index: int) -> Term:
    """Returns a zero-based tuple selector."""
    result = VAR(size - index - 1)
    for _ in range(size):
        result = ABS(result)
    return result


@to_ast.register(tuple)
@to_ast.register(list)
def _(pythonic: Iterable[Any]) -> Term:
    """Convert a tuple or list to a Term."""
    return TUPLE(*(to_ast(arg) for arg in pythonic))


# Numerals: mu x. 1 + x.


def zero() -> Term:
    """Barendregt encoding of zero."""
    return ABS(ABS(VAR(1)))


def succ(term: Term) -> Term:
    """Barendregt encoding of the successor function."""
    return ABS(ABS(APP(VAR(0), term)))


@to_ast.register(int)
@cache
def _(pythonic: int) -> Term:
    """Convert an integer to a Term."""
    if pythonic == 0:
        return zero()
    else:
        return succ(to_ast(pythonic - 1))
