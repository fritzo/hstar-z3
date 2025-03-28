"""
# Grammars for Syntax-Guided Synthesis (SyGuS).
"""

from collections.abc import Iterator
from dataclasses import dataclass

import z3
from z3 import ForAll, MultiPattern

from . import language
from .language import Term


@dataclass
class Grammar:
    sort: z3.SortRef
    eval: z3.ExprRef
    """A function from Grammar.sort to Term."""
    eval_theory: list[z3.ExprRef]


# Declare a datatype for combinators
Comb: z3.SortRef
Comb = z3.Datatype("Comb")
Comb.declare("S")
Comb.declare("K")
Comb.declare("J")
Comb.declare("APP", ("app_lhs", Comb), ("app_rhs", Comb))
Comb = Comb.create()

# Declare a conversion from Comb to Term
EvalComb = z3.Function("EvalComb", Comb, Term)


def combinator_theory() -> Iterator[z3.ExprRef]:
    """Theory for evaluating combinators."""
    x, y = z3.Consts("x y", Comb)
    yield EvalComb(Comb.S) == language.S
    yield EvalComb(Comb.K) == language.K
    yield EvalComb(Comb.J) == language.J
    yield ForAll(
        [x, y],
        EvalComb(Comb.APP(x, y)) == language.APP(EvalComb(x), EvalComb(y)),
        patterns=[
            EvalComb(Comb.APP(x, y)),
            MultiPattern(Comb.APP(x, y), language.APP(EvalComb(x), EvalComb(y))),
        ],
        qid="eval_comb_app",
    )


comb_grammar = Grammar(Comb, EvalComb, list(combinator_theory()))
