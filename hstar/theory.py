"""
# Theory of λ-join-calculus expressions with Z3.

This uses a de Bruijn indexed representation of λ-join-calculus terms, with a
`LEQ` relation for the Scott ordering, and explicit `BOT` (bottom), `TOP` (top),
and binary `JOIN` operation wrt the Scott ordering.

The theory includes de Bruijn syntax, Scott ordering, λ-calculus, and
types-as-closures.
"""

import logging
from collections.abc import Callable, Iterable, Iterator

import z3
from z3 import And, ExprRef, ForAll, If, Implies, MultiPattern, Not, Or

from .language import (
    ABS,
    APP,
    BOT,
    CB,
    CI,
    COMP,
    DIV,
    JOIN,
    KI,
    LEQ,
    OFTYPE,
    SHIFT,
    SIMPLE,
    SUBST,
    TOP,
    TUPLE,
    V_,
    VAR,
    Y_,
    B,
    C,
    I,
    J,
    K,
    QEHindley,
    S,
    Term,
    V,
    W,
    Y,
    app,
    bool_,
    boool,
    semi,
    unit,
)
from .metrics import COUNTERS

logger = logging.getLogger(__name__)
counter = COUNTERS[__name__]

v0 = VAR(0)
v1 = VAR(1)
f, g, h = z3.Consts("f g h", Term)
r, s, t = z3.Consts("r s t", Term)
x, y, z = z3.Consts("x y z", Term)


def de_bruijn_theory() -> Iterator[ExprRef]:
    """Theory of de Bruijn operations SHIFT and SUBST."""
    i = z3.Int("i")
    j = z3.Int("j")
    body = z3.Const("body", Term)
    lhs = z3.Const("lhs", Term)
    rhs = z3.Const("rhs", Term)
    start = z3.Int("start")
    delta = z3.Int("delta")

    # SHIFT axioms, deterministically executed forwards.
    yield ForAll(
        [i, start, delta],
        SHIFT(VAR(i), start, delta) == VAR(If(i >= start, i + delta, i)),
        patterns=[SHIFT(VAR(i), start, delta)],
        qid="shift_var",
    )
    yield ForAll(
        [x, start, delta],
        SHIFT(ABS(x), start, delta) == ABS(SHIFT(x, start + 1, delta)),
        patterns=[SHIFT(ABS(x), start, delta)],
        qid="shift_abs",
    )
    yield ForAll(
        [lhs, rhs, start, delta],
        SHIFT(APP(lhs, rhs), start, delta)
        == APP(SHIFT(lhs, start, delta), SHIFT(rhs, start, delta)),
        patterns=[SHIFT(APP(lhs, rhs), start, delta)],
        qid="shift_app",
    )
    yield ForAll(
        [lhs, rhs, start, delta],
        SHIFT(JOIN(lhs, rhs), start, delta)
        == JOIN(SHIFT(lhs, start, delta), SHIFT(rhs, start, delta)),
        patterns=[SHIFT(JOIN(lhs, rhs), start, delta)],
        qid="shift_join",
    )
    yield ForAll(
        [lhs, rhs, start, delta],
        SHIFT(COMP(lhs, rhs), start, delta)
        == COMP(SHIFT(lhs, start, delta), SHIFT(rhs, start, delta)),
        patterns=[SHIFT(COMP(lhs, rhs), start, delta)],
        qid="shift_comp",
    )
    yield ForAll([start, delta], SHIFT(TOP, start, delta) == TOP, qid="shift_top")
    yield ForAll([start, delta], SHIFT(BOT, start, delta) == BOT, qid="shift_bot")

    # SUBST axioms, deterministically executed forwards.
    yield ForAll(
        [j, i, x],
        SUBST(i, x, VAR(j)) == If(j == i, x, VAR(j)),
        patterns=[SUBST(i, x, VAR(j))],
        qid="subst_var",
    )
    yield ForAll(
        [body, i, x],
        SUBST(i, x, ABS(body)) == ABS(SUBST(i + 1, SHIFT(x, 0, 1), body)),
        patterns=[SUBST(i, x, ABS(body))],
        qid="subst_abs",
    )
    yield ForAll(
        [lhs, rhs, i, x],
        SUBST(i, x, APP(lhs, rhs)) == APP(SUBST(i, x, lhs), SUBST(i, x, rhs)),
        patterns=[SUBST(i, x, APP(lhs, rhs))],
        qid="subst_app",
    )
    yield ForAll(
        [lhs, rhs, i, x],
        SUBST(i, x, JOIN(lhs, rhs)) == JOIN(SUBST(i, x, lhs), SUBST(i, x, rhs)),
        patterns=[SUBST(i, x, JOIN(lhs, rhs))],
        qid="subst_join",
    )
    yield ForAll(
        [lhs, rhs, i, x],
        SUBST(i, x, COMP(lhs, rhs)) == COMP(SUBST(i, x, lhs), SUBST(i, x, rhs)),
        patterns=[SUBST(i, x, COMP(lhs, rhs))],
        qid="subst_comp",
    )
    yield ForAll([i, x], SUBST(i, x, TOP) == TOP, qid="subst_top")
    yield ForAll([i, x], SUBST(i, x, BOT) == BOT, qid="subst_bot")


# Theory of Scott ordering.
def order_theory() -> Iterator[ExprRef]:
    """Theory of Scott ordering and join."""
    # Basic order axioms
    yield ForAll([x], LEQ(x, TOP), qid="leq_top")
    yield ForAll([x], LEQ(BOT, x), qid="leq_bot")
    yield ForAll([x], LEQ(x, x), qid="leq_reflexive")
    yield ForAll(
        [x, y],
        And(LEQ(x, y), LEQ(y, x)) == (x == y),
        patterns=[MultiPattern(LEQ(x, y), LEQ(y, x))],
        qid="leq_antisym",
    )
    yield ForAll(
        [x, y, z],
        Implies(And(LEQ(x, y), LEQ(y, z)), LEQ(x, z)),
        patterns=[MultiPattern(LEQ(x, y), LEQ(y, z), LEQ(x, z))],
        qid="leq_trans",
    )
    yield Not(LEQ(TOP, BOT))

    # JOIN is least upper bound
    yield ForAll([x, y], LEQ(x, JOIN(x, y)), qid="leq_join")
    yield ForAll(
        [x, y, z],
        And(LEQ(x, z), LEQ(y, z)) == LEQ(JOIN(x, y), z),
        qid="join_lub",
    )  # Least upper bound property

    # JOIN is associative, commutative, and idempotent
    yield ForAll([x, y], JOIN(x, y) == JOIN(y, x), qid="join_commute")
    yield ForAll(
        [x, y, z],
        JOIN(x, JOIN(y, z)) == JOIN(JOIN(x, y), z),
        patterns=[
            MultiPattern(JOIN(x, JOIN(y, z)), JOIN(x, y)),
            MultiPattern(JOIN(y, z), JOIN(JOIN(x, y), z)),
        ],
        qid="join_assoc",
    )
    yield ForAll([x], JOIN(x, x) == x, qid="join_idem")

    # JOIN with BOT/TOP
    yield ForAll([x], JOIN(x, BOT) == x, qid="join_bot")  # BOT is identity
    yield ForAll([x], JOIN(x, TOP) == TOP, qid="join_top")  # TOP absorbs


def lambda_theory() -> Iterator[ExprRef]:
    """Theory of lambda calculus and combinators."""
    i = z3.Int("i")

    # Composition properties
    yield ForAll(
        [f, g, x],
        APP(COMP(f, g), x) == APP(f, APP(g, x)),
        patterns=[
            MultiPattern(APP(COMP(f, g), x), APP(g, x)),
            MultiPattern(COMP(f, g), APP(f, APP(g, x))),
        ],
        qid="beta_comp",
    )
    yield ForAll([f], COMP(f, I) == f, qid="comp_id_right")
    yield ForAll([f], COMP(I, f) == f, qid="comp_id_left")
    yield ForAll([f], COMP(BOT, f) == BOT, qid="comp_bot")
    yield ForAll([f], COMP(TOP, f) == TOP, qid="comp_top")

    # Composition is associative
    yield ForAll(
        [f, g, h],
        COMP(f, COMP(g, h)) == COMP(COMP(f, g), h),
        patterns=[
            MultiPattern(COMP(f, COMP(g, h)), COMP(f, g)),
            MultiPattern(COMP(COMP(f, g), h), COMP(g, h)),
        ],
        qid="comp_assoc",
    )

    # Composition is monotonic in both arguments
    yield ForAll(
        [f, g, h],
        Implies(LEQ(f, g), LEQ(COMP(f, h), COMP(g, h))),
        patterns=[
            MultiPattern(LEQ(f, g), COMP(f, h), COMP(g, h)),
            LEQ(COMP(f, h), COMP(g, h)),
        ],
        qid="comp_mono_left",
    )
    yield ForAll(
        [f, g, h],
        Implies(LEQ(g, h), LEQ(COMP(f, g), COMP(f, h))),
        patterns=[
            MultiPattern(LEQ(g, h), COMP(f, g), COMP(f, h)),
            LEQ(COMP(f, g), COMP(f, h)),
        ],
        qid="comp_mono_right",
    )

    # Combinator equations
    yield KI == app(K, I)
    yield CB == app(C, B)
    yield CI == app(C, I)
    yield J == app(C, J)
    yield I == app(W, J)
    yield Y == Y_
    yield V == V_

    # Beta reduction of combinators
    yield ForAll([x], app(BOT, x) == BOT, qid="beta_bot")
    yield ForAll([x], app(TOP, x) == TOP, qid="beta_top")
    yield ForAll([x], app(I, x) == x, qid="beta_i")
    yield ForAll(
        [x, y],
        app(K, x, y) == x,
        qid="beta_k",
    )
    yield ForAll(
        [x, y],
        app(KI, x, y) == y,
        qid="beta_ki",
    )
    yield ForAll(
        [x, y],
        app(J, x, y) == JOIN(x, y),
        patterns=[
            app(J, x, y),
            MultiPattern(app(J, x), JOIN(x, y)),
        ],
        qid="beta_j",
    )
    yield ForAll(
        [x, y],
        app(CI, x, y) == app(y, x),
        patterns=[
            MultiPattern(app(CI, x, y)),
            MultiPattern(app(CI, x), app(y, x)),
        ],
        qid="beta_ci",
    )
    yield ForAll(
        [x, y],
        app(B, x, y) == COMP(x, y),
        patterns=[
            app(B, x, y),
            MultiPattern(app(B, x), COMP(x, y)),
        ],
        qid="beta_b",
    )
    yield ForAll(
        [x, y],
        app(CB, x, y) == COMP(y, x),
        patterns=[
            app(CB, x, y),
            MultiPattern(app(CB, x), COMP(y, x)),
        ],
        qid="beta_cb",
    )
    yield ForAll(
        [x, y],
        app(W, x, y) == app(x, y, y),
        patterns=[
            MultiPattern(app(W, x, y), app(x, y)),
            MultiPattern(app(W, x), app(x, y, y)),
        ],
        qid="beta_w",
    )
    yield ForAll(
        [x, y, z],
        app(C, x, y, z) == app(x, z, y),
        patterns=[
            MultiPattern(app(C, x, y, z), app(x, z)),
            MultiPattern(app(C, x, y), app(x, z, y)),
        ],
        qid="beta_c",
    )
    yield ForAll(
        [x, y, z],
        app(S, x, y, z) == app(x, z, app(y, z)),
        patterns=[
            MultiPattern(app(S, x, y, z), app(x, z), app(y, z)),
            MultiPattern(app(S, x, y), app(x, z, app(y, z))),
        ],
        qid="beta_s",
    )
    yield ForAll(
        [f],
        app(Y, f) == app(f, app(Y, f)),
        # patterns=[app(Y, f)],
        qid="beta_y",
    )

    # Fixed point equations
    yield ForAll([y], (app(S, I, y) == y) == (y == Y), qid="siy")

    # Beta reduction using Z3's SUBST
    # The general pattern is lazy, but the BOT,TOP,VAR versions are eager.
    yield ForAll(
        [x, y],
        APP(ABS(x), y) == SHIFT(SUBST(0, SHIFT(y, 0, 1), x), 0, -1),
        patterns=[
            MultiPattern(APP(ABS(x), y), SUBST(0, SHIFT(y, 0, 1), x)),
            MultiPattern(ABS(x), SHIFT(SUBST(0, SHIFT(y, 0, 1), x), 0, -1)),
        ],
        qid="beta_abs",
    )
    yield ForAll(
        [x],
        APP(ABS(x), BOT) == SHIFT(SUBST(0, BOT, x), 0, -1),
        patterns=[APP(ABS(x), BOT)],
        qid="beta_abs_bot",
    )
    yield ForAll(
        [x],
        APP(ABS(x), TOP) == SHIFT(SUBST(0, TOP, x), 0, -1),
        patterns=[APP(ABS(x), TOP)],
        qid="beta_abs_top",
    )
    yield ForAll(
        [x, i],
        APP(ABS(x), VAR(i)) == SHIFT(SUBST(0, VAR(i + 1), x), 0, -1),
        patterns=[APP(ABS(x), VAR(i))],
        qid="beta_abs_var",
    )

    # APP-JOIN distributivity (both directions)
    yield ForAll(
        [f, g, x],
        APP(JOIN(f, g), x) == JOIN(APP(f, x), APP(g, x)),
        patterns=[
            MultiPattern(APP(JOIN(f, g), x), APP(g, x), APP(f, x)),
            MultiPattern(JOIN(f, g), JOIN(APP(f, x), APP(g, x))),
        ],
        qid="app_join_dist",
    )
    yield ForAll(
        [f, x, y],
        LEQ(JOIN(APP(f, x), APP(f, y)), APP(f, JOIN(x, y))),
        patterns=[
            MultiPattern(JOIN(APP(f, x), APP(f, y)), JOIN(x, y)),
            MultiPattern(APP(f, x), APP(f, y), APP(f, JOIN(x, y))),
        ],
        qid="app_join_mono",
    )

    # APP monotonicity (in both arguments)
    yield ForAll(
        [f, g, x],
        Implies(LEQ(f, g), LEQ(APP(f, x), APP(g, x))),
        patterns=[
            MultiPattern(LEQ(f, g), APP(f, x), APP(g, x)),
            LEQ(APP(f, x), APP(g, x)),
        ],
        qid="app_mono_fun",
    )
    yield ForAll(
        [f, x, y],
        Implies(LEQ(x, y), LEQ(APP(f, x), APP(f, y))),
        patterns=[
            MultiPattern(LEQ(x, y), APP(f, x), APP(f, y)),
            LEQ(APP(f, x), APP(f, y)),
        ],
        qid="app_mono_arg",
    )

    # ABS monotonicity
    yield ForAll(
        [x, y],
        (ABS(x) == ABS(y)) == (x == y),
        patterns=[
            ABS(x) == ABS(y),
            MultiPattern(ABS(x), ABS(y), x == y),
        ],
        qid="abs_inj",
    )
    yield ForAll(
        [x, y],
        LEQ(x, y) == LEQ(ABS(x), ABS(y)),
        patterns=[MultiPattern(LEQ(x, y), ABS(x), ABS(y))],
        qid="abs_mono",
    )

    # BOT/TOP preservation
    yield ABS(BOT) == BOT
    yield ABS(TOP) == TOP

    # JOIN distributivity over ABS
    yield ForAll(
        [x, y],
        ABS(JOIN(x, y)) == JOIN(ABS(x), ABS(y)),
        patterns=[
            MultiPattern(ABS(JOIN(x, y)), ABS(x), ABS(y)),
            MultiPattern(JOIN(x, y), JOIN(ABS(x), ABS(y))),
        ],
        qid="abs_join_dist",
    )

    # Eta conversion
    if not "FIXME":  # this proximally causes unsatisfiability
        yield ForAll([f], ABS(APP(SHIFT(f, 0, 1), VAR(0))) == f, qid="eta_conv")


# FIXME this theory hangs.
def extensional_theory() -> Iterator[ExprRef]:
    """Extensionality axioms of lambda calculus."""
    # These nested quantifiers are hopeless.
    yield ForAll(
        [f, g],
        Implies(ForAll([x], APP(f, x) == APP(g, x)), f == g),
        qid="ext_app",
    )
    yield ForAll(
        [f, g],
        Implies(ForAll([x], LEQ(APP(f, x), APP(g, x))), LEQ(f, g)),
        qid="ext_leq",
    )


def hindley_axioms() -> Iterator[ExprRef]:
    """Yields beta reduction axioms for combinators."""
    yield ForAll([x], app(TOP, x) == TOP)
    yield ForAll([x], app(BOT, x) == BOT)
    yield ForAll([x], app(I, x) == x)
    yield ForAll([x, y], app(K, x, y) == x)
    yield ForAll([x, y], app(KI, x, y) == y)
    yield ForAll([x, y], app(J, x, y) == JOIN(x, y))
    yield ForAll([x, y], app(CI, x, y) == app(y, x))
    yield ForAll([x, y], app(B, x, y) == COMP(x, y))
    yield ForAll([x, y], app(CB, x, y) == COMP(y, x))
    yield ForAll([x, y], app(W, x, y) == app(x, y, y))
    yield ForAll([x, y, z], app(C, x, y, z) == app(x, z, y))
    yield ForAll([x, y, z], app(S, x, y, z) == app(x, z, app(y, z)))
    yield ForAll([x, y, z], app(COMP(x, y), z) == app(x, app(y, z)))
    yield ForAll([x, y, z], app(JOIN(x, y), z) == JOIN(app(x, z), app(y, z)))
    yield ForAll([f], app(Y, f) == app(f, app(Y, f)))
    yield ForAll([f], app(V, f) == JOIN(I, COMP(f, app(V, f))))
    yield ForAll([f], app(V, f) == JOIN(I, COMP(app(V, f), f)))


def hindley_theory() -> Iterator[ExprRef]:
    """
    Hindley-style quantifier free equations for extensionality.

    Returns the original list of universally quantified axioms.

    1. Roger Hindley (1967) "Axioms for strong reduction in combinatory logic"
    """
    equations: set[ExprRef] = set()  # deduplicate
    for axiom in hindley_axioms():
        equations.update(QEHindley(axiom))
    logger.info(f"Generated {len(equations)} Hindley equations")
    yield from equations


def simple_theory() -> Iterator[ExprRef]:
    """Theory of SIMPLE type, defined as join of section-retract pairs."""

    def above_all_sr(candidate: ExprRef) -> ExprRef:
        s1, r1 = z3.Consts("s1 r1", Term)  # Different names for bound variables
        return ForAll(
            [s1, r1],
            Implies(LEQ(COMP(r1, s1), I), LEQ(TUPLE(s1, r1), candidate)),
            qid="sr_above",
        )

    # SIMPLE is above all section-retract pairs.
    yield above_all_sr(SIMPLE)

    # SIMPLE is the least such term.
    yield ForAll([x], Implies(above_all_sr(x), LEQ(SIMPLE, x)), qid="simple_least")

    # Inhabitation. These nested quantifiers are hopeless.
    yield ForAll(
        [t, x],
        Implies(
            ForAll(
                [s, r],
                Implies(LEQ(COMP(r, s), I), LEQ(app(t, s, r, x), x)),
                qid="t_s_r_x",
            ),
            LEQ(app(SIMPLE, t, x), x),
        ),
        patterns=[LEQ(app(SIMPLE, t, x), x)],
        qid="simple_inhab",
    )


def closure_theory() -> Iterator[ExprRef]:
    """Theory of types and type membership."""
    # # Types are closures.
    yield ForAll([t], LEQ(I, APP(V, t)), qid="v_id")
    yield ForAll([t], COMP(APP(V, t), APP(V, t)) == APP(V, t), qid="v_comp")
    yield V == ABS(APP(Y, ABS(JOIN(I, COMP(v1, v0)))))
    yield V == ABS(APP(Y, ABS(JOIN(I, COMP(v0, v1)))))

    # TYPE is a type.
    yield LEQ(I, V)
    yield COMP(V, V) == V
    yield ForAll([t], APP(V, APP(V, t)) == APP(V, t), qid="v_idem")

    # Inhabitants are fixed points.
    yield OFTYPE(V, V)
    yield ForAll([t], OFTYPE(APP(V, t), V), qid="type_of_type")
    yield ForAll([t], APP(V, t) == JOIN(I, COMP(t, APP(V, t))), qid="v_join_left")
    yield ForAll([t], APP(V, t) == JOIN(I, COMP(APP(V, t), t)), qid="v_join_right")


def declare_type(t: ExprRef, inhabs: list[ExprRef], *, qid: str) -> Iterator[ExprRef]:
    # t is a type
    yield OFTYPE(t, V)
    # t contains all its inhabitants
    for x in inhabs:
        yield OFTYPE(x, t)
    # t contains only its inhabitants
    # FIXME how does this interact with variables?
    yield ForAll([x], Or(*[APP(t, x) == i for i in inhabs]), qid=f"inhab_{qid}")


def types_theory() -> Iterator[ExprRef]:
    """Theory of concrete types."""
    axioms = [
        *declare_type(DIV, [TOP, BOT], qid="div"),
        *declare_type(semi, [TOP, BOT, I], qid="semi"),
        *declare_type(unit, [TOP, I], qid="unit"),
        *declare_type(boool, [TOP, K, KI, J, BOT], qid="boool"),
        *declare_type(bool_, [TOP, K, KI, BOT], qid="bool"),
    ]
    logger.info(f"Generated {len(axioms)} type axioms")
    yield from axioms


def add_theory(solver: z3.Solver, *, include_all: bool = False) -> None:
    """Add all theories to the solver."""
    counter["add_theory"] += 1

    def add(theory: Callable[[], Iterable[ExprRef]]) -> None:
        axioms = list(theory())
        name = theory.__name__.replace("_theory", "")
        counter[name + "_axioms"] += len(axioms)
        counter["axioms"] += len(axioms)
        solver.add(*axioms)

    add(de_bruijn_theory)
    add(order_theory)
    add(lambda_theory)
    if include_all:
        add(extensional_theory)
        add(hindley_theory)
        add(simple_theory)
        add(closure_theory)
        add(types_theory)

    logger.info(f"Solver statistics:\n{solver.statistics()}")
