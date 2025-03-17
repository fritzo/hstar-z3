"""
# Theory of λ-join-calculus expressions with Z3.

This uses a de Bruijn indexed representation of λ-join-calculus terms, with a
`LEQ` relation for the Scott ordering, and explicit `BOT` (bottom), `TOP` (top),
and binary `JOIN` operation wrt the Scott ordering.

The theory includes de Bruijn syntax, Scott ordering, lambda calculus, and
a types-as-closures.
"""

import logging
from collections.abc import Iterator

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


def de_bruijn_theory(solver: z3.Solver) -> None:
    """Theory of de Bruijn operations SHIFT and SUBST."""
    logger.info("Adding de Bruijn theory")
    i = z3.Int("i")
    j = z3.Int("j")
    body = z3.Const("body", Term)
    lhs = z3.Const("lhs", Term)
    rhs = z3.Const("rhs", Term)
    start = z3.Int("start")
    solver.add(
        # SHIFT axioms, deterministically executed forwards.
        ForAll(
            [i, start],
            SHIFT(VAR(i), start) == VAR(If(i >= start, i + 1, i)),
            patterns=[SHIFT(VAR(i), start)],
            qid="shift_var",
        ),
        ForAll(
            [x, start],
            SHIFT(ABS(x), start) == ABS(SHIFT(x, start + 1)),
            patterns=[SHIFT(ABS(x), start)],
            qid="shift_abs",
        ),
        ForAll(
            [lhs, rhs, start],
            SHIFT(APP(lhs, rhs), start) == APP(SHIFT(lhs, start), SHIFT(rhs, start)),
            patterns=[SHIFT(APP(lhs, rhs), start)],
            qid="shift_app",
        ),
        ForAll(
            [lhs, rhs, start],
            SHIFT(JOIN(lhs, rhs), start) == JOIN(SHIFT(lhs, start), SHIFT(rhs, start)),
            patterns=[SHIFT(JOIN(lhs, rhs), start)],
            qid="shift_join",
        ),
        ForAll(
            [lhs, rhs, start],
            SHIFT(COMP(lhs, rhs), start) == COMP(SHIFT(lhs, start), SHIFT(rhs, start)),
            patterns=[SHIFT(COMP(lhs, rhs), start)],
            qid="shift_comp",
        ),
        ForAll([start], SHIFT(TOP, start) == TOP, qid="shift_top"),
        ForAll([start], SHIFT(BOT, start) == BOT, qid="shift_bot"),
        # SUBST axioms, deterministically executed forwards.
        ForAll(
            [j, i, x],
            SUBST(i, x, VAR(j)) == If(j == i, x, VAR(j)),
            patterns=[SUBST(i, x, VAR(j))],
            qid="subst_var",
        ),
        ForAll(
            [body, i, x],
            SUBST(i, x, ABS(body)) == ABS(SUBST(i + 1, SHIFT(x, 0), body)),
            patterns=[SUBST(i, x, ABS(body))],
            qid="subst_abs",
        ),
        ForAll(
            [lhs, rhs, i, x],
            SUBST(i, x, APP(lhs, rhs)) == APP(SUBST(i, x, lhs), SUBST(i, x, rhs)),
            patterns=[SUBST(i, x, APP(lhs, rhs))],
            qid="subst_app",
        ),
        ForAll(
            [lhs, rhs, i, x],
            SUBST(i, x, JOIN(lhs, rhs)) == JOIN(SUBST(i, x, lhs), SUBST(i, x, rhs)),
            patterns=[SUBST(i, x, JOIN(lhs, rhs))],
            qid="subst_join",
        ),
        ForAll(
            [lhs, rhs, i, x],
            SUBST(i, x, COMP(lhs, rhs)) == COMP(SUBST(i, x, lhs), SUBST(i, x, rhs)),
            patterns=[SUBST(i, x, COMP(lhs, rhs))],
            qid="subst_comp",
        ),
        ForAll([i, x], SUBST(i, x, TOP) == TOP, qid="subst_top"),
        ForAll([i, x], SUBST(i, x, BOT) == BOT, qid="subst_bot"),
    )


# Theory of Scott ordering.
def order_theory(solver: z3.Solver) -> None:
    logger.info("Adding Scott order theory")
    solver.add(
        # Basic order axioms
        ForAll([x], LEQ(x, TOP), qid="leq_top"),
        ForAll([x], LEQ(BOT, x), qid="leq_bot"),
        ForAll([x], LEQ(x, x), qid="leq_reflexive"),
        ForAll(
            [x, y],
            And(LEQ(x, y), LEQ(y, x)) == (x == y),
            patterns=[MultiPattern(LEQ(x, y), LEQ(y, x))],
            qid="leq_antisym",
        ),
        ForAll(
            [x, y, z],
            Implies(And(LEQ(x, y), LEQ(y, z)), LEQ(x, z)),
            patterns=[MultiPattern(LEQ(x, y), LEQ(y, z), LEQ(x, z))],
            qid="leq_trans",
        ),
        Not(LEQ(TOP, BOT)),
        # JOIN is least upper bound
        ForAll([x, y], LEQ(x, JOIN(x, y)), qid="leq_join_left"),
        ForAll([x, y], LEQ(y, JOIN(x, y)), qid="leq_join_right"),
        ForAll(
            [x, y, z],
            And(LEQ(x, z), LEQ(y, z)) == LEQ(JOIN(x, y), z),
            qid="join_lub",
        ),  # Least upper bound property
        # JOIN is associative, commutative, and idempotent
        ForAll([x, y], JOIN(x, y) == JOIN(y, x), qid="join_commute"),
        ForAll(
            [x, y, z],
            JOIN(x, JOIN(y, z)) == JOIN(JOIN(x, y), z),
            patterns=[
                MultiPattern(JOIN(x, JOIN(y, z)), JOIN(x, y)),
                MultiPattern(JOIN(y, z), JOIN(JOIN(x, y), z)),
            ],
            qid="join_assoc",
        ),
        ForAll([x], JOIN(x, x) == x, qid="join_idem"),
        # Distributivity
        ForAll(
            [x, y, z],
            JOIN(x, JOIN(y, z)) == JOIN(JOIN(x, y), JOIN(x, z)),
            patterns=[
                MultiPattern(JOIN(x, JOIN(y, z)), JOIN(x, y), JOIN(x, z)),
                MultiPattern(JOIN(y, z), JOIN(JOIN(x, y), JOIN(x, z))),
            ],
            qid="join_dist",
        ),
        # JOIN with BOT/TOP
        ForAll([x], JOIN(x, BOT) == x, qid="join_bot"),  # BOT is identity
        ForAll([x], JOIN(x, TOP) == TOP, qid="join_top"),  # TOP absorbs
    )


# Theory of lambda calculus.
def lambda_theory(solver: z3.Solver) -> None:
    logger.info("Adding lambda theory")
    i = z3.Int("i")
    solver.add(
        # Composition properties
        ForAll(
            [f, g, x],
            APP(COMP(f, g), x) == APP(f, APP(g, x)),
            patterns=[
                MultiPattern(APP(COMP(f, g), x), APP(g, x)),
                MultiPattern(COMP(f, g), APP(f, APP(g, x))),
            ],
            qid="comp_beta",
        ),
        ForAll([f], COMP(f, I) == f, qid="comp_id_right"),
        ForAll([f], COMP(I, f) == f, qid="comp_id_left"),
        ForAll([f], COMP(BOT, f) == BOT, qid="comp_bot"),
        ForAll([f], COMP(TOP, f) == TOP, qid="comp_top"),
        # Composition is associative
        ForAll(
            [f, g, h],
            COMP(f, COMP(g, h)) == COMP(COMP(f, g), h),
            patterns=[
                MultiPattern(COMP(f, COMP(g, h)), COMP(f, g)),
                MultiPattern(COMP(COMP(f, g), h), COMP(g, h)),
            ],
            qid="comp_assoc",
        ),
        # Composition is monotonic in both arguments
        ForAll(
            [f, g, h],
            Implies(LEQ(f, g), LEQ(COMP(f, h), COMP(g, h))),
            patterns=[
                MultiPattern(LEQ(f, g), COMP(f, h), COMP(g, h)),
                LEQ(COMP(f, h), COMP(g, h)),
            ],
            qid="comp_mono_left",
        ),
        ForAll(
            [f, g, h],
            Implies(LEQ(g, h), LEQ(COMP(f, g), COMP(f, h))),
            patterns=[
                MultiPattern(LEQ(g, h), COMP(f, g), COMP(f, h)),
                LEQ(COMP(f, g), COMP(f, h)),
            ],
            qid="comp_mono_right",
        ),
        # Combinator equations
        KI == app(K, I),
        CB == app(C, B),
        CI == app(C, I),
        J == app(C, J),
        I == app(W, J),
        Y == Y_,
        V == V_,
        # Beta reduction of combinators
        ForAll([x], app(I, x) == x, qid="beta_i"),
        ForAll(
            [x, y],
            app(K, x, y) == x,
            qid="beta_k",
        ),
        ForAll(
            [x, y],
            app(KI, x, y) == y,
            qid="beta_ki",
        ),
        ForAll(
            [x, y],
            app(J, x, y) == JOIN(x, y),
            patterns=[
                app(J, x, y),
                MultiPattern(app(J, x), JOIN(x, y)),
            ],
            qid="beta_j",
        ),
        ForAll(
            [x, y],
            app(B, x, y) == COMP(x, y),
            patterns=[
                app(B, x, y),
                MultiPattern(app(B, x), COMP(x, y)),
            ],
            qid="beta_b",
        ),
        ForAll(
            [x, y],
            app(CB, x, y) == COMP(y, x),
            patterns=[
                app(CB, x, y),
                MultiPattern(app(CB, x), COMP(y, x)),
            ],
            qid="beta_cb",
        ),
        ForAll(
            [x, y, z],
            app(C, x, y, z) == app(x, z, y),
            patterns=[
                MultiPattern(app(C, x, y, z), app(x, z)),
                MultiPattern(app(C, x, y), app(x, z, y)),
            ],
            qid="beta_c",
        ),
        ForAll(
            [x, y],
            app(CI, x, y) == app(y, x),
            patterns=[
                MultiPattern(app(CI, x, y)),
                MultiPattern(app(CI, x), app(y, x)),
            ],
            qid="beta_ci",
        ),
        ForAll(
            [x, y],
            app(W, x, y) == app(x, y, y),
            patterns=[
                MultiPattern(app(W, x, y), app(x, y)),
                MultiPattern(app(W, x), app(x, y, y)),
            ],
            qid="beta_w",
        ),
        ForAll(
            [x, y, z],
            app(S, x, y, z) == app(x, z, app(y, z)),
            patterns=[
                MultiPattern(app(S, x, y, z), app(x, z), app(y, z)),
                MultiPattern(app(S, x, y), app(x, z, app(y, z))),
            ],
            qid="beta_s",
        ),
        ForAll(
            [f],
            app(Y, f) == app(f, app(Y, f)),
            # patterns=[app(Y, f)],
            qid="beta_y",
        ),
        # Fixed point equations
        ForAll([y], (app(S, I, y) == y) == (y == Y), qid="siy"),
        # Beta reduction using Z3's SUBST
        # The general pattern is lazy, but the BOT,TOP,VAR versions are eager.
        ForAll(
            [x, y],
            APP(ABS(x), y) == SUBST(0, y, x),
            patterns=[MultiPattern(APP(ABS(x), y), SUBST(0, y, x))],
            qid="beta_abs",
        ),
        ForAll(
            [x],
            APP(ABS(x), BOT) == SUBST(0, BOT, x),
            patterns=[APP(ABS(x), BOT)],
            qid="beta_abs_bot",
        ),
        ForAll(
            [x],
            APP(ABS(x), TOP) == SUBST(0, TOP, x),
            patterns=[APP(ABS(x), TOP)],
            qid="beta_abs_top",
        ),
        ForAll(
            [x, i],
            APP(ABS(x), VAR(i)) == SUBST(0, VAR(i), x),
            patterns=[APP(ABS(x), VAR(i))],
            qid="beta_abs_var",
        ),
        # APP-JOIN distributivity (both directions)
        ForAll(
            [f, g, x],
            APP(JOIN(f, g), x) == JOIN(APP(f, x), APP(g, x)),
            patterns=[
                MultiPattern(APP(JOIN(f, g), x), APP(g, x), APP(f, x)),
                MultiPattern(JOIN(f, g), JOIN(APP(f, x), APP(g, x))),
            ],
            qid="app_join_dist",
        ),
        ForAll(
            [f, x, y],
            LEQ(JOIN(APP(f, x), APP(f, y)), APP(f, JOIN(x, y))),
            patterns=[
                MultiPattern(JOIN(APP(f, x), APP(f, y)), JOIN(x, y)),
                MultiPattern(APP(f, x), APP(f, y), APP(f, JOIN(x, y))),
            ],
            qid="app_join_mono",
        ),
        # APP monotonicity (in both arguments)
        ForAll(
            [f, g, x],
            Implies(LEQ(f, g), LEQ(APP(f, x), APP(g, x))),
            patterns=[
                MultiPattern(LEQ(f, g), APP(f, x), APP(g, x)),
                LEQ(APP(f, x), APP(g, x)),
            ],
            qid="app_mono_fun",
        ),
        ForAll(
            [f, x, y],
            Implies(LEQ(x, y), LEQ(APP(f, x), APP(f, y))),
            patterns=[
                MultiPattern(LEQ(x, y), APP(f, x), APP(f, y)),
                LEQ(APP(f, x), APP(f, y)),
            ],
            qid="app_mono_arg",
        ),
        # ABS monotonicity
        ForAll(
            [x, y],
            (ABS(x) == ABS(y)) == (x == y),
            patterns=[
                ABS(x) == ABS(y),
                MultiPattern(ABS(x), ABS(y), x == y),
            ],
            qid="abs_inj",
        ),
        ForAll(
            [x, y],
            LEQ(x, y) == LEQ(ABS(x), ABS(y)),
            patterns=[MultiPattern(LEQ(x, y), ABS(x), ABS(y))],
            qid="abs_mono",
        ),
        # BOT/TOP preservation
        ForAll([x], APP(BOT, x) == BOT, qid="app_bot"),
        ForAll([x], APP(TOP, x) == TOP, qid="app_top"),
        ABS(BOT) == BOT,
        ABS(TOP) == TOP,
        # JOIN distributivity over ABS
        ForAll(
            [x, y],
            ABS(JOIN(x, y)) == JOIN(ABS(x), ABS(y)),
            patterns=[
                MultiPattern(ABS(JOIN(x, y)), ABS(x), ABS(y)),
                MultiPattern(JOIN(x, y), JOIN(ABS(x), ABS(y))),
            ],
            qid="abs_join_dist",
        ),
        # Eta conversion
        ForAll([f], ABS(APP(SHIFT(f, 0), VAR(0))) == f, qid="eta_conv"),
    )


# FIXME this theory hangs.
def extensional_theory(solver: z3.Solver) -> None:
    """Extensionality axioms of lambda calculus."""
    logger.info("Adding extensional theory")
    solver.add(
        # Extensionality
        ForAll(
            [f, g],
            Implies(ForAll([x], APP(f, x) == APP(g, x)), f == g),
            qid="ext_app",
        ),
        ForAll(
            [f, g],
            Implies(ForAll([x], LEQ(APP(f, x), APP(g, x))), LEQ(f, g)),
            qid="ext_leq",
        ),
    )


def hindley_theory(solver: z3.Solver | None = None) -> list[ExprRef]:
    """
    Hindley-style quantifier free equations for extensionality.

    Returns the original list of universally quantified axioms.

    1. Roger Hindley (1967) "Axioms for strong reduction in combinatory logic"
    """
    axioms = [
        ForAll([x], app(TOP, x) == TOP),
        ForAll([x], app(BOT, x) == BOT),
        ForAll([x], app(I, x) == x),
        ForAll([x, y], app(K, x, y) == x),
        ForAll([x, y], app(KI, x, y) == y),
        ForAll([x, y], app(J, x, y) == JOIN(x, y)),
        ForAll([x, y], app(CI, x, y) == app(y, x)),
        ForAll([x, y], app(W, x, y) == app(x, y, y)),
        ForAll([x, y, z], app(B, x, y, z) == app(x, app(y, z))),
        ForAll([x, y, z], app(C, x, y, z) == app(x, z, y)),
        ForAll([x, y, z], app(CB, x, y, z) == app(y, app(x, z))),
        ForAll([x, y, z], app(S, x, y, z) == app(x, z, app(y, z))),
        ForAll([f], app(Y, f) == app(f, app(Y, f))),
    ]
    if solver is not None:
        logger.info("Adding Hindley theory")
        equations: set[ExprRef] = set()
        for axiom in axioms:
            equations.update(QEHindley(axiom))
        logger.info(f"Generated {len(axioms)} Hindley equations")
        solver.add(*equations)
    return axioms


def simple_theory(solver: z3.Solver) -> None:
    """Theory of SIMPLE type, defined as join of section-retract pairs."""
    logger.info("Adding simple type theory")

    def above_all_sr(candidate: ExprRef) -> ExprRef:
        s1, r1 = z3.Consts("s1 r1", Term)  # Different names for bound variables
        return ForAll(
            [s1, r1],
            Implies(LEQ(COMP(r1, s1), I), LEQ(TUPLE(s1, r1), candidate)),
            qid="sr_above",
        )

    solver.add(
        # SIMPLE is above all section-retract pairs.
        above_all_sr(SIMPLE),
        # SIMPLE is the least such term.
        ForAll([x], Implies(above_all_sr(x), LEQ(SIMPLE, x)), qid="simple_least"),
        # inhabitation
        ForAll(
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
        ),
    )


def closure_theory(solver: z3.Solver) -> None:
    """Theory of types and type membership."""
    logger.info("Adding closure theory")
    solver.add(
        # # Types are closures.
        ForAll([t], LEQ(I, APP(V, t)), qid="v_id"),
        ForAll([t], COMP(APP(V, t), APP(V, t)) == APP(V, t), qid="v_comp"),
        V == ABS(APP(Y, ABS(JOIN(I, COMP(v1, v0))))),
        V == ABS(APP(Y, ABS(JOIN(I, COMP(v0, v1))))),
        # TYPE is a type.
        LEQ(I, V),
        COMP(V, V) == V,
        ForAll([t], APP(V, APP(V, t)) == APP(V, t), qid="v_idem"),
        # Inhabitants are fixed points.
        OFTYPE(V, V),
        ForAll([t], OFTYPE(APP(V, t), V), qid="type_of_type"),
        ForAll([t], APP(V, t) == JOIN(I, COMP(t, APP(V, t))), qid="v_join_left"),
        ForAll([t], APP(V, t) == JOIN(I, COMP(APP(V, t), t)), qid="v_join_right"),
    )


def declare_type(t: ExprRef, inhabs: list[ExprRef], *, qid: str) -> Iterator[ExprRef]:
    # t is a type
    yield OFTYPE(t, V)
    # t contains all its inhabitants
    for x in inhabs:
        yield OFTYPE(x, t)
    # t contains only its inhabitants
    # FIXME how does this interact with variables?
    yield ForAll([x], Or(*[APP(t, x) == i for i in inhabs]), qid=f"inhab_{qid}")


def types_theory(solver: z3.Solver | None = None) -> list[ExprRef]:
    """Theory of concrete types."""
    axioms = [
        *declare_type(DIV, [TOP, BOT], qid="div"),
        *declare_type(semi, [TOP, BOT, I], qid="semi"),
        *declare_type(unit, [TOP, I], qid="unit"),
        *declare_type(boool, [TOP, K, KI, J, BOT], qid="boool"),
        *declare_type(bool_, [TOP, K, KI, BOT], qid="bool"),
    ]
    logger.info(f"Generated {len(axioms)} type axioms")
    if solver is not None:
        logger.info("Adding types theory")
        solver.add(*axioms)
    return axioms


def add_theory(solver: z3.Solver, *, include_slow: bool = False) -> None:
    """Add all theories to the solver."""
    counter["add_theory"] += 1
    de_bruijn_theory(solver)
    order_theory(solver)
    lambda_theory(solver)
    if include_slow:
        extensional_theory(solver)
        hindley_theory(solver)
        simple_theory(solver)
        closure_theory(solver)
    if include_slow:
        types_theory(solver)
    logger.info(f"Solver statistics:\n{solver.statistics()}")
