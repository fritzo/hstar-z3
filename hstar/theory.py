"""
# Theory of untyped λ-join-calculus modulo observable equality.

This uses combinators to represent closed λ-join-calculus terms, with a `LEQ`
relation for the Scott ordering, and explicit `BOT` (bottom), `TOP` (top), and
binary `JOIN` operation wrt the Scott ordering. Equality in this partial order
is observable equality, i.e. Hyland and Wadsworth's maximally coarse sensible
equivalence relation H*. H* is pi^0_2-complete and hence undecidable, however we
add Hindley's axioms for extensionality and several other axioms beyond mere
beta-eta equivalence.

In addition to untyped λ-join-calculus, we adopt Scott's framework of
types-as-closures, where a closure `a` is an idempotent increasing function
`I [= a = a o a`. Closures afford a rich type system including a universal type
`V` and a term `SIMPLE` that constructs simple types from codes.

All terms in this theory are definable from the generators APP, S, K, J, and
SIMPLE. Furthermore, it is conjectured that SIMPLE is definable from the
generators APP, S, K, and J.
"""

import functools
import logging
from collections.abc import Callable, Iterable, Iterator

import z3
from z3 import And, ExprRef, ForAll, Implies, MultiPattern, Not, Or

from .language import (
    APP,
    BOT,
    CB,
    CI,
    COMP,
    CONV,
    DIV,
    JOIN,
    KI,
    LEQ,
    LEQ_IS_Z3_PARTIAL_ORDER,
    OFTYPE,
    SIMPLE,
    TOP,
    TUPLE,
    VAR,
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
    lam,
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


def partial_order_v1(leq: Callable[[ExprRef, ExprRef], ExprRef]) -> Iterator[ExprRef]:
    """Generic theory of a partial order."""
    yield ForAll([x], leq(x, x), qid="leq_reflexive")
    yield ForAll(
        [x, y],
        And(leq(x, y), leq(y, x)) == (x == y),
        patterns=[MultiPattern(leq(x, y), leq(y, x))],
        qid="leq_antisym",
    )
    yield ForAll(
        [x, y, z],
        Implies(And(leq(x, y), leq(y, z)), leq(x, z)),
        patterns=[MultiPattern(leq(x, y), leq(y, z), leq(x, z))],
        qid="leq_trans",
    )


def partial_order_v2(leq: Callable[[ExprRef, ExprRef], ExprRef]) -> Iterator[ExprRef]:
    """Generic theory of a partial order."""
    # Adapted from
    # https://github.com/Z3Prover/z3/blob/master/src/tactic/core/special_relations_tactic.cpp
    # by adding patterns.
    yield ForAll(
        [x],
        leq(x, x),
        patterns=[leq(x, x)],
        qid="leq_reflexive",
    )
    yield ForAll(
        [x, y, z],
        Or(Not(leq(x, y)), Not(leq(y, z)), leq(x, z)),
        patterns=[
            MultiPattern(leq(x, y), leq(y, z)),
            MultiPattern(leq(x, y), leq(x, z)),
            MultiPattern(leq(y, z), leq(x, z)),
        ],
        qid="leq_trans_or",
    )
    yield ForAll(
        [x, y, z],
        Or(Not(And(leq(x, y), leq(y, z))), leq(x, z)),
        patterns=[
            MultiPattern(leq(x, y), leq(y, z)),
            MultiPattern(leq(x, y), leq(x, z)),
            MultiPattern(leq(y, z), leq(x, z)),
        ],
        qid="leq_trans_and",
    )
    yield ForAll(
        [x, y],
        Or(Not(leq(x, y)), Not(leq(y, x)), x == y),
        patterns=[MultiPattern(leq(x, y), leq(y, x))],
        qid="leq_antisym_or",
    )
    yield ForAll(
        [x, y],
        Or(Not(And(leq(x, y), leq(y, x))), x == y),
        patterns=[MultiPattern(leq(x, y), leq(y, x))],
        qid="leq_antisym_and",
    )


partial_order = partial_order_v2


# Theory of Scott ordering.
def order_theory() -> Iterator[ExprRef]:
    """Theory of Scott ordering and join."""
    # Basic order axioms
    yield Not(LEQ(TOP, BOT))
    yield ForAll([x], LEQ(x, TOP), qid="leq_top")
    yield ForAll([x], LEQ(BOT, x), qid="leq_bot")
    if not LEQ_IS_Z3_PARTIAL_ORDER:
        yield from partial_order(LEQ)

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


def combinator_theory() -> Iterator[ExprRef]:
    """
    Relations among of basic combinators.

    Note `==` denotes observable equivalence, not syntactic equality.
    """
    x = VAR(0)
    y = VAR(1)
    z = VAR(2)

    # Basic combinators
    yield I == lam(x, x)
    yield K == lam(x, lam(y, x))
    yield KI == lam(x, lam(y, y))
    yield J == JOIN(K, KI)
    yield B == lam(x, lam(y, lam(z, app(x, app(y, z)))))
    yield C == lam(x, lam(y, lam(z, app(x, z, y))))
    yield CI == lam(x, lam(y, app(y, x)))
    yield CB == lam(x, lam(y, lam(z, app(y, app(x, z)))))
    yield W == lam(x, lam(y, app(x, y, y)))
    yield S == lam(x, lam(y, lam(z, app(x, z, app(y, z)))))

    # Combinator equations
    yield KI == app(K, I)
    yield KI == app(C, K)
    yield CB == app(C, B)
    yield CI == app(C, I)
    yield J == app(C, J)
    yield I == app(W, J)

    # Fixed points
    lam_y_yy = lam(y, app(y, y))
    lam_y_x_yy = lam(y, app(x, app(y, y)))
    yield Y == lam(x, app(lam_y_yy, lam_y_x_yy))
    yield Y == lam(x, app(lam_y_x_yy, lam_y_x_yy))
    yield Y == app(S, I, Y)
    yield APP(Y, J) == TOP
    yield V == lam(x, app(Y, lam(y, JOIN(I, COMP(x, y)))))
    yield V == lam(x, app(Y, lam(y, JOIN(I, COMP(y, x)))))
    yield DIV == app(V, lam(x, app(x, TOP)))
    # TODO define SIMPLE


def closure_theory() -> Iterator[ExprRef]:
    """Theory of closures."""
    # Types are closures.
    yield ForAll(
        [t],
        LEQ(I, APP(V, t)),
        qid="v_id",
        patterns=[APP(V, t)],
    )
    yield ForAll(
        [t],
        COMP(APP(V, t), APP(V, t)) == APP(V, t),
        patterns=[APP(V, t)],
        qid="v_comp",
    )

    # TYPE is a type.
    yield LEQ(I, V)
    yield COMP(V, V) == V
    yield APP(V, V) == V


def lambda_theory() -> Iterator[ExprRef]:
    """Theory of lambda calculus and combinators."""
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
    yield ForAll(
        [x],
        app(V, x) == JOIN(I, COMP(x, app(V, x))),
        patterns=[COMP(x, app(Y, x))],
        qid="beta_v_left",
    )
    yield ForAll(
        [x],
        app(V, x) == JOIN(I, COMP(app(V, x), x)),
        patterns=[COMP(app(Y, x), x)],
        qid="beta_v_right",
    )
    yield ForAll(
        [x],
        app(DIV, x) == JOIN(x, app(DIV, x, TOP)),
        patterns=[app(DIV, x, TOP)],
        qid="beta_div",
    )

    # Fixed points
    yield ForAll([y], Implies(app(S, I, y) == y, y == Y), qid="siy")
    yield ForAll([x], APP(Y, APP(J, x)) == x, qid="yj")

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
    combinators = {"k": K, "ki": KI, "b": B, "c": C, "w": W, "s": S, "j": J}
    for name, c in sorted(combinators.items()):
        yield ForAll(
            [x, y],
            APP(c, JOIN(x, y)) == JOIN(APP(c, x), APP(c, y)),
            patterns=[
                MultiPattern(APP(c, JOIN(x, y)), APP(c, x), APP(c, y)),
                MultiPattern(JOIN(x, y), JOIN(APP(c, x), APP(c, y))),
            ],
            qid=f"{name}_join",
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


def extensional_theory() -> Iterator[ExprRef]:
    """Extensionality axioms of lambda calculus."""
    # These nested quantifiers are hopeless.
    yield ForAll(
        [f, g],
        Implies(
            ForAll([x], APP(f, x) == APP(g, x)),
            f == g,
        ),
        qid="ext_eq",
    )
    yield ForAll(
        [f, g],
        Implies(
            ForAll([x], LEQ(APP(f, x), APP(g, x))),
            LEQ(f, g),
        ),
        qid="ext_leq",
    )
    yield ForAll(
        [x],
        CONV(x) == Or(x == TOP, CONV(APP(x, TOP))),
        qid="conv_fix",
    )
    yield ForAll(
        [x, y],
        Implies(
            ForAll([f], CONV(APP(f, x)) == CONV(APP(f, y))),
            x == y,
        ),
        qid="hstar_eq",
    )
    yield ForAll(
        [x, y],
        Implies(
            ForAll([f], Implies(CONV(APP(f, y)), CONV(APP(f, x)))),
            LEQ(x, y),
        ),
        qid="hstar_leq",
    )


def simple_theory(*, include_all: bool = False) -> Iterator[ExprRef]:
    """
    Theory of a SIMPLE type constructor, defined as join of section-retract pairs.
    """

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
    if include_all:
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


def declare_type(t: ExprRef, inhabs: list[ExprRef], *, qid: str) -> Iterator[ExprRef]:
    # t is a type
    yield OFTYPE(t, V)
    # t contains all its inhabitants
    for x in inhabs:
        yield OFTYPE(x, t)
    # t contains only its inhabitants
    yield ForAll([x], Or(*[APP(t, x) == i for i in inhabs]), qid=f"inhab_{qid}")


def types_theory() -> Iterator[ExprRef]:
    """
    Theory of concrete types and their inhabitants.

    Note these types are embedded in the untyped lambda calculus.
    """
    axioms = [
        *declare_type(DIV, [TOP, BOT], qid="div"),
        *declare_type(semi, [TOP, BOT, I], qid="semi"),
        *declare_type(unit, [TOP, I], qid="unit"),
        *declare_type(boool, [TOP, K, KI, J, BOT], qid="boool"),
        *declare_type(bool_, [TOP, K, KI, BOT], qid="bool"),
    ]
    logger.info(f"Generated {len(axioms)} type axioms")
    yield from axioms


@functools.cache
def get_theory(*, include_all: bool = False) -> tuple[tuple[str, ExprRef], ...]:
    """Get the entire default theory."""
    counter["add_theory"] += 1
    result: list[tuple[str, ExprRef]] = []
    seen: set[ExprRef] = set()
    ax_count = 0
    eq_count = 0

    def order(eq: ExprRef) -> tuple[int, str]:
        s = str(eq)
        return len(s), s

    def add(theory: Callable[[], Iterable[ExprRef]]) -> None:
        nonlocal ax_count, eq_count
        for axiom in theory():
            if axiom in seen:
                continue
            ax_count += 1
            prefix = ""
            if z3.is_quantifier(axiom) and axiom.is_forall():
                if qid := axiom.qid():
                    prefix = qid + ": "
            result.append((prefix + str(axiom), axiom))
            seen.add(axiom)

            # Add Hindley equations.
            equations = sorted(QEHindley(axiom) - seen, key=order)
            ax_count += len(equations)
            eq_count += len(equations)
            seen.update(equations)
            for eq in equations:
                result.append((prefix + str(eq), eq))

    add(order_theory)
    add(combinator_theory)
    add(closure_theory)
    add(lambda_theory)
    add(simple_theory)
    if include_all:
        add(extensional_theory)
        add(types_theory)

    counter["axioms"] = max(counter["axioms"], ax_count)
    counter["equations"] = max(counter["equations"], eq_count)
    return tuple(result)


def add_theory(
    solver: z3.Solver,
    *,
    unsat_core: bool = True,
    include_all: bool = False,
) -> None:
    """Add all theories to the solver."""
    counter["add_theory"] += 1
    logger.info("Initializing theory")
    for name, axiom in get_theory(include_all=include_all):
        if unsat_core:
            # Add named axiom to the solver.
            # This uses assert_and_track to support unsat_core.
            # Note assert_and_track is absent from solver.assertions(), so profiling
            # requires unsat_core=False.
            solver.assert_and_track(axiom, name)
        else:
            solver.add(axiom)
    logger.info(f"Solver statistics:\n{solver.statistics()}")
