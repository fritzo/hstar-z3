import logging

import pytest
from immutables import Map

from hstar.enumeration import enumerator
from hstar.normal import (
    ABS,
    APP,
    BOT,
    JOIN,
    TOP,
    VAR,
    Env,
    TermType,
    _Term,
    approximate,
    beta_ball,
    complexity,
    env_compose,
    is_normal,
    leq,
    shift,
    subst,
)

logger = logging.getLogger(__name__)


def test_hash_consing() -> None:
    """Test that identical terms are hash-consed to the same object."""
    # Test basic terms
    assert VAR(0) is VAR(0)
    assert VAR(1) is VAR(1)

    # Test compound terms
    assert JOIN(VAR(0), VAR(1)) is JOIN(VAR(0), VAR(1))
    assert APP(VAR(0), JOIN(VAR(1))) is APP(VAR(0), JOIN(VAR(1)))

    # Test associativity, commutativity, and idempotence of JOIN
    assert JOIN(VAR(0), JOIN(VAR(1), VAR(2))) is JOIN(JOIN(VAR(0), VAR(1)), VAR(2))
    assert JOIN(VAR(0), VAR(1)) is JOIN(VAR(1), VAR(0))
    assert JOIN(VAR(0), VAR(0)) is VAR(0), VAR(1)
    assert JOIN(TOP) is TOP
    assert JOIN(BOT) is BOT

    # Test that different terms are different objects
    assert VAR(0) is not VAR(1)

    # Test ABS hash consing
    lambda_term1 = ABS(JOIN(VAR(0)))
    lambda_term2 = ABS(JOIN(VAR(0)))
    assert lambda_term1 is lambda_term2

    # Test that TOP absorbs other terms in a join
    assert JOIN(TOP, VAR(0)) is TOP

    # Test environments
    assert Env() is Env()
    assert Env({0: VAR(0)}) is Env({0: VAR(0)})


def test_shift() -> None:
    """Test that the shift operation works correctly."""
    # Test variable shifting
    assert shift(VAR(0)) is VAR(1)
    assert shift(VAR(1)) is VAR(2)

    # Test shifting with a custom start point
    assert shift(VAR(0), start=1) is VAR(0)  # No shift, as index < starting_at
    assert shift(VAR(1), start=1) is VAR(2)  # Shift by 1, as index >= starting_at
    assert shift(VAR(2), start=2) is VAR(3)  # Only shift if index >= starting_at

    # Test that TOP and BOT are invariant under shift
    assert shift(TOP) is TOP
    assert shift(BOT) is BOT

    # Test shifting of lambda terms
    lam_var0 = ABS(JOIN(VAR(0)))
    assert shift(lam_var0) is lam_var0  # Bound variable, shouldn't change

    # Test that free variables in abstractions are correctly shifted
    lam_with_free = ABS(JOIN(VAR(1)))  # \x.y
    assert shift(lam_with_free) is ABS(JOIN(VAR(2)))  # \x.z

    # Test application shift
    app_term = APP(VAR(0), JOIN(VAR(1)))
    shifted_app = shift(app_term)
    assert shifted_app is APP(VAR(1), JOIN(VAR(2)))

    # Test nested terms
    nested = APP(VAR(0), JOIN(APP(VAR(1), JOIN(VAR(2)))))
    shifted_nested = shift(nested)
    assert shifted_nested is APP(VAR(1), JOIN(APP(VAR(2), JOIN(VAR(3)))))


def test_subst() -> None:
    """Test that the substitution operation works correctly."""
    # Basic variable substitution
    assert subst(VAR(0), Env({0: TOP})) is TOP  # [TOP/0]0 = TOP
    assert subst(VAR(1), Env({0: TOP})) is VAR(1)  # [TOP/0]1 = 1
    assert subst(VAR(0), Env({1: TOP})) is VAR(0)  # [TOP/1]0 = 0
    assert subst(VAR(1), Env({1: TOP})) is TOP  # [TOP/1]1 = TOP

    # Multiple simultaneous substitutions
    assert subst(VAR(0), Env({0: VAR(1), 1: TOP})) is VAR(1)  # [1/0, TOP/1]0 = 1
    assert subst(VAR(1), Env({0: VAR(2), 1: TOP})) is TOP  # [2/0, TOP/1]1 = TOP
    actual = subst(JOIN(VAR(0), VAR(1)), Env({0: VAR(2), 1: VAR(3)}))
    assert actual is JOIN(VAR(2), VAR(3))

    # Identity function substitution
    id_term = ABS(JOIN(VAR(0)))  # \x.x
    assert subst(id_term, Env({0: TOP})) is id_term  # [TOP/0](\x.x) = \x.x
    assert subst(id_term, Env({1: TOP})) is id_term  # [TOP/1](\x.x) = \x.x

    # Application substitution
    app_term = APP(VAR(0), JOIN(VAR(1)))  # 0 1
    assert subst(app_term, Env({0: TOP})) is APP(
        TOP, JOIN(VAR(1))
    )  # [TOP/0](0 1) = TOP 1
    assert subst(app_term, Env({1: TOP})) is APP(
        VAR(0), JOIN(TOP)
    )  # [TOP/1](0 1) = 0 TOP

    # Multiple substitutions in application
    assert subst(app_term, Env({0: VAR(2), 1: TOP})) is APP(
        VAR(2), JOIN(TOP)
    )  # [2/0, TOP/1](0 1) = 2 TOP

    # Nested abstraction substitution
    # \x.1 x
    nested = ABS(JOIN(APP(VAR(1), JOIN(VAR(0)))))
    # [TOP/1](\x.1 x)
    assert subst(nested, Env({0: TOP})) is TOP

    # More complex substitution cases
    complex_term = ABS(APP(VAR(0), APP(VAR(1), VAR(2))))  # \x. x (1 2)
    # When substituting for var 1
    actual = subst(complex_term, Env({0: TOP}))
    # Expected: \x. x (TOP 2)
    expected = ABS(APP(VAR(0), TOP))
    assert actual is expected

    # Join substitution
    join_term = JOIN(VAR(0), VAR(1))  # 0 | 1
    assert subst(join_term, Env({0: TOP})) is JOIN(TOP, VAR(1))
    assert subst(join_term, Env({1: TOP})) is JOIN(VAR(0), TOP)

    # Multiple substitutions in join
    assert subst(join_term, Env({0: TOP, 1: VAR(2)})) is JOIN(TOP, VAR(2))

    # Test BOT substitution
    assert subst(VAR(0), Env({0: BOT})) is BOT
    assert subst(APP(VAR(0), JOIN(VAR(1))), Env({0: BOT})) is APP(BOT, JOIN(VAR(1)))

    # Test empty environment
    assert subst(VAR(0), Env()) is VAR(0)
    assert subst(APP(VAR(0), JOIN(VAR(1))), Env()) is APP(VAR(0), JOIN(VAR(1)))


def test_abs() -> None:
    """Test that the ABS operation works correctly with hash consing."""
    # Test basic lambda abstraction
    lam0 = ABS(JOIN(VAR(0)))  # \x.x
    free_vars = Map({0: 1})
    assert lam0.parts == frozenset(
        [_Term(TermType.ABS, head=_Term(TermType.VAR, varname=0, free_vars=free_vars))]
    )

    # zero occurrences
    lam_constant = ABS(JOIN(VAR(1)))  # \x.y
    assert list(lam_constant.parts)[0].typ == TermType.ABS

    # one occurrence
    lam_linear = ABS(JOIN(VAR(0)))  # \x.x
    assert list(lam_linear.parts)[0].typ == TermType.ABS

    # two or more occurrences
    lam_nonlinear = ABS(JOIN(APP(VAR(0), JOIN(VAR(0)))))  # \x.x x
    assert list(lam_nonlinear.parts)[0].typ == TermType.ABS

    # Test that ABS is idempotent with hash consing
    assert ABS(JOIN(VAR(0))) is ABS(JOIN(VAR(0)))

    # Test that ABS(TOP) is TOP
    assert ABS(TOP) is TOP


def test_eager_linear_reduction() -> None:
    # JOIN reduction
    assert JOIN(TOP) is TOP
    assert JOIN(BOT) is BOT
    assert JOIN(VAR(0)) is VAR(0)
    assert JOIN(VAR(0), VAR(0)) is VAR(0)
    assert JOIN(VAR(0), VAR(1)) is JOIN(VAR(0), VAR(1))
    assert JOIN(JOIN(VAR(0), VAR(1)), VAR(2)) is JOIN(VAR(0), VAR(1), VAR(2))
    assert JOIN(VAR(0), JOIN(VAR(1), VAR(2))) is JOIN(VAR(0), VAR(1), VAR(2))
    assert JOIN(BOT, VAR(0)) is VAR(0)
    assert JOIN(TOP, VAR(0)) is TOP

    # Linear beta reduction
    assert ABS(TOP) is TOP
    assert ABS(BOT) is BOT
    assert APP(ABS(VAR(1)), VAR(1)) is VAR(0)
    assert APP(ABS(VAR(0)), VAR(1)) is VAR(1)
    assert APP(ABS(APP(VAR(1), VAR(0))), BOT) is APP(VAR(0), BOT)
    assert APP(ABS(APP(VAR(1), VAR(0))), TOP) is APP(VAR(0), TOP)
    assert APP(ABS(APP(VAR(1), VAR(0))), VAR(0)) is APP(VAR(0), VAR(0))
    assert APP(ABS(APP(VAR(1), VAR(0))), VAR(1)) is APP(VAR(0), VAR(1))
    assert APP(ABS(APP(VAR(1), VAR(0))), VAR(2)) is APP(VAR(0), VAR(2))

    # Eta conversion
    assert ABS(APP(VAR(1), VAR(0))) is VAR(0)


def test_complexity() -> None:
    """Test that complexity calculation is correct for various terms."""
    # Test complexity of basic terms
    assert complexity(TOP) == 1
    assert complexity(BOT) == 1
    assert complexity(VAR(0)) == 1
    assert complexity(VAR(1)) == 2
    assert complexity(VAR(2)) == 3

    # Test complexity of single application
    app_term = APP(VAR(0), VAR(1))
    assert complexity(app_term) == 4  # 1 (APP) + 1 (VAR) + 2 (VAR)

    # Test complexity of lambda terms
    id_term = ABS(VAR(0))  # λx.x
    assert complexity(id_term) == 2  # 1 (ABS) + 1 (VAR)

    const_term = ABS(VAR(1))  # λx.y
    assert complexity(const_term) == 3  # 2 (ABS) + 1 (VAR)

    # Test complexity of nested applications
    nested_app = APP(VAR(0), APP(VAR(1), VAR(2)))
    assert complexity(nested_app) == 8
    # 1 (APP) + 1 (VAR) + 1 (APP) + 2 (VAR) + 3 (VAR)

    # Test complexity of join terms
    join_term = JOIN(VAR(0), VAR(1))
    assert complexity(join_term) == 4  # 1 (VAR) + 2 (VAR) + (2 - 1)


def test_env_compose() -> None:
    """Test that environment composition works correctly."""
    # Test basic environment composition
    env1 = Env({0: VAR(1)})
    env2 = Env({1: VAR(2)})
    composed = env_compose(env1, env2)
    assert composed == Env({0: VAR(2), 1: VAR(2)})  # VAR(1) substituted to VAR(2)

    # Test composition with empty environments
    assert env_compose(Env(), env2) == env2
    assert env_compose(env1, Env()) == env1

    # Test composition with overlapping variables
    env3 = Env({0: VAR(2), 2: VAR(3)})
    env4 = Env({0: VAR(4), 2: VAR(5)})
    composed = env_compose(env3, env4)
    # VAR(2) is substituted to VAR(5) due to the mapping 2->VAR(5) in env4
    # VAR(3) remains unchanged as there's no mapping for 3 in env4
    assert composed == Env({0: VAR(5), 2: VAR(3)})

    # Test the composition property:
    # subst(term, env_compose(lhs, rhs)) == subst(subst(term, lhs), rhs)
    term = JOIN(VAR(0), APP(VAR(1), VAR(2)))
    env_a = Env({0: VAR(3), 1: VAR(4)})
    env_b = Env({3: VAR(5), 4: VAR(6)})

    # Left-hand side of the equation
    lhs_result = subst(term, env_compose(env_a, env_b))

    # Right-hand side of the equation
    rhs_result = subst(subst(term, env_a), env_b)

    # They should be equal
    assert lhs_result is rhs_result

    # Test with more complex terms and environments
    complex_term = APP(ABS(JOIN(VAR(0), VAR(1))), VAR(2))
    env_c = Env({1: TOP, 2: BOT})
    env_d = Env({0: VAR(3)})

    lhs_result = subst(complex_term, env_compose(env_c, env_d))
    rhs_result = subst(subst(complex_term, env_c), env_d)

    assert lhs_result is rhs_result


def test_approximate() -> None:
    quota = 20  # test at least a few non-normal terms
    for total, term in enumerate(enumerator):
        lb, ub = approximate(term)
        if is_normal(term):
            assert lb is term
            assert ub is term
            continue
        logger.info(f"Approximated term {lb} [= {term} [= {ub}")
        assert is_normal(ub)
        assert is_normal(lb)
        assert leq(lb, ub)
        assert leq(lb, term) is not False
        assert leq(term, ub) is not False
        quota -= 1
        if not quota:
            logger.info(f"Approximated {1 + total} terms")
            return
    pytest.fail("Ran out of terms to approximate")


def test_beta_ball() -> None:
    # Case 1: Already normal forms (should return just the term itself for radius 0)
    assert beta_ball(VAR(0), 0) == {VAR(0)}
    assert beta_ball(TOP, 0) == {TOP}
    assert beta_ball(BOT, 0) == {BOT}
    assert beta_ball(ABS(VAR(0)), 0) == {ABS(VAR(0))}

    # Case 2: single beta redex
    x = APP(ABS(APP(VAR(0), VAR(0))), ABS(VAR(0)))
    x_0 = ABS(VAR(0))
    assert beta_ball(x, 0) == {x}
    assert beta_ball(x, 1) == {x, x_0}
    assert beta_ball(x, 2) == {x, x_0}

    # Case 2: single beta redex
    y = APP(ABS(APP(VAR(0), VAR(0))), ABS(VAR(1)))
    y_0 = VAR(0)
    assert beta_ball(y, 0) == {y}
    assert beta_ball(y, 1) == {y, y_0}
    assert beta_ball(y, 2) == {y, y_0}

    # Case 3: two beta redexes
    z = JOIN(x, y)
    z_0 = JOIN(x, y_0)
    z_1 = JOIN(x_0, y)
    z_0_0 = JOIN(x_0, y_0)
    assert beta_ball(z, 0) == {z}
    assert beta_ball(z, 1) == {z, z_0, z_1}
    assert beta_ball(z, 2) == {z, z_0, z_1, z_0_0}
    assert beta_ball(z, 3) == {z, z_0, z_1, z_0_0}

    # Case 4: omega, the self-reducing term
    w = APP(ABS(APP(VAR(0), VAR(0))), ABS(APP(VAR(0), VAR(0))))
    assert beta_ball(w, 0) == {w}
    assert beta_ball(w, 1) == {w}

    # Case 5: an infinite reduction chain
    lam_y_yxx = ABS(APP(VAR(1), APP(VAR(0), VAR(0))))
    v = VAR(0)
    Yv = APP(lam_y_yxx, lam_y_yxx)
    v_Yv = APP(v, Yv)
    vv_Yv = APP(v, v_Yv)
    vvv_Yv = APP(v, vv_Yv)
    assert beta_ball(Yv, 0) == {Yv}
    assert beta_ball(Yv, 1) == {Yv, v_Yv}
    assert beta_ball(Yv, 2) == {Yv, v_Yv, vv_Yv}
    assert beta_ball(Yv, 3) == {Yv, v_Yv, vv_Yv, vvv_Yv}
