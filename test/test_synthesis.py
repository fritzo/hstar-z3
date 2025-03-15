import logging

import pytest
import z3
from immutables import Map

from hstar import solvers
from hstar.bridge import nf_to_z3
from hstar.normal import ABS, APP, VAR, Env, Term, app, env_free_vars
from hstar.solvers import LEQ
from hstar.synthesis import CEGISSynthesizer, EnvSynthesizer, Synthesizer

logger = logging.getLogger(__name__)


@pytest.mark.skip(reason="slow")
@pytest.mark.timeout(5)
def test_synthesizer() -> None:
    # Sketch: (\x. x r s) == <r,s>
    sketch = ABS(app(VAR(0), VAR(1), VAR(2)))

    # Constraint: s o r [= I
    def constraint(candidate: Term) -> z3.ExprRef:
        lhs = APP(candidate, ABS(ABS(ABS(APP(VAR(1), APP(VAR(2), VAR(0)))))))
        rhs = ABS(VAR(0))
        return LEQ(nf_to_z3(lhs), nf_to_z3(rhs))

    synthesizer = Synthesizer(sketch, constraint)
    for _ in range(10):
        candidate, valid = synthesizer.step(timeout_ms=10)
        logger.debug(f"{candidate}, valid = {valid}")
        assert isinstance(candidate, Term)


@pytest.mark.skip(reason="slow")
@pytest.mark.timeout(5)
def test_cegis_synthesizer() -> None:
    # Sketch: (\x. x r s) == <r,s>
    sketch = ABS(app(VAR(0), VAR(1), VAR(2)))

    # Constraint: For all inputs x, s(r(x)) [= x
    # This is the extensional form of "s o r [= I"
    def constraint(candidate: Term, example: z3.ExprRef) -> z3.ExprRef:
        r = nf_to_z3(APP(candidate, ABS(ABS(VAR(1)))))
        s = nf_to_z3(APP(candidate, ABS(ABS(VAR(0)))))
        return LEQ(solvers.APP(s, solvers.APP(r, example)), example)

    synthesizer = CEGISSynthesizer(sketch, constraint)
    for _ in range(10):
        candidate, valid = synthesizer.step(timeout_ms=100)
        logger.debug(f"{candidate}, valid = {valid}")
        assert isinstance(candidate, Term)


@pytest.mark.skip(reason="slow")
@pytest.mark.xfail(reason="timeout")
@pytest.mark.timeout(5)
def test_env_synthesizer() -> None:
    sketch = Env(
        {
            0: ABS(ABS(ABS(app(VAR(2), VAR(0), VAR(1))))),  # pair
            1: app(VAR(0), VAR(2), VAR(3)),  # <r,s>
        }
    )
    assert env_free_vars(sketch) == Map({2: 1, 3: 1})

    def constraint(candidate: Env) -> z3.ExprRef:
        rs = candidate[1]
        CB = ABS(ABS(ABS(APP(VAR(1), APP(VAR(2), VAR(0))))))
        lhs = APP(rs, CB)
        rhs = ABS(VAR(0))
        return LEQ(nf_to_z3(lhs), nf_to_z3(rhs))

    synthesizer = EnvSynthesizer(sketch, constraint)
    for _ in range(10):
        candidate, valid = synthesizer.step(timeout_ms=10)
        logger.debug(f"{candidate}, valid = {valid}")
        assert isinstance(candidate, Map)
