import pytest
import z3

from hstar.bridge import py_to_z3
from hstar.grammar import ABS, APP, VAR, Term, app
from hstar.solvers import LEQ
from hstar.synthesis import Synthesizer


@pytest.mark.xfail(reason="timeout")
@pytest.mark.timeout(0.1)
def test_synthesizer() -> None:
    # Sketch: (\x. x r s) == <r,s>
    sketch = ABS(app(VAR(0), VAR(1), VAR(2)))

    # Constraint: s o r [= I
    def constraint(candidate: Term) -> z3.ExprRef:
        lhs = APP(candidate, ABS(ABS(ABS(APP(VAR(1), APP(VAR(2), VAR(0)))))))
        rhs = ABS(VAR(0))
        return LEQ(py_to_z3(lhs), py_to_z3(rhs))

    synthesizer = Synthesizer(sketch, constraint)
    for _ in range(100):
        candidate, valid = synthesizer.step()
        assert isinstance(candidate, Term)
