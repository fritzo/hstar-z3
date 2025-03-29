import logging

import pytest
import z3

from hstar.bridge import nf_to_z3
from hstar.language import LEQ
from hstar.normal import ABS, APP, VAR, Term, app
from hstar.synthesis import Synthesizer

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

    def on_fact(term: Term, valid: bool) -> None:
        assert isinstance(term, Term)
        logger.debug(f"{term}, valid = {valid}")

    synthesizer = Synthesizer(
        sketch,
        constraint,
        on_fact,
        timeout_ms=10,
    )
    for _ in range(10):
        synthesizer.step()
