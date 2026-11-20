"""Drive every benchmark the front door advertises, through the real stack.

Coverage comes from tests/benchmarks.py, which derives the list from
benchmark-registry.json. That is what makes this exhaustive rather than a
hand-picked sample: a benchmark added to the registry is driven here
automatically. The one deliberate reduction is BO4Mob, where 420 templated names
share a single code path and one per network is enough.

Two things are checked per benchmark: that it answers with the objective shape
clients rely on (one objective, or the declared count for the multi-objective
ones in tests/benchmarks.py), and -- for the deterministic ones -- that its f0
value has not drifted.
"""
import math

import pytest
from bencherscaffold.protoclasses.bencher_pb2 import EvaluationResult, Value, ValueType


def _point(benchmark):
    value_type = getattr(ValueType, benchmark["value_type"])
    return [Value(type=value_type, value=benchmark["fill"])
            for _ in range(benchmark["dimensions"])]


@pytest.mark.e2e
def test_benchmark_answers_with_its_declared_objectives(client, benchmark, recorder, goldens):
    """Every advertised benchmark must be reachable and answer in the MOBO shape.

    This also records the f0 value when run with --update-goldens, so a single
    sweep both verifies the stack and refreshes goldens.json.
    """
    name = benchmark["name"]
    result = client.evaluate_point(name, _point(benchmark))

    assert isinstance(result, EvaluationResult), f"{name} returned {type(result).__name__}"
    expected = benchmark["objectives"]
    assert len(result.objectives) == expected, (
        f"{name} reported {len(result.objectives)} objectives, expected {expected}")
    names = [o.name for o in result.objectives]
    assert names == [f"f{i}" for i in range(expected)], f"{name} named its objectives {names!r}"
    for o in result.objectives:
        assert math.isfinite(o.value), f"{name} returned {o.name}={o.value}"
    objective = result.objectives[0]

    if recorder is not None and benchmark["deterministic"]:
        recorder[name] = objective.value


@pytest.mark.e2e
def test_deterministic_benchmark_matches_its_golden(client, benchmark, golden_for, recorder):
    """Catches silent numerical drift -- a normalisation or sign regression."""
    name = benchmark["name"]
    if not benchmark["deterministic"]:
        pytest.skip(f"{name} is stochastic; no golden until random_seed is wired through")
    if recorder is not None:
        pytest.skip("recording goldens")
    expected = golden_for(name)
    if expected is None:
        pytest.skip(f"no golden recorded for {name} yet (run with --update-goldens)")

    actual = client.evaluate_point(name, _point(benchmark)).objectives[0].value
    assert math.isclose(actual, expected, rel_tol=1e-9), (
        f"{name} returned {actual!r}, golden is {expected!r}. If this change is "
        f"intended, re-run with --update-goldens.")


@pytest.mark.e2e
def test_deterministic_benchmark_is_actually_deterministic(client, benchmark, recorder):
    """A benchmark we pin a golden for must return the same value twice.

    Without this a secretly-stochastic benchmark would have whatever it happened
    to return frozen in as a constant, and fail confusingly later.
    """
    if not benchmark["deterministic"]:
        pytest.skip("declared stochastic")
    point = _point(benchmark)
    first = client.evaluate_point(benchmark["name"], point).objectives[0].value
    second = client.evaluate_point(benchmark["name"], point).objectives[0].value
    assert first == second, (
        f"{benchmark['name']} is declared deterministic but returned {first!r} "
        f"then {second!r}; add it to STOCHASTIC in tests/benchmarks.py")
