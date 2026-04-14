"""Tests for the CI drift exit-code gate in src.monitoring.drift_decision.

Covers should_trip_ci_gate() across all (severity, fail_on_severity, interactive)
combinations and verifies the exit-code contract constant.
"""

from __future__ import annotations

import pytest

from src.monitoring.drift_decision import (
    EXIT_CODE_DRIFT_GATE,
    should_trip_ci_gate,
)


def test_exit_code_is_two():
    """Exit code 2 is reserved for 'drift gate tripped' (distinct from 1)."""
    assert EXIT_CODE_DRIFT_GATE == 2


# ---------------------------------------------------------------------------
# Interactive runs never trip the gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("severity", ["low", "medium", "high"])
@pytest.mark.parametrize("fail_on", ["low", "medium", "high"])
def test_interactive_never_trips(severity, fail_on):
    assert should_trip_ci_gate(severity, fail_on, is_interactive=True) is False


# ---------------------------------------------------------------------------
# "never" disables the gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("severity", ["low", "medium", "high"])
def test_never_disables_gate(severity):
    assert should_trip_ci_gate(severity, "never", is_interactive=False) is False


# ---------------------------------------------------------------------------
# Non-interactive gate: observed >= threshold trips; below does not
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "severity,fail_on,expected",
    [
        # fail_on=high (default, conservative)
        ("low",    "high",   False),
        ("medium", "high",   False),
        ("high",   "high",   True),
        # fail_on=medium
        ("low",    "medium", False),
        ("medium", "medium", True),
        ("high",   "medium", True),
        # fail_on=low (strictest)
        ("low",    "low",    True),
        ("medium", "low",    True),
        ("high",   "low",    True),
    ],
)
def test_non_interactive_threshold(severity, fail_on, expected):
    assert should_trip_ci_gate(severity, fail_on, is_interactive=False) is expected


# ---------------------------------------------------------------------------
# Unknown severity strings are treated as missing signal
# ---------------------------------------------------------------------------

def test_unknown_severity_does_not_trip():
    assert should_trip_ci_gate("unknown", "high", is_interactive=False) is False


def test_unknown_fail_on_does_not_trip():
    # Loader validates this, but the helper must be safe on garbage input.
    assert should_trip_ci_gate("high", "bogus", is_interactive=False) is False
