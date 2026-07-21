"""Minimal smoke test (sanity check for the environment setup)."""

import epistemic_inertial_calibration as eic


def test_version():
    assert eic.__version__ == "0.1.0"


def test_core_deps_importable():
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import sympy  # noqa: F401
