"""Epistemic Inertial Calibration.

Research codebase for identifying a robot's dynamic parameters (mass, center of
mass, moment of inertia = inertial parameters) through information-seeking
excitation motions that actively reduce their uncertainty.

The final S5 pipeline studies a three-layer decomposition: base-parameter
identifiability, task-compatible excitation, and terminal feasibility.

Main modules:
  - dynamics:    planar 2-DoF/4-DoF rigid-body dynamics and regressor Y(q, qd, qdd)
  - parameters:  standard inertial parameters phi and SVD base-subspace extraction
  - estimation:  IDIM torque observation and a base-coordinate Bayesian posterior
  - excitation:  excitation policies (static/random/fourier/fim_greedy/active_ig)
  - evaluation:  identifiability, task compatibility, terminal-risk diagnostics
"""

__version__ = "0.1.0"
