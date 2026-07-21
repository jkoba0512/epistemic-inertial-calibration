# Using an external virtual environment

By default, `uv sync` creates `.venv/` inside the repository. This directory is
gitignored. On a network-mounted or synchronized filesystem, keeping the virtual
environment outside the repository can avoid synchronization overhead and
platform-specific binary conflicts.

Set a dedicated environment path before running `uv`:

```bash
export UV_PROJECT_ENVIRONMENT=/absolute/path/to/epistemic-inertial-calibration-venv
uv sync
uv run pytest
uv run ruff check .
```

Use a different environment for each operating system. Do not synchronize or
commit virtual-environment contents. The committed `uv.lock` file defines the
resolved dependency versions used by the project.
