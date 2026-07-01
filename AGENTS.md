# AGENTS.md

## Cursor Cloud specific instructions

This is a single Python project: a Streamlit UI (`app.py`) on top of the reusable
`avldrive` engine package, with a `pytest` suite in `tests/`. Standard install/run
commands are in `README.md`.

Dependencies are installed into a virtualenv at `.venv/` (gitignored). Use
`.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/streamlit` (or activate it).

- **Run tests:** `PYTHONPATH=. .venv/bin/pytest -q` — the `PYTHONPATH=.` is
  required because `tests/conftest.py` imports the top-level `avldrive` package but
  there is no packaging/`conftest.py` at the repo root to put it on `sys.path`.
  The suite builds a synthetic `.mf4` in a temp dir via `asammdf`, so no fixtures
  are checked in.
- **Run the app:** `PYTHONPATH=. .venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true`.
  `app.py` calls `main()` at import time (both under `__main__` and when imported),
  so it must be launched via `streamlit run`, not `python app.py`.
- **Using the app:** it does nothing until you upload a measurement. It needs an
  `.mf4`/`.dat` with at least `AcceleratorPedal` and `AccelerationChassis`
  channels. To smoke-test without real data, generate a synthetic `.mf4` using the
  same signal recipe as `tests/conftest.py::synthetic_mf4` and upload it via the
  sidebar file uploader.
- **Lint:** there is no linter configured (no ruff/flake8/black config or dev dep).
