# GitHub Copilot Instructions for MAICoS

## Project overview

MAICoS (**Molecular Analysis for Interfacial and Confined Systems**) is a Python toolkit
for analyzing the structure and dynamics of interfacial and confined fluids from
molecular simulations. It builds on [MDAnalysis](https://www.mdanalysis.org)
and supports trajectory files from LAMMPS, GROMACS, CHARMM, and NAMD.

## Repository layout

```
src/maicos/
  core/        # Base classes (planar, cylinder, sphere geometries)
  modules/     # Analysis modules (DensityPlanar, DielectricPlanar, …)
  lib/         # Shared library utilities
tests/
  core/        # Tests for base classes
  modules/     # Tests per analysis module
docs/          # Sphinx documentation
```

## Code style and tooling

- **Formatter/linter**: [Ruff](https://docs.astral.sh/ruff/) (`ruff format` + `ruff check`). Line length 88.
- **Tests**: pytest with coverage (`tox -e tests`).
- **CI**: Tox-based. Run `tox -e lint` for linting and `tox -e tests` for the test suite.
- **Python**: 3.11+ syntax; no compatibility shims for older versions.
- **Prefer MDAnalysis**: use MDA functions and methods wherever possible. Custom
  implementations are only justified when profiling shows a significant performance
  regression introduced by the MDA equivalent.
- The same applies to NumPy and SciPy: their implementations are typically faster than
  pure-Python custom code and should be preferred for the same reasons.
- Docstrings follow **NumPy/PEP 257** conventions (enforced by `ruff D`-rules).
- Class docstrings use a template system: shared parameter/attribute descriptions live
  in `DOC_DICT` in `src/maicos/lib/util.py` and are injected at import time via the
  `@render_docs` decorator (`maicos.lib.util.render_docs`). Use `${KEY}` placeholders
  in docstrings and add new entries to `DOC_DICT` rather than duplicating text.

## Contributing guidelines

- Open an issue and discuss changes **before** submitting a pull request.
- All PRs target the `main` branch.
- Every new analysis module or feature must include tests and updated documentation.
- Update `CHANGELOG.rst` for user-visible changes.
- Disclose AI tool usage in the PR description (see PR template).

### Pull request guidelines
Before submitting a pull request, ensure:

- Code is formatted: Run tox -e format and tox -e lint passes before committing changes
- All tox checks pass locally: tox (or specific environments like tox -e tests)
- Tests pass: tox -e tests
- Documentation is updated if functionality changes: docs/ uses Sphinx; update docstrings and .rst files as needed
- PR description includes:  
  - Summary of changes
  - Any new dependencies added
  - Links to relevant issues or discussions
  - Context for reviewers (e.g., "fixes #1234")


## AI-assisted code policy

MAICoS requires human judgment and domain knowledge in interfacial fluid simulations.
Copilot suggestions are acceptable for routine tasks (boilerplate, docstrings, tests),
but **review every suggestion carefully** before accepting it — especially for physical
formulas, unit conversions, and MDAnalysis API usage. Do not accept code you cannot
explain. Significant AI-assisted contributions must be disclosed in the PR.

## Domain context for better suggestions

- Analysis modules inherit from geometry base classes in `src/maicos/core/` (planar,
  cylindrical, spherical).
- Results are stored in `self.results` (an `MDAnalysis.analysis.Results` dict-like
  object). Bin positions go in `self.results.bin_pos`; profiles in
  `self.results.profile`.
- Per-frame observables are written to `self._obs` (a `Results` object reset each
  frame in `_prepare`). The base class accumulates `_obs` entries automatically into
  `self.sums`, `self.means`, and `self.sems` across frames — subclasses must only
  populate `self._obs` inside `_single_frame`, not implement the averaging themselves.
- Physical quantities use **MDAnalysis base units**: length in Å, mass in u, time in ps,
  energy in kJ/mol, charge in e, force in kJ/(mol·Å), speed in Å/ps.
- MDAnalysis `AtomGroup` objects are the primary input; `Universe` is accessed via
  `self._universe` in base classes.

## Testing

- Tests use **pytest** and live in `tests/`. Mirror the `src/maicos/` layout: one test
  file per module under `tests/modules/`, core tests under `tests/core/`.
- The goal is to maximise **physical tests**: construct systems with known analytical
  solutions (e.g. ideal dipole arrangements) and assert that module results match the
  analytical expectation within numerical tolerance. Prefer this over testing
  implementation details.
- See `tests/modules/test_dielectricplanar.py` and its siblings for the reference
  pattern: synthetic MDAnalysis universes are built from controlled dipole positions and
  orientations, then the output is compared to closed-form expressions derived with
  sympy.

## Documentation and examples

- Docs are built with **Sphinx** from `docs/src/`. Each analysis module has exactly one
  dedicated `.rst` page in `docs/src/analysis-modules/` that uses `.. autoclass::` to
  pull the docstring. When adding a new module, add a corresponding `.rst` page there.
- Python usage examples live in `examples/` and are executed and rendered by
  **sphinx-gallery**. Each example is a plain `.py` file with a reStructuredText
  docstring at the top (the gallery title and description) followed by runnable code.
  Add new examples there rather than in the narrative docs.

## What to avoid

- Do not add Python 2 compatibility code.
- Do not mock MDAnalysis internals in tests; use the test data in `tests/data/`.
- Do not introduce new dependencies without discussion; keep the dependency footprint
  small.
- Do not generate fully automated issues or PRs.
