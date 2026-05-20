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
- **Type checking**: mypy (`tox -e mypy`).
- **Tests**: pytest with coverage (`tox -e tests`).
- **CI**: Tox-based. Run `tox -e lint` for linting and `tox -e tests` for the test suite.
- **Python**: 3.11+ syntax; no compatibility shims for older versions.
- Docstrings follow **NumPy/PEP 257** conventions (enforced by `ruff D`-rules).

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
- Physical quantities use **MDAnalysis base units**: length in Å, mass in u, time in ps,
  energy in kJ/mol, charge in e, force in kJ/(mol·Å), speed in Å/ps.
- MDAnalysis `AtomGroup` objects are the primary input; `Universe` is accessed via
  `self._universe` in base classes.

## What to avoid

- Do not add Python 2 compatibility code.
- Do not mock MDAnalysis internals in tests; use the test data in `tests/data/`.
- Do not introduce new dependencies without discussion; keep the dependency footprint
  small.
- Do not generate fully automated issues or PRs.
