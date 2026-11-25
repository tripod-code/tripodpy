# Changelog

## [Unreleased]

### Added
- Quick start notebook, statement of need and "Contributing and support" page in the documentation.
- Contributing guide, issue templates and pull request template.
- Comparison setups with DustPy in `tests/` (planetary gap, ice lines) with plotting
  notebooks and sample DustPy snapshots, including the setup of the JOSS paper figure.

### Changed
- Dependencies are specified only in `pyproject.toml`: `dustpy>=1.0.8`, `simframe>=1.0.6,<1.1`.
  The GitHub versions of DustPy and Simframe are no longer needed.
- Unit tests run automatically on every push to `main` and on every pull request.

### Removed
- `requirements.txt`. Use `pip install .` instead.

### Fixed
- Example scripts in `tests/` import `tripodpy` instead of `tripod` and no longer need `dustpylib`.

## [1.0.0.beta] - 2026-10-22

First public release, submitted to the Journal of Open Source Software.

[Unreleased]: https://github.com/tripod-code/tripodpy/compare/v1.0.0.beta...HEAD