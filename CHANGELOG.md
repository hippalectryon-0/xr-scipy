# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0]

The main breaking change is the removal of custom filters that are not in the scipy API.
This update re-affirms the goal of this library - to be a wrapper around `scipy` functions for easy use on xarrays, not to add additional functions.

### Added

- Add many tests, especially on `xrscipy.signal`
- Add `integrate.cumulative_simpson`
- Add tentative support for `xarray<2023,>0.16`

### Changed

- Rewrote `savgol_filter` and `decimate` to match the `scipy` implementation.
- Renamed `psd` to `welch` to align with `scipy`
- `welch` now returns `PSD**2`, as in `scipy`

### Fixed

- Fixed n-dimentional `hilbert`

### Removed

- Remove all filters that are not part of the `scipy` API

## [2.2.0]

### Changed

- Transition from `poetry` to `uv` and `black/isort/pycln` to `ruff`
- Updated dependencies

### Fixed

- Minor docs fixes

## [2.2.0]

### Changed

- Transition from `poetry` to `uv` and `black/isort/pycln` to `ruff`
- Updated dependencies

### Fixed

- Minor docs fixes

## [2.0.0]

### Added

- `signal.extra.psd` is now available as `signal.welch` (as in `scipy`)

### Changed

- Move `other.signal` into `signal` and `signal.extra` to better mirror `scipy`'s namespace (thanks @smartass101)
- Remove `urllib` dependency (not relevant anymore)

### Fixed

- Fix publishing CI

## [1.1.2]

### Changed

- Update docs

### Fixed

- Fix pypi pipeline

## [1.1.1]

### Fixed

- Fix publishing CI

## [1.1.0]

### Changed

- Refactor some inner functions in `fft, fftpack`
- Bump minimum python version to 3.10 to support pipes `|` in typing.
- Change arg `a` to `x` in scipy.fftpack to match the scipy signature
- Add package to pipy

### Fixed

- Correctly rename `shape` to `s` in the docs for `fft`

## [1.0.0]

### Added

- Add a changelog
- Add `pre-commit`
- Add `poetry`
- Add several typing hints

### Changed

- Turn the travis CI into a github CI
- Update the README
- Move several doc files into python docstrings
- Bump the minimum version of python to `3.9`. Also bump the requirements for `scipy` and `xarray`. If those are too restrictive, we may broaden them later.
- Refactor docs
- `xrscipy.fft` now mirrors `scipy.fft` rather than `numpy.fft`
- Replace custom documentation parsed by tweaked `docstring_parser` module
- Move `xrscipy.signal` to `xrscipy.other.signal`

### Fixed

- Fix `fft, fftpack, integrate` utests and modules

### Removed

- Remove the `interpolate` module, which is now supported natively by `xarray`

## [0.1.0] (6 March 2018)

Initial release. This is the "old" version.
