# Changelog

All notable changes will be documented in this file. The project follows semantic
versioning once a DOI-backed release is published.

## Unreleased

### Added
- Split dependency specification into `requirements-core.txt` and
  `requirements-optional.txt`, clarifying when PyVista/ASE/BM3D are needed.
- Documented publication workflow, citation guidance, and demo asset expectations in
  both the README and the Sphinx docs.
- Introduced real-space visualizer presenter unit tests, raising end-to-end coverage above
  the 80 % gate.
- Added licence audit and contributing guides to assist with publication-ready releases.

### Changed
- Refactored the real-space visualizer dialog to delegate formatting logic to a presenter,
  improving separation of concerns and testability.
- Updated development and installation instructions to reflect the new dependency layout.

### Removed
- None.
