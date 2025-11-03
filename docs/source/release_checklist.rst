.. _release-checklist:

==========================
Release Preparation Guide
==========================

This checklist captures the tasks we expect to complete before tagging and
publishing an official LFA release.  Adapt it as needed for hotfixes, but avoid
skipping steps for feature releases or publication artefacts.

Metadata & Planning
===================

#. Confirm the target version number and update it wherever it is surfaced:
   ``docs/source/conf.py`` (``release``), installer scripts, and any About
   dialogs.
#. Draft or refresh release notes (``CHANGELOG.md`` if present, or add a new
   entry under ``notes/``) summarising user-visible changes since the previous
   release.
#. Review open items in :doc:`migration_notes` and ``todo.md``; close or defer
   anything that cannot ship.
#. Ensure sample datasets and reference sessions in ``data/`` are current and
   include the metadata used in tutorials.
#. Decide which dependency profile to ship: install ``requirements-core.txt`` for the
   base toolchain and ``requirements-optional.txt`` when 3D visualisation or ASE exports
   are part of your publication.

Automated Quality Gates
=======================

Run the same tooling enforced by CI, capturing artefacts (coverage XML, lint
reports) for archival with the release:

.. code-block:: bash

   # Linting and formatting
   ruff check .
   black --check lfa tests

   # Auto-fix import order before linting (matches CI expectations)
   python -m ruff check --select I --fix .

   # Static typing
   mypy lfa

   # Full test suite with coverage (matches pytest.ini options)
   pytest

   # GUI smoke workflow
   pytest tests/gui/test_main_window_workflow.py

   # Preprocessing dialog smoke coverage
   pytest tests/gui/smoke --maxfail=1

Manual Verification
===================

#. Launch the application via ``python -m lfa.main`` using a clean environment.
#. Load each sample STM file in ``data/`` and confirm:

   - preprocessing dialogs open through the menu and respect ROI/live preview toggles,
   - FFT calculation succeeds on at least one ROI and on the full image,
   - substrate and adsorbate spot selection produce lattice vectors with uncertainties,
   - ``Analysis -> Visualize Real Space`` becomes available after computing parameters, and the dialog shows uncertainties plus layer offsets,
   - real-space reconstruction runs when complex FFT data is present.
#. Inspect exports (JSON/CSV, clipboard summaries) and session save/load to
   verify uncertainties and calibration sigmas persist.
#. Run the real-space visualiser's 3D viewer, ensure manual offsets persist
   through session save/load, and take screenshots for documentation if needed.

Documentation
=============

#. Update screenshots or walkthroughs in ``README.md`` and Sphinx guides if the
   UI changed.
#. Build the docs with warnings treated as errors:

   .. code-block:: bash

      sphinx-build -b html docs/source docs/build/html

#. Upload the rendered HTML (``docs/build/html``) or ensure the CI artefact is
   attached to the release.

Packaging & Distribution
========================

#. Verify the source tree is clean (`git status` shows no untracked artefacts
   besides generated docs).
#. Create distribution archives (requires ``build`` or a similar tool):

   .. code-block:: bash

      python -m build

#. Smoke-test the generated wheel/tarball inside a fresh virtual environment.
#. Tag the release (``git tag vX.Y.Z``) and push both commit and tag.  If a DOI
   archive (e.g., Zenodo) is required, upload the matching artefacts and record
   the DOI in the release notes.

Post-Release
============

#. Close or update items in ``todo.md`` and ``plan.md`` to reflect the release
   status.
#. Announce the release through the chosen channels (project page, lab wiki,
   mailing list) and reference the migration notes for users upgrading from
   earlier versions.
#. Monitor CI and issue tracker for regression reports; create a follow-up plan
   for any hotfixes discovered shortly after publication.
