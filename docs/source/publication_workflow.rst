Publication Workflow and Optional Dependencies
==============================================

This guide summarises practical steps for preparing a publication-ready build of
LFA, the optional dependencies that unlock advanced features, and the assets
you should curate when sharing results with collaborators or reviewers.

Packaging & Test Checklist
--------------------------

1. Install the core requirements (`requirements-core.txt`) and run the lint/test
   commands listed in :file:`docs/TEST_LINT_COMMANDS.csv` (or the equivalent
   table in the README) until they pass.
2. Install the optional extras (`requirements-optional.txt`) if you rely on the
   3D viewer, ASE export, or BM3D denoising. The UI hides those tools when the
   libraries are absent, so install only what you need.
3. Build the documentation with ``sphinx-build -b html docs/source docs/build/html``.
4. Generate distribution artefacts with ``python -m build`` and smoke-test the
   resulting wheel/tarball in a clean environment.
5. Draft release notes (see :doc:`release_checklist`) and record highlights in
   :file:`CHANGELOG.md`.
6. Archive the tagged release on Zenodo (or another DOI service) so the dataset
   and binaries are citable. Once the DOI is minted, quote it in the README and
   in your manuscripts.

Demo Video & CI Artefacts
-------------------------

*Recording*: use ``ffmpeg`` (or another recorder) to capture a short session
covering import → preprocessing → FFT → real-space visualiser. Example commands:

.. code-block:: bash

   ffmpeg -f gdigrab -framerate 30 -i desktop -t 00:00:45 demo.mp4
   ffmpeg -i demo.mp4 -vf "fps=12,scale=960:-1:flags=lanczos" demo.gif

Place the resulting ``demo.mp4``/``demo.gif`` in ``docs/media/`` so they can be
embedded in the documentation or README.

*CI Artefacts*: extend your CI pipeline to upload ``coverage.xml``, the zipped
documentation build (``docs/build/html``), and any demo media via the CI
platform's artefact feature (for example, GitHub Actions ``upload-artifact``).
This gives reviewers quick access to coverage reports and documentation without
rebuilding locally.

Optional Dependencies & Fallback Behaviour
------------------------------------------

``requirements-optional.txt`` contains packages that activate specialised
features:

* ``pyqtgraph`` and ``PyOpenGL`` (faster FFT and ROI rendering).
* ``pyvista`` / ``pyvistaqt`` (interactive 3D lattice viewer). Without them the
  "Visualize Real Space" dialog omits the 3D tab.
* ``ase`` (exporting supercells into crystallographic toolchains).
* ``bm3d`` (state-of-the-art denoising filter). When missing, the dialog hides
  itself and the rest of the preprocessing pipeline remains available.

LFA probes for these modules at runtime and disables dependent dialogs or menu
entries when they are not installed.

Demo Assets & Reproducibility
-----------------------------

* The :mod:`data` directory ships with compact STM scans used by the automated
  tests (for example ``8343.stp``). Use them when validating CI pipelines or
  reproducing documentation screenshots.
* Store representative analysis sessions (`*.lfa_proj`) alongside your papers.
  They retain preprocessing history, uncertainty metadata, and manual offsets.
* Capture scripted workflows in a companion notebook or Python script. A
  minimal pattern is:

  .. code-block:: python

     from pathlib import Path
     from lfa.logic.session_serializer import SessionSerializer
     from lfa.logic.app_controller import AppController
     # load a session and extract summary metrics for the manuscript

* Document the location of the raw STM files, processed sessions, and export
  tables in your project README so reviewers can replay the steps.
* Host the demo MP4/GIF alongside the release notes so readers can preview the
  workflow before installing the application.

Citation & Licensing
--------------------

* :file:`docs/LICENSE_AUDIT.md` lists bundled third-party licences. Extend it
  with any dataset-specific terms or lab-specific references before publishing.
* :file:`docs/CONTRIBUTING.md` describes how to run the lint/test suite, submit
  patches, and link release notes to DOI records.
* Once a DOI is available, cite it in the README and in the ``Publication
  Workflow & Citation`` section so downstream users can reference the exact
  build.
