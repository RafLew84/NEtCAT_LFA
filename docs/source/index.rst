.. Lattice Fourier Analyzer (LFA) documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Lattice Fourier Analyzer (LFA) documentation!
================================================================

LFA is a desktop application designed for the scientific analysis of Scanning Tunneling Microscopy (STM) images. Its primary goal is to provide a comprehensive suite of tools for determining crystal lattice parameters from real-space images by leveraging Fast Fourier Transform (FFT) analysis. The application guides the user from raw data loading and preprocessing to advanced reciprocal space analysis, including drift correction and characterization of adsorbate overlayers.

Key Features
------------

* **Data Handling**: Import common STM file formats (Omicron `.stp`, SPECS `.s94`).
* **Image Preprocessing**: A complete set of tools to improve image quality, including plane leveling, filtering (Gaussian, Median), denoising (NL-Means, BM3D), and sharpening.
* **FFT Analysis**: Robust FFT calculation with support for various windowing functions and scaling modes (Log, Power, Linear).
* **Interactive Spot Selection**: Precise selection of reciprocal lattice spots with refinement methods like "Max Pixel" and "2D Gaussian Fit".
* **Drift & Distortion Correction**: Automatic calculation of the affine transformation matrix that maps the distorted, measured lattice to an ideal one, providing quantitative data on rotation and strain.
* **Adsorbate Lattice Analysis**: Correction of adsorbate spot positions using the calculated substrate transformation, enabling accurate analysis of overlayer structures.
* **Advanced Tools**: Includes modules for superstructure periodicity analysis and real-space reconstruction from a masked FFT.
  * **Interactive 3D Viewer**: Inspect substrate and adsorbate lattices in 3D, with manual XY offsets to align layers before export or screenshots.
* **History Tracking**: A non-linear history system allows users to seamlessly switch between different stages of the analysis.

Core Workflow
-------------

A typical analysis session in LFA follows these steps:

1.  **Load Image**: Open an STM data file (`.stp`, `.s94`) using the ``File > Open...`` menu.
2.  **Preprocess (Optional)**: Use the tools in the ``Preprocessing`` menu to level the background, remove noise, or enhance features. Each operation creates a new state in the history panel.
3.  **Calculate FFT**: Select a processed or original image and use ``Analysis > Calculate FFT...`` to generate the reciprocal space image.
4.  **Analyze Substrate**:
    * In the "FFT Analysis Tools" panel, select the substrate type (e.g., "Au(111)") or define a custom one.
    * Open the ``Select Substrate Spots...`` dialog, select the primary Bragg peaks, and calculate the transformation matrix to correct for drift and distortion.
5.  **Analyze Adsorbate**:
    * In the main panel, create a new "Adsorbate Set".
    * Open the ``Select Adsorbate Spots...`` dialog. The substrate correction is automatically applied to the selected adsorbate spots, transforming them into an ideal, undistorted coordinate system.
6.  **Calculate Parameters**: Use the "Calculate" buttons in the "FFT Analysis Tools" panel to obtain the final real-space lattice parameters for both the substrate and the adsorbate overlayer.

What You Can Calculate
----------------------

LFA allows you to quantify several physical properties of your sample:

* **Lattice Vectors and Constants**: Determine the real-space lattice vectors ($a_1, a_2$), their magnitudes ($|a_1|, |a_2|$), and the angle ($\alpha$) between them for both the substrate and any adsorbate layers.
* **Drift and Distortion Parameters**: Quantify instrumental drift and sample distortion through the affine transformation matrix ($F$). This includes:
    * **Rotation Angle**: The angle of rotation of the measured lattice relative to the ideal one.
    * **Principal Stretches**: The amount of stretching or compression along the principal axes of distortion.
    * **Fit Quality**: The Root Mean Square Error (RMSE) of the transformation fit, indicating how well the measured spots match an ideal lattice.
* **Superstructure Periodicity**: By analyzing the splitting of satellite peaks from main Bragg peaks, you can calculate the real-space periodicity of surface superstructures.
* **Autocorrelation Map**: Generate a Patterson map from the FFT to visualize real-space periodicities and vector relationships.

Architecture Overview
---------------------

The refactor completed in 2025 reorganised LFA into explicit layers:

- **Domain foundations** (``lfa.core``, ``lfa.preprocessing``, ``lfa.analysis``) keep
  immutable STM data models, preprocessing operators, FFT/spot-fitting math, and
  uncertainty propagation helpers. They are UI-agnostic and unit-tested.
- **Application services** (``lfa.logic.services``) provide focused responsibilities:
  ``HistoryOrchestrator`` mutates the history tree, ``SpotSetService`` manages substrate and
  adsorbate selections, ``AnalysisExecutor`` drives FFT/superstructure/real-space jobs, and
  ``SessionService`` handles migrations plus persistence. The thin ``AppController`` composes
  these services for the GUI.
- **Presentation coordinators** (``lfa.gui.controllers`` and ``lfa.gui.actions``) translate
  Qt events into controller calls. ``DialogCoordinator`` centralises dialog wiring,
  ``ProcessingDialogLauncher`` binds preprocessing menus to the shared base dialog,
  ``HistoryViewHandler`` keeps the history dock in sync, and ``UIStateBinder`` toggles actions
  based on the active node.
- **Dialogs and visualisers** (``lfa.gui.dialogs`` / ``lfa.gui.visualizers``) expose
  viewmodel-style APIs. ``BasePreprocessingDialog`` houses the live-preview/ROI logic used by
  all preprocessing tools, while the real-space visualiser consumes uncertainty-enriched data
  supplied by the services layer.
- **History & session pipeline** (``lfa.logic.history_manager``, ``lfa.logic.session_state``)
  manages a non-linear tree of analysis states. Session files are versioned and upgraded when
  older projects are opened.

This structure improves testability and isolates future enhancements such as non-linear drift
correction and additional import formats.

Testing and Quality Gates
-------------------------

The project ships with an opinionated quality bar that can be reproduced locally:

- **Unit tests** (``pytest``) cover uncertainty propagation, service orchestration, and the
  math helpers in ``lfa.analysis``.
- **Widget/controller tests** (``pytest-qt``) target the dialog viewmodels and controllers,
  including ``UIStateBinder`` enablement logic.
- **GUI smoke tests** (``pytest tests/gui/smoke``) open each preprocessing dialog through
  ``ProcessingDialogLauncher`` to verify shared base behaviour, while
  ``tests/gui/test_main_window_workflow.py`` exercises the primary load → FFT → substrate/adsorbate
  analysis → visualisation flow.
- **Static analysis**: ``ruff``, ``black``, and ``mypy`` run in CI (see ``requirements-dev.txt``).
- **Coverage**: ``pytest --cov=lfa --cov-report=term-missing`` must stay above 80% for logic and
  controller layers; the CI pipeline enforces ``--cov-fail-under=80``.
- **Documentation**: build with ``sphinx-build -b html docs/source docs/build/html``; warnings are
  treated as errors to prevent stale references.

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/modules.rst
