# Lattice Fourier Analyzer (LFA)

Lattice Fourier Analyzer (LFA) is a Python desktop application for the scientific
analysis of scanning tunneling microscopy (STM) images. It guides the user from
raw STM data to quantitative lattice parameters by combining flexible
preprocessing, reciprocal space analysis, and real space reconstruction inside a
PyQt6 interface.

## Key Capabilities

- **Data import**: Load Omicron `.stp` and SPECS `.s94` files into a unified
  `STMImage` object together with scan size, bias, tunneling current, and
  embedded metadata.
- **Image inspection**: Display height maps with adjustable color maps, scaling,
  and metadata readouts. Inspect the original data or any derived state.
- **Preprocessing toolbox**: Apply plane leveling, polynomial and adaptive
  leveling, blur and sharpen filters, median and local median filtering,
  histogram based adjustments, Gaussian blur/sharpen, non local means, BM3D, and
  other denoising methods. Every tool supports live parameter tweaking and can
  be restricted to a rectangular region of interest (ROI).
- **History management**: A non linear history tree captures every preprocessing
  step and analysis result, making it easy to branch, revisit, or compare
  outcomes.
- **FFT analysis**: Compute 2D FFTs with optional windowing (Hann, Hamming,
  Blackman, etc.) and different display scalings (log magnitude, power spectrum,
  linear magnitude). FFTs can be generated for the full image or a chosen ROI.
- **Interactive spot selection**: Precisely select reciprocal lattice spots
  through draggable ROIs with refinement modes such as direct click, maximum
  pixel, or 2D Gaussian fitting.
- **Drift and distortion correction**: Derive the affine transformation that
  maps measured substrate peaks to an ideal lattice, yielding rotation, strain,
  and fit quality metrics.
- **Adsorbate lattice analysis**: Manage multiple adsorbate spot sets, apply the
  substrate correction automatically, and compute real space lattice vectors for
  each overlayer.
- **Custom overlays**: Report overlays let you define a custom real-space
  adsorbate lattice (manual vectors or length/angle input, custom offset and marker shape) and toggle visibility per adsorbate set.
- **Advanced analysis tools**: Perform superstructure periodicity analysis,
  generate Patterson (autocorrelation) maps, and reconstruct real space images
  from masked FFT components.
- **Session persistence**: Save and restore complete analysis sessions,
  including controller state, history tree, and selected peaks.

## Installation

1. Install Python 3.9 or newer.
2. Clone or download this repository.
3. (Optional) Create and activate a virtual environment.
4. Install dependencies:

   ```bash
   # Core runtime + analysis stack
   pip install -r requirements-core.txt

   # Optional 3D/advanced extras (PyVista, ASE, bm3d, ...)
   pip install -r requirements-optional.txt
   ```

   A convenience `requirements.txt` aggregates the core file for users who prefer a
   single install command. Leave the optional extra out if you do not need 3D
   visualisation or heavy denoising; the UI will gracefully hide those tools when the
   libraries are missing.

## Running the Application

Launch the GUI from the project root:

```bash
python -m lfa.main
```

Sample STM datasets are located in the `data/` directory and can be used to
explore the interface and analysis workflow.

## 3D Visualization Tips

- Launch the interactive 3D lattice viewer from the real-space visualizer to inspect substrate and adsorbate supercells.
- Use the layer offset controls (`dX`/`dY`) to nudge substrate or adsorbate atoms when layers render side-by-side.
- These shifts are stored in the session, letting you fine-tune alignment before exporting screenshots or continuing analysis.

## Optional Dependencies & Fallbacks

`requirements-optional.txt` bundles the heavier packages that power the 3D viewer and
high-end denoising toolchain:

- `pyqtgraph` and `PyOpenGL` accelerate interactive plotting.
- `pyvista`/`pyvistaqt` are used for the 3D lattice viewer; without them the menu entry is disabled.
- `ase` enables crystallographic export/inspection of derived lattices.
- `bm3d` provides state-of-the-art denoising; when absent the BM3D dialog is hidden and other filters remain available.

Install them only on machines where the extra footprint is acceptable. LFA detects their presence at runtime and hides the associated UI when they are missing.

## Reproducible Demo Assets

- The `data/` folder ships with small STM samples (for example `8343.stp`) that back the automated tests and the walkthrough in the documentation.
- `docs/source/publication_workflow.rst` lists the reference session and notebook expectations for publications. Store generated sessions/notebooks alongside your manuscripts so reviewers can replay the full workflow.
- Use `python -m lfa.main --session <path>` to reload a saved `.lfa_proj` file and confirm that preprocessing history, uncertainty metadata, and visual offsets are preserved.

## Publication Workflow & Citation

- Follow the packaging/test checklist in `docs/source/publication_workflow.rst` before tagging a release. It references linting commands, distribution builds, and DOI archival steps.
- `docs/LICENSE_AUDIT.md` summarises bundled third-party licences. Expand it with project-specific libraries or dataset attributions as needed.
- Add release notes to `CHANGELOG.md` and archive tagged builds on Zenodo to mint a DOI. Once the DOI is available, quote it in this section and in the README header.
- The contributing guide (`docs/CONTRIBUTING.md`) captures the submission workflow and links to the same checklist so the community can help reproduce publication artefacts.

## Typical Workflow

1. **Load data**: Use `File > Open...` to import STM files. Each image is added
   as a separate root in the history panel (for example, `Original Image 1`,
   `Original Image 2`), letting you run a single analysis across multiple scans.
2. **Inspect metadata**: Select any history node to review original-file details
   and processing parameters in the metadata dock, including the source image
   label for derived nodes.
3. **Preprocess**: Apply leveling, filtering, or denoising operations. Each
   step branches off its originating image, preserving the `Original Image N`
   grouping so you always know which scan you're editing.
4. **Compute FFT**: Choose a node from any image and run
   `Analysis > Calculate FFT...`; the dialog title indicates which original
   image the FFT belongs to. Use `Apply FFT` to append results without closing
   the dialog - handy for sampling multiple ROIs or parameter sets before you hit
   `Close`.
5. **Analyze substrate**: In the FFT analysis dock, select your substrate,
   open `Select Substrate Spots...`, pick the primary Bragg peaks, and compute
   the affine transform that corrects drift and distortions.
6. **Analyze adsorbate**: Create adsorbate sets, open `Select Adsorbate
   Spots...`, and mark overlayer peaks. You can reuse substrate transforms from
   one image on FFTs derived from another.
7. **Obtain parameters**: Use the analysis dock and specialized dialogs to
   compute lattice constants, superstructure periodicity, and real-space vectors.
8. **Persist your work**: Save the session via `File > Save Analysis...`. Sessions
   now capture all images, history branches, and cross-image metadata. Reloading
   restores every `Original Image N` grouping so you can continue exactly where
   you left off or load legacy sessions saved before multi-image support was
   added.

## Quantitative Outputs

LFA provides several measurable results:

- Real space lattice vectors (magnitudes and enclosed angle) for both substrate
  and adsorbate lattices.
- Affine transformation matrix and translation, along with derived rotation and
  principal stretch values that describe instrumental drift or sample strain.
- Fit quality metrics, including root mean square error (RMSE) for peak
  alignment.
- Superstructure periodicity derived from satellite peak
  analysis.
- Patterson (autocorrelation) maps and masked FFT reconstructions to connect
  reciprocal features with real space motifs.

## Project Structure

```
lfa/                     Core application package
  core/                  Data models, constants, and shared structures
  io/                    STM readers and writers
  preprocessing/         Image processing operators and pipelines
  analysis/              FFT, lattice math, drift, and domain tools
  logic/                 Application controller, session state, and services
    services/            History orchestration, spot management, analysis runners, session IO
  gui/                   PyQt6 presentation layer
    actions/             Menu helpers and dialog launchers
    controllers/         UI coordinators (DialogCoordinator, HistoryViewHandler, UIStateBinder)
    dialogs/             Interactive tools (spot selection, preprocessing, FFT, reports)
    panels/              Dock widgets embedded in the main window
    visualizers/         2D/3D canvas management
    widgets/             Reusable input components and metadata views
  visualization/         Rendering helpers and overlay logic
  utils/                 Shared helpers (formatting, math, configuration)
tests/                   PyTest-based unit and widget tests
docs/                    Sphinx documentation (source and build targets)
data/                    Sample STM datasets
scripts/                 Utility scripts and helpers
```

## Architecture Overview

LFA now follows a service-oriented layering that separates data handling, business
logic, and presentation concerns:

- **Domain foundations** (`lfa.core`, `lfa.preprocessing`, `lfa.analysis`): hold immutable
  data models (`STMImage`, history nodes), image processing operators, FFT/spot fitting
  math, and uncertainty propagation helpers. These modules remain UI-agnostic and are
  exercised directly by unit tests.
- **Application services** (`lfa.logic.services`): consolidate controller workloads into
  narrow classes. `HistoryOrchestrator` wraps HistoryManager mutations, `SpotSetService`
  coordinates substrate/adsorbate selections and affine fits, `AnalysisExecutor` funnels
  heavy calculations (FFT, superstructure, reconstructions), and `SessionService` handles
  migrations plus persistence. The top-level `AppController` acts as a lightweight facade
  that wires the services together and exposes signals/state for the GUI.
- **Presentation controllers** (`lfa.gui.controllers` and `lfa.gui.actions`): the main
  window delegates menu and dialog orchestration to `DialogCoordinator`,
  `ProcessingDialogLauncher`, `HistoryViewHandler`, and `UIStateBinder`. They translate
  Qt events into controller calls, keep docks/actions enabled in sync with history state,
  and encapsulate dialog wiring so the window class focuses on layout.
- **Dialogs and visualizers** (`lfa.gui.dialogs`, `lfa.gui.visualizers`): each dialog now
  exposes a viewmodel-style API. Shared preprocessing behaviour lives in
  `BasePreprocessingDialog`, while the real-space visualizer reuses controller services to
  retrieve lattice parameters, uncertainties, and overlays.
- **History & session pipeline** (`lfa.logic.history_manager`, `lfa.logic.session_state`):
  maintain a non-linear tree of analysis states with full uncertainty metadata. Session
  files are versioned and upgraded on load; migrations capture schema changes introduced
  during refactors.

This decomposition reduces the responsibilities of the Qt layer, simplifies testing, and
mirrors the layered architecture outlined in `context.md`.

## Development Notes

### Quality Gates
- GUI smoke tests: `pytest tests/gui/smoke --maxfail=1` (covers aggregated preprocessing dialogs, FFT workflow, and key viewmodels; runs with pytest-qt under xvfb/Windows headless).
- Controller and service coverage: `pytest --cov=lfa --cov-report=term-missing` (CI fails under 80% combined coverage for `lfa` and `tests`).
- Lint/type gates: `ruff check .`, `black --check lfa tests`, `mypy lfa` (enabled in CI via pre-commit hooks and GitHub Actions).
- Documentation build: `sphinx-build -b html docs/source docs/build/html` (warnings are treated as errors in CI).
- Smoke GUI sanity: `pytest tests/gui/test_main_window_workflow.py` exercises load → FFT → substrate/adsorbate analysis → real-space visualization.

### Development Environment

- Install development dependencies:

  ```bash
  pip install -r requirements-core.txt -r requirements-dev.txt
  # Optional extras for 3D / denoising
  pip install -r requirements-optional.txt
  ```

- Format and lint the codebase:

  ```bash
  black lfa tests
  ruff check .
  ```

- Run static type checks:

  ```bash
  mypy lfa
  ```

- Execute the automated test suite with `pytest`.
- Build the documentation with

  ```bash
  sphinx-build -b html docs/source docs/build/html
  ```

- Follow the layered architecture outlined in `context.md` and keep new code,
  comments, and documentation in English for consistency.
- Avoid modifying sample data in `data/` unless you intend to provide updated
  reference datasets.
- When preparing a tagged release, consult ``docs/source/release_checklist.rst``
  (also available in the generated HTML docs under Guides) for a full run-down
  of required QA and packaging steps.

### Testing Suite Overview

- **Unit tests** guard math-heavy modules (`lfa.analysis`, `lfa.logic.services`), history
  orchestration, uncertainty propagation, and session serialization.
- **Widget/controller tests** (pytest-qt) validate individual dialogs, presenters, and the
  `UIStateBinder` enablement logic.
- **Smoke tests** simulate end-to-end GUI flows for preprocessing dialogs and the
  main-window workflow, ensuring signal wiring survives refactors without demanding a full
  manual QA pass.
- **Reporting/exports tests** confirm that CSV/JSON summaries include uncertainties,
  calibration sigmas, and transform covariances introduced during milestones 3b.x.

See `CHANGELOG.md` for a curated list of user-facing improvements in recent iterations.

## License

This project is distributed under the terms of the MIT License. See `LICENSE`
for details.


