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
- **Advanced analysis tools**: Perform domain wall and superstructure analysis,
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
   pip install -r requirements.txt
   ```

The supplied `requirements.txt` includes both core and optional packages.
Modules such as PyOpenGL, PyVista, ASE, and BM3D enable enhanced visualization
or denoising features; if a heavy optional dependency is missing the
corresponding tool will either be disabled or fall back to a simpler
implementation.

## Running the Application

Launch the GUI from the project root:

```bash
python -m lfa.main
```

Sample STM datasets are located in the `data/` directory and can be used to
explore the interface and analysis workflow.

## Typical Workflow

1. **Load data**: Use `File > Open...` to import STM files. Each image is added
   as a separate root in the history panel (e.g., “Original Image 1”,
   “Original Image 2”), letting you run a single analysis across multiple scans.
2. **Inspect metadata**: Select any history node to review original-file details
   and processing parameters in the metadata dock, including the source image
   label for derived nodes.
3. **Preprocess**: Apply leveling, filtering, or denoising operations. Each
   step branches off its originating image, preserving the “Original Image N”
   grouping so you always know which scan you’re editing.
4. **Compute FFT**: Choose a node from any image and run
   `Analysis > Calculate FFT...`; the dialog title indicates which original
   image the FFT belongs to. Use `Apply FFT` to append results without closing
   the dialog—handy for sampling multiple ROIs or parameter sets before you hit
   `Close`.
5. **Analyze substrate**: In the FFT analysis dock, select your substrate,
   open `Select Substrate Spots...`, pick the primary Bragg peaks, and compute
   the affine transform that corrects drift and distortions.
6. **Analyze adsorbate**: Create adsorbate sets, open `Select Adsorbate
   Spots...`, and mark overlayer peaks. You can reuse substrate transforms from
   one image on FFTs derived from another.
7. **Obtain parameters**: Use the analysis dock and specialized dialogs to
   compute lattice constants, domain wall periodicity, and real-space vectors.
8. **Persist your work**: Save the session via `File > Save Analysis…`. Sessions
   now capture all images, history branches, and cross-image metadata. Reloading
   restores every “Original Image N” grouping so you can continue exactly where
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
- Domain wall and superstructure periodicity derived from satellite peak
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
  gui/                   PyQt6 widgets, dialogs, and main window
  logic/                 Application controller and history management
  visualization/         Rendering helpers and overlay logic
tests/                   PyTest-based unit and widget tests
docs/                    Sphinx documentation (source and build targets)
data/                    Sample STM datasets
scripts/                 Utility scripts and helpers
```

## Development Notes

- Run the automated test suite with `pytest`.
- Build the documentation with

  ```bash
  sphinx-build -b html docs/source docs/build/html
  ```

- Follow the layered architecture outlined in `context.md` and keep new code,
  comments, and documentation in English for consistency.
- Avoid modifying sample data in `data/` unless you intend to provide updated
  reference datasets.

## License

This project is distributed under the terms of the MIT License. See `LICENSE`
for details.
