# AtomMapper

AtomMapper is a standalone desktop tool for interactive analysis of STM images at the single-atom and atomic-row level.

The application is built next to the main `lfa` project and reuses its STM file loaders for `.stp` and `.s94` data. It is intended for workflows where the user wants to:

- load an STM image,
- denoise or smooth it without destroying the original data,
- place and move a local ROI,
- refine atom positions with a local 2D Gaussian fit,
- group saved atoms into rows,
- correct saved positions manually,
- inspect row geometry in pixels and nanometers,
- detect local geometric disturbances that may indicate domain-wall candidates,
- export results and save the full analysis session.

## Main capabilities

- STM file loading from `.stp` and `.s94`
- live main-image view based on `pyqtgraph`
- interactive ROI with live ROI preview
- live 2D Gaussian preview for the current ROI
- preprocessing dialog with:
  - Gaussian blur
  - non-local means denoising
  - BM3D denoising when the optional `bm3d` package is available
- image variants derived from the same original file
- atom rows and saved atom points
- point editing:
  - select from the table or directly on the image
  - delete points
  - drag points manually on the STM image
- calibration from image metadata:
  - `px -> nm`
  - point coordinates stored in both `px` and `nm` when calibration is available
- analysis dock with:
  - saved points table
  - selected-row plot
  - global rows plot
  - row geometry metrics
  - row disturbance candidates
- session save and load
- CSV export of saved points

## Requirements

AtomMapper runs inside the same repository as `lfa` and depends on its IO layer.

Minimum practical environment:

- Python 3
- `PyQt6`
- `pyqtgraph`
- `numpy`
- `scipy`
- `scikit-image`

Optional:

- `bm3d`
  - needed only for BM3D preview and BM3D variant creation

If you already use the project environment for `lfa`, use the same interpreter for AtomMapper.

## Launch

Run from the repository root:

```bash
python -m AtomMapper.app.main
```

If your project uses a dedicated environment, activate it first and then run the command above.

## Input data

Supported input formats:

- `.stp`
- `.s94`

Each loaded image keeps:

- the original pixel data,
- image dimensions in pixels,
- physical scan size from metadata,
- derived `nm/px` calibration when metadata are valid.

## Main window layout

### Left panel

- `Load STM Files...`
- `Preprocessing`
- `Export CSV`
- `Save Session`
- `Load Session`
- list of loaded images and derived variants
- `Atom rows` section with:
  - `New Row`
  - `Delete Row`
  - `Add Point`
  - `Delete Point`
  - list of rows for the active image family

### Central / right area

- main STM image
- `ROI preview`
- `Gaussian fit preview`
- `Show Gaussian Fit` checkbox

### Bottom dock: `Analysis`

- `Saved points`
- `Selected row plot`
- `Global rows plot`
- `Row geometry metrics`
- `Row disturbance candidates`

## Typical workflow

### 1. Load an STM image

Use `Load STM Files...` and select one or more `.stp` or `.s94` files.

After loading:

- the first image becomes active,
- a default ROI is created,
- the ROI and Gaussian preview start updating immediately.

### 2. Inspect or preprocess the image

Use `Preprocessing` to open the preprocessing dialog.

The dialog provides:

- original preview,
- processed preview,
- method selection,
- live parameter tuning,
- `Apply` to create a derived image variant.

Current methods:

- `Blur`
- `Non-local means`
- `BM3D` if the backend is installed

Important:

- preprocessing does not overwrite the original image,
- the result is added to the file list as a new variant,
- rows belong to the image family, so you can continue working on the same row across original and derived variants.

### 3. Define rows

Use `New Row` to create a row for the active image family.

Rows are logical containers for atom positions. They are shared across variants derived from the same original file.

### 4. Move the ROI and inspect the Gaussian fit

On the main STM image:

- move the ROI to the atom of interest,
- resize the ROI if needed,
- inspect:
  - `ROI preview`
  - `Gaussian fit preview`

The Gaussian preview is visual only. The saved point will still use the fit result even if the preview is hidden, as long as the current ROI can be fitted.

### 5. Save a point

With an active row selected, click `Add Point`.

Point creation logic:

- preferred source: Gaussian-fit center,
- fallback: ROI center if a stable Gaussian fit is not available.

Saved points are shown:

- in the `Saved points` table,
- on the STM image as overlay markers.

### 6. Select, inspect, and edit points

You can select a saved point:

- by clicking a row in the `Saved points` table,
- or by clicking its marker on the STM image.

You can then:

- `Delete Point`
- drag the selected point directly on the image

Manual movement behavior:

- the point keeps the original fitted position internally,
- the current displayed position becomes the manual position,
- the point status changes to `manual (drag)`.

### 7. Analyze row geometry

The `Selected row plot` can display:

- `x(i)`
- `y(i)`
- `distance(i,i+1)`
- `along(i)`
- `transverse(i)`
- `spacing along(i,i+1)`

Available units:

- `px`
- `nm`

Notes:

- `along` and `transverse` use the fitted row axis,
- `spacing along` is based on the order of points projected along that axis,
- `nm` modes require valid image calibration.

### 8. Inspect global row geometry

The `Global rows plot` shows all saved points from the active image family in a common scatter plot.

Available units:

- `px`
- `nm`

This is useful for checking:

- relative row placement,
- large-scale distortions,
- consistency across image variants.

### 9. Inspect geometry metrics and disturbance candidates

`Row geometry metrics` shows summary values for the active row, including:

- number of points,
- number of fitted points,
- number of along-axis segments,
- axis angle,
- RMS transverse deviation,
- mean step along the row,
- standard deviation of step along the row.

`Row disturbance candidates` shows a compact summary of local anomalies derived from the fitted row geometry.

Current disturbance markers include:

- local jump in spacing,
- local jump in transverse deviation,
- local change in direction.

These are heuristic candidate markers. They are intended as a first-pass aid for identifying suspicious local changes, not as a final domain-wall classifier.

The active row axis is also drawn on the STM image. Candidate disturbance points are marked directly on the image overlay.

## Units: px and nm

AtomMapper reads scan size from STM metadata and computes:

- `pixel_size_nm_x = size_nm_x / pixels_x`
- `pixel_size_nm_y = size_nm_y / pixels_y`

As a result:

- saved points can carry both `x_px/y_px` and `x_nm/y_nm`,
- row plots and metrics can be shown in `px` or `nm`,
- `nm` views are unavailable if the source image lacks valid physical calibration.

## Saved points table

The `Saved points` table shows saved points for the active image family.

Current columns:

- row
- index
- `x_px`
- `y_px`
- `sigma_x`
- `sigma_y`
- status

Point status can be:

- fit-based
- ROI fallback
- manual override after drag

## CSV export

Use `Export CSV` to save points from the active image family.

The exported CSV includes:

- image and variant identifiers
- row and point identifiers
- coordinates in `px`
- coordinates in `nm` when available
- fit parameters
- manual override flags
- point status

## Session save / load

Use:

- `Save Session`
- `Load Session`

The session file stores the current AtomMapper project state, including:

- loaded images and variants
- active image
- ROI state
- rows
- points
- active row
- active point
- plot/view preferences

After loading a session, AtomMapper restores:

- the table,
- image overlays,
- plots,
- geometry metrics,
- disturbance panel,
- selected state.

## Practical guidance

- Start with the original image, then create preprocessing variants only when needed.
- Keep rows logically separated. One row should represent one physical atomic row.
- Use the Gaussian fit as the default localization method, but correct points manually when the local shape is distorted.
- When comparing original and preprocessed variants, keep the same row active and continue collecting points into that same row family.
- Use `nm` views only when calibration is present and physically meaningful for the loaded data.

## Current scope and limitations

AtomMapper already supports row geometry analysis and disturbance candidates, but it does not yet provide:

- a full domain-wall classifier,
- uncertainty propagation for fitted row geometry,
- publication-ready report generation,
- multi-session comparison workflows.

The disturbance panel should be treated as a guided screening tool, not a final physical decision layer.

## Summary

AtomMapper is a focused STM annotation and geometry-analysis tool for:

- localizing atoms,
- organizing them into rows,
- refining positions manually,
- analyzing row geometry in `px` and `nm`,
- screening for local geometric disturbances,
- exporting and preserving the full workflow.
