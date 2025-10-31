# Lattice Fourier Analyzer - Uncertainty Propagation

This note summarises how **Lattice Fourier Analyzer (LFA)** estimates and propagates uncertainties. We follow the processing pipeline: peak localisation in FFT space, affine drift correction, conversion to physical units, real-space lattice reconstruction, and auxiliary metrics such as the substrate-versus-adsorbate angle. References to the underlying numerical techniques are provided at the end of the document.

---

## 1. Peak localisation in FFT space

Let the FFT image be a discrete grid of intensities `I(kx, ky)`. Depending on the refinement method we obtain sub-pixel coordinates `(kx, ky)` and their uncertainties as follows.

### 1.1 `REFINEMENT_DIRECT_CLICK`
The user selects a point manually; we therefore record no covariance (`None`). If an error budget is needed, a conservative default is `sigma = 0.5` px (variance `0.25` px^2), but LFA leaves the value unspecified.

### 1.2 `REFINEMENT_MAX_PIXEL`
We locate the brightest pixel in the ROI and add `(0.5, 0.5)` so that the reported position represents the pixel centre. The covariance is set to `None`; users may substitute a generic `sigma` later (for a uniform pixel, `sigma = 0.29` px).

### 1.3 `REFINEMENT_GAUSSIAN_FIT`
Inside the ROI we fit a two-dimensional Gaussian

```
I(kx, ky) = A * exp(-0.5 * p^T * C^-1 * p) + B,  p = [kx - mux, ky - muy]^T,
```

using `scipy.optimize.curve_fit`. The returned parameter covariance `Sigma_theta` contains the variances for `mux` and `muy`, which we store as `sigma_x^2` and `sigma_y^2`. If the fit fails, a Monte Carlo fallback synthesises noisy ROIs (based on the estimated background noise) and estimates the spread of `(mux, muy)`.

### 1.4 `REFINEMENT_PARABOLA_3X3`
We approximate the peak with a quadratic Taylor expansion on a `3x3` neighbourhood and solve `grad I = 0`. The helper returns the sub-pixel maximum along with uncertainties derived from the local Hessian of the quadratic surface.

### 1.5 `REFINEMENT_LOCAL_DFT`
A square ROI of size `m x n` is upsampled by zero-padding its discrete Fourier transform:

1. Compute `F = FFT(ROI)` and centre it with `fftshift`.
2. Embed `F` into a zero matrix of size `(m*s) x (n*s)` where `s` is the upsampling factor (default 8). This corresponds to sinc interpolation of the spectrum [1].
3. Perform the inverse FFT; the resulting magnitude map has a spacing of `1/s` pixels.
4. The sub-pixel location equals `y_start + j/s`, `x_start + i/s`, where `(j, i)` is the index of the maximum in the upsampled patch.

Uncertainties are estimated by Monte Carlo: the ROI is perturbed with Gaussian noise of standard deviation `_estimate_noise_sigma`, the local DFT refinement is repeated, and the empirical standard deviations from typically 128 runs deliver `sigma_x` and `sigma_y`.

### 1.6 Normalisation
Whenever `sigma_x` and `sigma_y` are known we store the diagonal covariance (note the internal `(ky, kx)` ordering)

```
Sigma_spot = [[sigma_y^2, 0], [0, sigma_x^2]].
```

If a method does not report uncertainties we keep `None`. Downstream components either ignore such points or plug in conservative defaults configurable by the user.

---

## 2. Affine transform fitting (substrate)

After substrate peaks are selected, `match_and_fit_transform` finds an affine transform `(F, t)` mapping measured FFT coordinates to the ideal lattice.

1. **Assignment** – the Hungarian algorithm links measured peaks with their ideal counterparts; each measured peak retains its covariance `Sigma_spot`.
2. **Propagation** – affine transforms are linear, hence `Sigma_ideal = F * Sigma_spot * F^T`.
3. **Transform analysis** – `fit_affine_measured_to_ideal` returns the covariance of the affine parameters. Using the polar decomposition `F = R * U`, we propagate this covariance to obtain the variance of the rotation angle as well as the variances of the principal stretches (singular values of `U`).

The application state records `matched_measured_covariances_px`, `fitted_spot_covariances_px`, `transform_analysis["rotation_angle_deg_sigma"]`, and `transform_analysis["principal_stretches_sigma"]`.

---

## 3. Adsorbate correction

`AdsorbateSpotPresenter.apply_substrate_correction` maps each adsorbate spot to the ideal frame using the substrate matrix `F`. For every raw covariance `Sigma_raw` we compute `Sigma_corr = F * Sigma_raw * F^T`. The UI therefore displays corrected coordinates together with their propagated uncertainties.

---

## 4. Conversion to reciprocal-space units

A displacement in pixels `g_px` (relative to the FFT centre) becomes

```
gx_nm^-1 = g_px.x / Lx,
gy_nm^-1 = g_px.y / Ly,
```

so the covariance scales as

```
Sigma_g_nm^-1 = diag(1/Ly, 1/Lx) * Sigma_g_px * diag(1/Ly, 1/Lx).
```

### Pixel-calibration uncertainty
If the pixel calibration is uncertain by `sigma_Lx` and `sigma_Ly` (set via `size_nm_x_sigma` / `size_nm_y_sigma`), `augment_covariance_with_calibration` adds the additional terms

```
Var(gx) += (gx / Lx)^2 * sigma_Lx^2,
Var(gy) += (gy / Ly)^2 * sigma_Ly^2.
```

This allows downstream quantities to reflect calibration uncertainty.

---

## 5. Real-space lattice parameters

Given two reciprocal vectors `(g1, g2)` in nm^-1 we invert the 2x2 matrix `G = [g1; g2]` to recover the direct lattice (`calculate_real_space_vectors_from_g`).

We build a 4x4 covariance for `(g1x, g1y, g2x, g2y)` and propagate it through the mapping

```
f(g) = [ |a1(g)|, |a2(g)|, alpha(g) ]^T,
```

where `alpha` is the angle between `a1` and `a2`. `compute_real_space_metric_uncertainty` evaluates the Jacobian and computes `Sigma_metrics = J_f * Sigma_g * J_f^T`. If the linear propagation becomes unstable (e.g. near-singular Jacobian), a Monte Carlo fallback (1024 samples by default) is used. The reported standard deviations are the square roots of the diagonal entries of `Sigma_metrics`.

---

## 6. Transform-analysis uncertainties

### 6.1 Rotation and principal stretches
The covariance of the affine parameters is propagated through the polar decomposition `F = R * U`. Gradients of the rotation angle and of the eigenvalues of `U` with respect to the matrix entries deliver `sigma_theta`, `sigma_lambda1`, and `sigma_lambda2` (see [2]).

### 6.2 Corrected adsorbate spots
Each corrected point inherits `F * Sigma_raw * F^T`. Consequently the "Corrected Adsorbate Spots" widget shows `(x +- sigma_x, y +- sigma_y)`.

### 6.3 Substrate-versus-adsorbate angle
The button **Calculate Sub-Ads Angle** evaluates

```
phi = arccos( (a1_S dot a1_A) / (|a1_S| * |a1_A|) ).
```

To estimate `sigma_phi` we perform Monte Carlo sampling (512 samples by default): draw `(g1, g2)` from their Gaussian distributions, convert them to `(a1, a2)`, compute `phi` for each sample, and report the sample standard deviation.

---

## 7. Overview of reported uncertainties

| Stage                              | Quantity                          | Uncertainty source                               |
|------------------------------------|-----------------------------------|--------------------------------------------------|
| Peak localisation (Gaussian)       | `(kx, ky)`                        | Curve-fit covariance / Monte Carlo fallback      |
| Peak localisation (Parabola, LDFT) | `(kx, ky)`                        | Returned by refinement helper                    |
| Affine transform                   | Rotation, principal stretches     | Propagation from affine-parameter covariance     |
| Corrected adsorbate spots          | `(x, y)` in ideal frame           | `F * Sigma_raw * F^T`                            |
| Reciprocal vectors                 | `g1, g2`                          | Pixel scaling plus calibration uncertainty       |
| Real-space metrics                 | `|a1|, |a2|, alpha`               | Jacobian propagation with Monte Carlo fallback   |
| Substrate-versus-adsorbate angle   | `phi`                             | Monte Carlo on reciprocal-vector distributions   |

---

## 8. Practical notes

1. **Pixel calibration** – provide realistic `size_nm_x_sigma` / `size_nm_y_sigma`; otherwise lattice lengths appear over-optimistic.
2. **ROI tuning** – larger ROIs for Gaussian or LDFT fits reduce `sigma`, but avoid including neighbouring peaks.
3. **Strain diagnosis** – if adsorbate vectors differ by e.g. 0.06 nm while `sigma` is about 0.001 nm, treat the 0.06 nm as a physical deformation (non-linear drift or strain) rather than noise.
4. **Transform quality** – monitor RMSE and matched pairs; large residuals suggest revisiting the chosen substrate peaks.
5. **Fallback semantics** – when an algorithm cannot provide `sigma`, LFA keeps `None`. Downstream code either drops such entries or replaces them with user-specified defaults.
6. **Monte Carlo samples** – the defaults (128, 512, 1024) balance accuracy and runtime; noisy data may require increasing these counts.
7. **Coordinate order** – internal covariances follow the `(ky, kx)` convention; convert carefully when exporting to external tools.

---

## 9. Summary

LFA propagates uncertainties throughout the full analysis chain: sub-pixel FFT fits, affine drift correction, conversion to physical units, real-space metrics, and cross-lattice angles. Linear covariance propagation is used whenever the mapping is analytic; Monte Carlo fallbacks cover non-linear steps. Consequently the entries displayed as `value +- sigma` in the UI and exports reflect both fit precision and calibration or transform uncertainties.

---

## References

[1] M. Guizar-Sicairos, S. T. Thurman, and J. R. Fienup, "Efficient subpixel image registration algorithms", *Optics Letters* 33(2), 156-158 (2008).

[2] F. L. Teixeira and W. C. Chew, "Matrix derivative calculations for affine transformations", *IEEE Transactions on Antennas and Propagation* 52(11), 3131-3138 (2004).
