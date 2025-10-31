# Lattice Fourier Analyzer — Uncertainty Propagation

This note summarizes how **Lattice Fourier Analyzer (LFA)** estimates and propagates uncertainties.  We follow the processing pipeline: peak localisation in FFT space, affine drift correction, conversion to physical units, real-space lattice reconstruction, and auxiliary metrics such as the substrate–adsorbate angle.

---

## 1. Peak localisation in FFT space

Let the FFT image be a discrete grid of intensities \(I(k_x, k_y)\). Depending on the refinement method we obtain sub-pixel coordinates \((k_x, k_y)\) and their uncertainties as follows.

### 1.1 `REFINEMENT_DIRECT_CLICK`
The user clicks a point manually. We record no covariance (`None`). If an error budget is later required, one may assume a generic \(\pm 0.5\) pixel uncertainty, but LFA treats this method as deterministic.

### 1.2 `REFINEMENT_MAX_PIXEL`
We take the brightest pixel in the ROI and shift by \((0.5, 0.5)\) to reach the pixel centre. The covariance is again left undefined (`None`). A typical fallback is \(\sigma \approx 0.29\) px (the standard deviation of a uniform pixel).

### 1.3 `REFINEMENT_GAUSSIAN_FIT`
Inside the ROI we fit a two-dimensional Gaussian

\[
I(k_x, k_y) = A \exp\!\bigl(-\tfrac{1}{2}(p^\top C^{-1} p)\bigr) + B,\qquad
p = \begin{bmatrix} k_x - \mu_x \\[0.2em] k_y - \mu_y \end{bmatrix}.
\]

`scipy.optimize.curve_fit` returns the parameter covariance \(\Sigma_\theta\). The diagonal entries corresponding to \(\mu_x, \mu_y\) give \(\sigma_x^2, \sigma_y^2\). If the fit fails, a Monte Carlo fallback synthesises noisy ROIs and estimates the spread of \((\mu_x, \mu_y)\).

### 1.4 `REFINEMENT_PARABOLA_3X3`
We approximate the peak with a quadratic Taylor expansion on a \(3\times 3\) neighbourhood and solve \(\nabla I = 0\). The helper returns the sub-pixel maximum and the uncertainties derived from the Hessian.

### 1.5 `REFINEMENT_LOCAL_DFT`
For a square ROI of size \(m\times n\) centred on the coarse peak we compute a *locally upsampled* spectrum by zero-padding the discrete Fourier transform of the ROI.  In code:

1. Compute \(F = \mathrm{FFT}(I_{\text{ROI}})\) and centre it with `fftshift`.
2. Embed \(F\) into a zero matrix of size \((m s)\times(n s)\) where \(s\) is the up-sampling factor (default 8).  This corresponds to sinc-interpolation of the spectrum.
3. Apply the inverse FFT to obtain an interpolated magnitude map with pixel spacing \(1/s\).
4. Locate the maximum of the upsampled magnitude.  The sub-pixel coordinates are
   \[
       k_y = y_{\text{start}} + \frac{j}{s},\qquad
       k_x = x_{\text{start}} + \frac{i}{s},
   \]
   where \((j,i)\) is the index of the maximum.

Uncertainties are estimated by Monte Carlo: we perturb the ROI with synthetic noise (using `_estimate_noise_sigma`) and repeat the local DFT refinement (`_monte_carlo_uncertainty`).  The empirical standard deviations of \((k_x, k_y)\) over typically 128 runs give \(\sigma_x, \sigma_y\).

### 1.6 Normalisation
Whenever \(\sigma_x, \sigma_y\) are known we store a diagonal covariance (note the code ordering \((k_y, k_x)\)):

\[
\Sigma_{\text{spot}} =
\begin{bmatrix}
\sigma_y^2 & 0\\
0 & \sigma_x^2
\end{bmatrix}.
\]

When a method does not report uncertainties we keep `None` and allow a higher-level module to substitute defaults if needed.

---

## 2. Affine transform fitting (substrate)

After substrate peaks are selected, `match_and_fit_transform` finds an affine transform \(F, t\) that maps measured FFT coordinates to the ideal lattice pool.

1. **Assignment** — the Hungarian algorithm links measured peaks with ideal ones. Each measured peak keeps its covariance \(\Sigma_{\text{spot}}\).
2. **Propagation** — applying an affine transform is linear, hence the covariance becomes
   \(\Sigma_{\text{ideal}} = F\,\Sigma_{\text{spot}}\,F^\top\).
3. **Transform analysis** — `fit_affine_measured_to_ideal` returns the covariance of affine parameters (\(\Sigma_F\)). We propagate it to
   - rotation-angle variance (via polar decomposition \(F = R U\) and linear propagation),
   - variances of principal stretches (eigenvalues of \(U\)).

The application state stores `matched_measured_covariances_px`, `fitted_spot_covariances_px`, and `transform_analysis["rotation_angle_deg_sigma"]` / `["principal_stretches_sigma"]`.

---

## 3. Adsorbate correction

`AdsorbateSpotPresenter.apply_substrate_correction` maps each adsorbate spot to the ideal frame using the substrate \(F\). For every raw covariance \(\Sigma_{\text{raw}}\) we compute

\[
\Sigma_{\text{corr}} = F\,\Sigma_{\text{raw}}\,F^\top,
\]

and store the result in `state.corrected_spot_covariances`. The UI therefore shows corrected coordinates with uncertainties.

---

## 4. Conversion to reciprocal-space units

A displacement in pixels \(g_{\text{px}}\) is converted to nm\(^{-1}\) using the image size \(L_x, L_y\):

\[
g_x^{[\text{nm}^{-1}]} = \frac{g_x^{[\text{px}]}}{L_x},\qquad
g_y^{[\text{nm}^{-1}]} = \frac{g_y^{[\text{px}]}}{L_y}.
\]

Therefore

\[
\Sigma_{g}^{[\text{nm}^{-1}]}
=
\begin{bmatrix}
1/L_y & 0\\
0 & 1/L_x
\end{bmatrix}
\Sigma_{g}^{[\text{px}]}
\begin{bmatrix}
1/L_y & 0\\
0 & 1/L_x
\end{bmatrix}.
\]

### Pixel-calibration uncertainty
If the calibration itself is uncertain by \(\sigma_{L_x}, \sigma_{L_y}\), `augment_covariance_with_calibration` adds terms

\[
\mathrm{Var}(g_x) \gets \mathrm{Var}(g_x) +
\left(\frac{g_x}{L_x}\right)^2 \sigma_{L_x}^2,
\]

and similarly for \(g_y\). This allows the final uncertainties to include calibration error.

---

## 5. Real-space lattice parameters

Given two reciprocal vectors \(g_1, g_2\) (nm\(^{-1}\)) we recover the direct lattice by inverting the 2×2 matrix of \(g\)-vectors. The helper `calculate_real_space_vectors_from_g` performs this step and returns \(a_1, a_2\).

We assemble a 4×4 covariance for \((g_{1x}, g_{1y}, g_{2x}, g_{2y})\). `compute_real_space_metric_uncertainty` evaluates the Jacobian of

\[
f(g) =
\begin{bmatrix}
\|a_1(g)\| \\
\|a_2(g)\| \\
\alpha(g)
\end{bmatrix},
\]

where \(\alpha\) is the angle between \(a_1\) and \(a_2\). The error propagation is
\(\Sigma_{\text{metrics}} = J_f\,\Sigma_g\,J_f^\top\). When linear propagation fails (e.g. due to near-singular matrices) we fall back to Monte Carlo (1024 samples by default). The reported standard deviations are \(\sqrt{(\Sigma_{\text{metrics}})_{00}}\), \(\sqrt{(\Sigma_{\text{metrics}})_{11}}\), and \(\sqrt{(\Sigma_{\text{metrics}})_{22}}\).

---

## 6. Transform-analysis uncertainties

### 6.1 Rotation and stretch (substrate)
Using the covariance of affine parameters \(\Sigma_F\) we derive the variance of the rotation angle via polar decomposition and linear propagation. Principal stretches (singular values of the deformation part) are treated similarly.

### 6.2 Corrected adsorbate spots
Each corrected point inherits \(F\,\Sigma F^\top\); the UI shows the corresponding \((x \pm \sigma_x, y \pm \sigma_y)\).

### 6.3 Substrate versus adsorbate angle
The button **Calculate Sub-Ads Angle** evaluates
\(\phi = \arccos\!\bigl((a_1^{(S)}\cdot a_1^{(A)})/(\|a_1^{(S)}\|\|a_1^{(A)}\|)\bigr)\).
To estimate \(\sigma_\phi\) the dialog performs Monte Carlo sampling (512 samples by default): it draws reciprocal vectors from their covariance (Gaussian), converts them to real space, computes the angle for each sample, and reports the sample standard deviation.

---

## 7. Overview of reported uncertainties

| Stage                              | Quantity                                  | Uncertainty source                                  |
|------------------------------------|-------------------------------------------|-----------------------------------------------------|
| Peak localisation (Gaussian)       | \((k_x, k_y)\)                            | Diagonal of curve-fit covariance / Monte Carlo      |
| Peak localisation (Parabola, LDFT) | \((k_x, k_y)\)                            | Returned by refinement helper                       |
| Affine transform                   | Rotation angle, principal stretches       | Propagation from \(\Sigma_F\)                       |
| Corrected adsorbate spots          | \((x, y)\) in ideal frame                 | \(F\,\Sigma_{\text{raw}}\,F^\top\)                   |
| Reciprocal vectors                 | \(g_1, g_2\)                              | Pixel-scale and calibration scaling                 |
| Real-space metrics                 | \(|a_1|, |a_2|, \alpha\)                  | Jacobian propagation with Monte Carlo fallback      |
| Substrate–adsorbate angle          | \(\phi\)                                  | Monte Carlo on reciprocal-vector distributions      |

---

## 8. Practical notes

1. **Pixel calibration** — provide realistic `size_nm_x_sigma` and `size_nm_y_sigma`; otherwise lengths appear over-optimistic.
2. **ROI tuning** — larger ROIs for Gaussian or LDFT fits reduce σ but beware of including neighbouring peaks.
3. **Strain diagnosis** — if adsorbate vectors differ by e.g. 0.06 nm while σ ≈ 0.001 nm, treat the 0.06 nm as a real deformation (non-linear drift or actual strain).
4. **Transform quality** — monitor RMSE and matched pairs; large residuals suggest revisiting substrate peaks.

---

## 9. Summary

LFA propagates uncertainties throughout the full analysis chain: sub-pixel FFT fits, affine drift correction, conversion to physical units, real-space metrics, and cross-lattice angles. Linear covariance propagation is used whenever the mapping is analytic; Monte Carlo fallbacks cover non-linear steps. Consequently the `value ± sigma` pairs shown in the UI and exports reflect both fit precision and calibration or transform uncertainties.
