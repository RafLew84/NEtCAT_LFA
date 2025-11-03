from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ...analysis.lattice import create_ase_supercell_from_2d_vectors

try:  # pragma: no cover
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter

    PYVISTA_AVAILABLE = True
except ImportError:  # pragma: no cover
    pv = None
    BackgroundPlotter = None
    PYVISTA_AVAILABLE = False


@dataclass(frozen=True)
class RealSpaceSceneConfig:
    """Container describing the 3D scene requested by the visualizer dialog."""

    supercell_size: int
    align_adsorbate: bool
    alignment_angle_rad: float
    substrate_params: Optional[Dict[str, object]]
    adsorbate_params: Optional[Dict[str, object]]
    substrate_offset_nm: Tuple[float, float]
    adsorbate_offset_nm: Tuple[float, float]
    substrate_symbol: str = "Au"
    adsorbate_symbol: str = "I"
    window_title: str = "Interactive 3D Lattice Viewer"


class RealSpacePyVistaAdapter:
    """Manages the optional PyVista background plotter for the visualizer dialog."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._plotter: Optional[BackgroundPlotter] = None
        if PYVISTA_AVAILABLE:  # pragma: no branch
            pv.set_plot_theme("document")

    # ---------------------------------------------------------------------#
    def is_available(self) -> bool:
        return PYVISTA_AVAILABLE

    def is_open(self) -> bool:
        return bool(
            PYVISTA_AVAILABLE
            and self._plotter is not None
            and not getattr(self._plotter, "_closed", False)
        )

    def close(self) -> None:
        if self._plotter is not None:
            try:
                self._plotter.close()  # pragma: no cover
            except Exception:  # pragma: no cover
                self._logger.exception("Failed to close PyVista plotter cleanly.")
            finally:
                self._plotter = None

    # ---------------------------------------------------------------------#
    def refresh_scene(self, config: RealSpaceSceneConfig) -> None:
        """Ensure the BackgroundPlotter is open and populated according to config."""
        if not PYVISTA_AVAILABLE:  # pragma: no cover
            raise RuntimeError("PyVista is not available in this environment.")

        plotter = self._ensure_plotter(config.window_title)
        plotter.clear()
        plotter.set_background("white")
        plotter.remove_all_lights()
        plotter.add_light(pv.Light(position=(5, 5, 10), intensity=1.5))
        plotter.add_light(pv.Light(position=(-5, -5, 5), intensity=0.5))

        size_tuple = (int(config.supercell_size), int(config.supercell_size))
        rotation_matrix = _build_rotation_matrix(config.align_adsorbate, config.alignment_angle_rad)

        self._draw_substrate(plotter, config, size_tuple)
        self._draw_adsorbate(plotter, config, size_tuple, rotation_matrix)

        plotter.camera_position = "xy"
        plotter.reset_camera()
        plotter.app_window.show()  # pragma: no cover

    # ------------------------------------------------------------------ priv
    def _ensure_plotter(self, title: str) -> BackgroundPlotter:
        if self._plotter is None or getattr(self._plotter, "_closed", False):
            self._logger.info("Creating a new PyVista BackgroundPlotter.")
            self._plotter = BackgroundPlotter(show=True, title=title)  # pragma: no cover
        else:
            self._logger.info("Reusing existing PyVista BackgroundPlotter.")
        return self._plotter

    def _draw_substrate(
        self,
        plotter: BackgroundPlotter,
        config: RealSpaceSceneConfig,
        size_tuple: Tuple[int, int],
    ) -> None:
        params = config.substrate_params
        if not params or "a1_vec_nm" not in params or "a2_vec_nm" not in params:
            return

        atoms = create_ase_supercell_from_2d_vectors(
            a1_vec_nm=np.array(params["a1_vec_nm"], dtype=float),
            a2_vec_nm=np.array(params["a2_vec_nm"], dtype=float),
            atom_symbol=config.substrate_symbol,
            size=size_tuple,
            offset_fractional=(0.0, 0.0),
            z_height_nm=1.0,
        )
        if atoms is None:
            return

        offset_nm = np.asarray(config.substrate_offset_nm, dtype=float)
        if np.any(offset_nm):
            atoms.positions[:, 0] += offset_nm[0]
            atoms.positions[:, 1] += offset_nm[1]

        for atom in atoms:
            sphere = pv.Sphere(center=atom.position, radius=0.07)
            sphere.compute_normals(inplace=True)
            actor = plotter.add_mesh(sphere, color="gold", smooth_shading=True)
            actor.prop.metallic = 0.8  # pragma: no cover
            actor.prop.roughness = 0.2  # pragma: no cover

    def _draw_adsorbate(
        self,
        plotter: BackgroundPlotter,
        config: RealSpaceSceneConfig,
        size_tuple: Tuple[int, int],
        rotation_matrix: Optional[np.ndarray],
    ) -> None:
        params = config.adsorbate_params
        if not params or "a1_vec_nm" not in params or "a2_vec_nm" not in params:
            return

        a1_vec_nm = np.array(params["a1_vec_nm"], dtype=float)
        a2_vec_nm = np.array(params["a2_vec_nm"], dtype=float)

        if rotation_matrix is not None:
            a1_vec_nm = rotation_matrix.dot(a1_vec_nm)
            a2_vec_nm = rotation_matrix.dot(a2_vec_nm)

        atoms = create_ase_supercell_from_2d_vectors(
            a1_vec_nm=a1_vec_nm,
            a2_vec_nm=a2_vec_nm,
            atom_symbol=config.adsorbate_symbol,
            size=size_tuple,
            offset_fractional=(0.0, 0.0),
            z_height_nm=1.15,
        )
        if atoms is None:
            return

        offset_nm = self._compute_adsorbate_offset(config, rotation_matrix)
        if np.any(offset_nm):
            atoms.positions[:, 0] += offset_nm[0]
            atoms.positions[:, 1] += offset_nm[1]

        for atom in atoms:
            sphere = pv.Sphere(center=atom.position, radius=0.1)
            sphere.compute_normals(inplace=True)
            actor = plotter.add_mesh(sphere, color="purple", smooth_shading=True)
            actor.prop.metallic = 0.2  # pragma: no cover
            actor.prop.roughness = 0.6  # pragma: no cover

    def _compute_adsorbate_offset(
        self,
        config: RealSpaceSceneConfig,
        rotation_matrix: Optional[np.ndarray],
    ) -> np.ndarray:
        default_offset = np.zeros(2, dtype=float)
        params = config.substrate_params
        if params and "a1_vec_nm" in params and "a2_vec_nm" in params:
            sub_a1 = np.array(params["a1_vec_nm"], dtype=float)
            sub_a2 = np.array(params["a2_vec_nm"], dtype=float)
            fractional = np.array([0.0, 1.0 / 3.0], dtype=float)
            default_offset = fractional[0] * sub_a1 + fractional[1] * sub_a2
            if rotation_matrix is not None:
                default_offset = rotation_matrix.dot(default_offset)

        user_offset = np.asarray(config.adsorbate_offset_nm, dtype=float)
        return default_offset + user_offset


def _build_rotation_matrix(align_adsorbate: bool, alignment_angle_rad: float) -> Optional[np.ndarray]:
    if not align_adsorbate:
        return None
    return np.array(
        [
            [np.cos(alignment_angle_rad), -np.sin(alignment_angle_rad)],
            [np.sin(alignment_angle_rad), np.cos(alignment_angle_rad)],
        ],
        dtype=float,
    )
