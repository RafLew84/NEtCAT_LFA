from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

try:  # pragma: no cover - purely UI wiring
    import pyqtgraph as pg
    from pyqtgraph import GraphicsLayoutWidget, ImageItem, PlotWidget, ViewBox
except ImportError:  # pragma: no cover - handled at dialog level
    pg = None  # type: ignore
    GraphicsLayoutWidget = None  # type: ignore
    ImageItem = None  # type: ignore
    PlotWidget = None  # type: ignore
    ViewBox = None  # type: ignore


@dataclass
class RealSpaceVisualizerWidgets:
    """Container for the widgets created for the real-space visualizer."""

    main_splitter: QSplitter
    fft_panel_widget: GraphicsLayoutWidget | None
    fft_view_box: ViewBox | None
    fft_image_item_vis: ImageItem | None
    real_space_plot_widget: PlotWidget | None
    real_space_view_box: ViewBox | None
    display_options_form: QFormLayout
    cb_show_substrate_real_lattice: QCheckBox
    adsorbate_display_checkbox_layout: QVBoxLayout
    custom_adsorbate_visibility_checkbox: QCheckBox
    adsorbate_sets_checkbox_layout: QVBoxLayout
    cb_show_g_substrate_fft: QCheckBox
    cb_show_g_adsorbate_fft: QCheckBox
    cb_visual_align: QCheckBox
    substrate_lattice_cells_spin: QSpinBox
    adsorbate_lattice_cells_spin: QSpinBox
    substrate_atom_size_spin: QDoubleSpinBox
    adsorbate_atom_size_spin: QDoubleSpinBox
    adsorbate_marker_alpha_spin: QDoubleSpinBox
    lattice_outline_width_spin: QDoubleSpinBox
    lattice_outline_alpha_spin: QDoubleSpinBox
    supercell_size_spinbox: QSpinBox
    custom_a1_x_spin: QDoubleSpinBox
    custom_a1_y_spin: QDoubleSpinBox
    custom_a2_x_spin: QDoubleSpinBox
    custom_a2_y_spin: QDoubleSpinBox
    custom_offset_x_spin: QDoubleSpinBox
    custom_offset_y_spin: QDoubleSpinBox
    custom_a1_length_spin: QDoubleSpinBox
    custom_a2_length_spin: QDoubleSpinBox
    custom_angle_a1_spin: QDoubleSpinBox
    custom_angle_between_spin: QDoubleSpinBox
    custom_symbol_combo: QComboBox
    custom_length_convert_button: QPushButton
    custom_adsorbate_apply_button: QPushButton
    custom_adsorbate_clear_button: QPushButton
    substrate_offset_x_spin: QDoubleSpinBox
    substrate_offset_y_spin: QDoubleSpinBox
    adsorbate_offset_x_spin: QDoubleSpinBox
    adsorbate_offset_y_spin: QDoubleSpinBox
    launch_3d_button: QPushButton
    info_sub_rot_label: QLabel
    info_sub_scale_label: QLabel
    info_sub_rmse_label: QLabel
    sub_real_a1_label: QLabel
    sub_real_a2_label: QLabel
    sub_real_alpha_label: QLabel
    calibration_sigma_label: QLabel
    ads_set_combo_vis: QComboBox
    ads_real_a1_label: QLabel
    ads_real_a2_label: QLabel
    ads_real_alpha_label: QLabel
    angle_sub_ads_label: QLabel
    calculate_sub_ads_angle_button: QPushButton
    close_button: QPushButton
    extra_references: Dict[str, QWidget | QLayout] = field(default_factory=dict)


def build_real_space_visualizer_ui(
    dialog: QWidget,
    *,
    on_real_space_view_range_changed: Callable[[ViewBox, Dict[str, float]], None] | None = None,
) -> RealSpaceVisualizerWidgets:
    """
    Construct the UI for the real-space visualizer dialog.

    The function intentionally keeps all widget creation in one place so the main dialog
    can focus on wiring signals and business logic. It sets no domain-specific state.
    """

    top_level_layout = QHBoxLayout(dialog)
    main_splitter = QSplitter(Qt.Orientation.Horizontal)
    top_level_layout.addWidget(main_splitter)

    fft_panel_widget = GraphicsLayoutWidget() if GraphicsLayoutWidget else None
    fft_view_box = None
    fft_image_item_vis = None
    if fft_panel_widget:
        fft_view_box = fft_panel_widget.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        fft_image_item_vis = ImageItem() if ImageItem else None
        if fft_image_item_vis and fft_view_box:
            fft_view_box.addItem(fft_image_item_vis)
            fft_view_box.setMenuEnabled(True)
            fft_view_box.setMouseMode(ViewBox.PanMode)
        main_splitter.addWidget(fft_panel_widget)

    real_space_plot_widget = PlotWidget() if PlotWidget else None
    real_space_view_box = None
    if real_space_plot_widget:
        real_space_view_box = real_space_plot_widget.getViewBox()
        if real_space_view_box and on_real_space_view_range_changed:
            real_space_view_box.sigRangeChanged.connect(on_real_space_view_range_changed)
        plot_item_rs = real_space_plot_widget.getPlotItem()
        if plot_item_rs:
            plot_item_rs.setAspectLocked(True)
            plot_item_rs.setTitle("Real Space Lattice Visualization")
            plot_item_rs.setLabel("left", "Y (nm)")
            plot_item_rs.setLabel("bottom", "X (nm)")
            plot_item_rs.showGrid(x=True, y=True, alpha=0.3)
        main_splitter.addWidget(real_space_plot_widget)

    controls_panel_widget = QWidget()
    controls_panel_layout = QVBoxLayout(controls_panel_widget)
    controls_panel_widget.setMinimumWidth(350)
    controls_panel_widget.setMaximumWidth(450)

    display_options_group = QGroupBox("Display Options")
    group_box_layout = QVBoxLayout(display_options_group)

    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    scroll_content_widget = QWidget()
    display_options_form = QFormLayout(scroll_content_widget)

    cb_show_substrate_real_lattice = QCheckBox("Substrate Real Lattice")
    cb_show_substrate_real_lattice.setChecked(True)
    display_options_form.addRow(cb_show_substrate_real_lattice)

    adsorbate_display_checkbox_layout = QVBoxLayout()
    display_options_form.addRow(QLabel("Adsorbate Sets (Real Space):"))
    display_options_form.addRow(adsorbate_display_checkbox_layout)

    custom_adsorbate_visibility_checkbox = QCheckBox("Custom Adsorbate Lattice")
    custom_adsorbate_visibility_checkbox.setChecked(False)
    adsorbate_display_checkbox_layout.addWidget(custom_adsorbate_visibility_checkbox)

    adsorbate_sets_checkbox_layout = QVBoxLayout()
    adsorbate_display_checkbox_layout.addLayout(adsorbate_sets_checkbox_layout)

    cb_show_g_substrate_fft = QCheckBox("Substrate g* vectors (on FFT)")
    cb_show_g_substrate_fft.setChecked(True)
    display_options_form.addRow(cb_show_g_substrate_fft)

    cb_show_g_adsorbate_fft = QCheckBox("Adsorbate g* vectors (Current Set, on FFT)")
    cb_show_g_adsorbate_fft.setChecked(True)
    display_options_form.addRow(cb_show_g_adsorbate_fft)

    cb_visual_align = QCheckBox("Visually align adsorbate lattice to substrate")
    cb_visual_align.setChecked(False)
    cb_visual_align.setToolTip(
        "Rotates only the visualization so the adsorbate a1 vector matches the substrate a1 vector."
    )
    display_options_form.addRow(cb_visual_align)

    substrate_lattice_cells_spin = QSpinBox()
    substrate_lattice_cells_spin.setRange(1, 50)
    substrate_lattice_cells_spin.setValue(10)
    substrate_lattice_cells_spin.setToolTip("Number of substrate lattice cells drawn in each direction (NxN).")
    display_options_form.addRow("Substrate lattice span (N):", substrate_lattice_cells_spin)

    adsorbate_lattice_cells_spin = QSpinBox()
    adsorbate_lattice_cells_spin.setRange(1, 50)
    adsorbate_lattice_cells_spin.setValue(10)
    adsorbate_lattice_cells_spin.setToolTip("Number of adsorbate lattice cells drawn in each direction (NxN).")
    display_options_form.addRow("Adsorbate lattice span (N):", adsorbate_lattice_cells_spin)

    substrate_atom_size_spin = QDoubleSpinBox()
    substrate_atom_size_spin.setRange(1.0, 30.0)
    substrate_atom_size_spin.setSingleStep(0.5)
    substrate_atom_size_spin.setValue(8.0)
    substrate_atom_size_spin.setToolTip("Marker size used for substrate lattice points.")
    display_options_form.addRow("Substrate atom size:", substrate_atom_size_spin)

    adsorbate_atom_size_spin = QDoubleSpinBox()
    adsorbate_atom_size_spin.setRange(1.0, 30.0)
    adsorbate_atom_size_spin.setSingleStep(0.5)
    adsorbate_atom_size_spin.setValue(8.0)
    adsorbate_atom_size_spin.setToolTip("Marker size used for adsorbate lattice points.")
    display_options_form.addRow("Adsorbate atom size:", adsorbate_atom_size_spin)

    adsorbate_marker_alpha_spin = QDoubleSpinBox()
    adsorbate_marker_alpha_spin.setRange(0.05, 1.0)
    adsorbate_marker_alpha_spin.setSingleStep(0.05)
    adsorbate_marker_alpha_spin.setValue(0.6)
    adsorbate_marker_alpha_spin.setToolTip("Alpha transparency for adsorbate lattice markers.")
    display_options_form.addRow("Adsorbate marker alpha:", adsorbate_marker_alpha_spin)

    lattice_outline_width_spin = QDoubleSpinBox()
    lattice_outline_width_spin.setRange(0.1, 10.0)
    lattice_outline_width_spin.setSingleStep(0.1)
    lattice_outline_width_spin.setValue(1.5)
    display_options_form.addRow("Lattice outline width:", lattice_outline_width_spin)

    lattice_outline_alpha_spin = QDoubleSpinBox()
    lattice_outline_alpha_spin.setRange(0.05, 1.0)
    lattice_outline_alpha_spin.setSingleStep(0.05)
    lattice_outline_alpha_spin.setValue(0.75)
    display_options_form.addRow("Lattice outline alpha:", lattice_outline_alpha_spin)

    custom_group = QGroupBox("Custom Adsorbate Lattice Definition")
    custom_form = QFormLayout(custom_group)

    def _mk_vec_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-100.0, 100.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.05)
        return spin

    custom_a1_x_spin = _mk_vec_spin()
    custom_form.addRow("a1 X (nm):", custom_a1_x_spin)
    custom_a1_y_spin = _mk_vec_spin()
    custom_form.addRow("a1 Y (nm):", custom_a1_y_spin)
    custom_a2_x_spin = _mk_vec_spin()
    custom_form.addRow("a2 X (nm):", custom_a2_x_spin)
    custom_a2_y_spin = _mk_vec_spin()
    custom_form.addRow("a2 Y (nm):", custom_a2_y_spin)

    custom_offset_x_spin = _mk_vec_spin()
    custom_form.addRow("Offset X (nm):", custom_offset_x_spin)
    custom_offset_y_spin = _mk_vec_spin()
    custom_form.addRow("Offset Y (nm):", custom_offset_y_spin)

    custom_a1_length_spin = QDoubleSpinBox()
    custom_a1_length_spin.setRange(0.01, 100.0)
    custom_a1_length_spin.setDecimals(3)
    custom_a1_length_spin.setSingleStep(0.05)
    custom_a1_length_spin.setValue(1.0)
    custom_form.addRow("|a1| (nm):", custom_a1_length_spin)

    custom_a2_length_spin = QDoubleSpinBox()
    custom_a2_length_spin.setRange(0.01, 100.0)
    custom_a2_length_spin.setDecimals(3)
    custom_a2_length_spin.setSingleStep(0.05)
    custom_a2_length_spin.setValue(1.0)
    custom_form.addRow("|a2| (nm):", custom_a2_length_spin)

    custom_angle_a1_spin = QDoubleSpinBox()
    custom_angle_a1_spin.setRange(-360.0, 360.0)
    custom_angle_a1_spin.setDecimals(2)
    custom_angle_a1_spin.setSingleStep(1.0)
    custom_angle_a1_spin.setValue(0.0)
    custom_form.addRow("Angle of a1 (deg):", custom_angle_a1_spin)

    custom_angle_between_spin = QDoubleSpinBox()
    custom_angle_between_spin.setRange(0.0, 180.0)
    custom_angle_between_spin.setDecimals(2)
    custom_angle_between_spin.setSingleStep(1.0)
    custom_angle_between_spin.setValue(60.0)
    custom_form.addRow("Angle between a1 & a2 (deg):", custom_angle_between_spin)

    custom_length_convert_button = QPushButton("Convert Lengths/Angles to Components")
    custom_form.addRow(custom_length_convert_button)

    custom_symbol_combo = QComboBox()
    custom_symbol_combo.addItems(["star", "o", "s", "t", "d", "+", "x"])
    custom_form.addRow("Marker symbol:", custom_symbol_combo)

    apply_clear_layout = QHBoxLayout()
    custom_adsorbate_apply_button = QPushButton("Apply Custom Lattice")
    custom_adsorbate_clear_button = QPushButton("Clear Custom Lattice")
    apply_clear_layout.addWidget(custom_adsorbate_apply_button)
    apply_clear_layout.addWidget(custom_adsorbate_clear_button)
    custom_form.addRow(apply_clear_layout)

    display_options_form.addRow(custom_group)

    supercell_size_spinbox = QSpinBox()
    supercell_size_spinbox.setMinimum(1)
    supercell_size_spinbox.setMaximum(50)
    supercell_size_spinbox.setValue(5)
    supercell_size_spinbox.setToolTip("Sets the NxN size of the supercell for 3D visualization.")
    display_options_form.addRow("3D Supercell Size (NxN):", supercell_size_spinbox)

    offsets_group = QGroupBox("3D Layer Offsets (nm)")
    offsets_form = QFormLayout(offsets_group)

    substrate_offset_x_spin = QDoubleSpinBox()
    substrate_offset_x_spin.setRange(-100.0, 100.0)
    substrate_offset_x_spin.setDecimals(3)
    substrate_offset_x_spin.setSingleStep(0.05)
    offsets_form.addRow("Substrate DeltaX:", substrate_offset_x_spin)

    substrate_offset_y_spin = QDoubleSpinBox()
    substrate_offset_y_spin.setRange(-100.0, 100.0)
    substrate_offset_y_spin.setDecimals(3)
    substrate_offset_y_spin.setSingleStep(0.05)
    offsets_form.addRow("Substrate DeltaY:", substrate_offset_y_spin)

    adsorbate_offset_x_spin = QDoubleSpinBox()
    adsorbate_offset_x_spin.setRange(-100.0, 100.0)
    adsorbate_offset_x_spin.setDecimals(3)
    adsorbate_offset_x_spin.setSingleStep(0.05)
    offsets_form.addRow("Adsorbate DeltaX:", adsorbate_offset_x_spin)

    adsorbate_offset_y_spin = QDoubleSpinBox()
    adsorbate_offset_y_spin.setRange(-100.0, 100.0)
    adsorbate_offset_y_spin.setDecimals(3)
    adsorbate_offset_y_spin.setSingleStep(0.05)
    offsets_form.addRow("Adsorbate DeltaY:", adsorbate_offset_y_spin)

    display_options_form.addRow(offsets_group)

    launch_3d_button = QPushButton("Launch Interactive 3D Viewer")
    launch_3d_button.setToolTip("Opens a new, interactive window with a 3D model of the lattices.")
    display_options_form.addRow(launch_3d_button)

    scroll_area.setWidget(scroll_content_widget)
    group_box_layout.addWidget(scroll_area)

    controls_panel_layout.addWidget(display_options_group)

    transform_info_group = QGroupBox("Substrate Transformation Info")
    transform_info_layout = QFormLayout(transform_info_group)
    info_sub_rot_label = QLabel("-")
    info_sub_scale_label = QLabel("-")
    info_sub_rmse_label = QLabel("-")
    transform_info_layout.addRow("Rot (M->I):", info_sub_rot_label)
    transform_info_layout.addRow("Stretch (M->I):", info_sub_scale_label)
    transform_info_layout.addRow("Fit RMSE (M->I, px):", info_sub_rmse_label)
    controls_panel_layout.addWidget(transform_info_group)

    sub_real_params_group = QGroupBox("Substrate Real Space Parameters")
    sub_real_params_layout = QFormLayout(sub_real_params_group)
    sub_real_a1_label = QLabel("- nm")
    sub_real_a2_label = QLabel("- nm")
    sub_real_alpha_label = QLabel("- deg")
    calibration_sigma_label = QLabel("- nm")
    sub_real_params_layout.addRow("|a1|:", sub_real_a1_label)
    sub_real_params_layout.addRow("|a2|:", sub_real_a2_label)
    sub_real_params_layout.addRow("Angle (a1,a2) [deg]:", sub_real_alpha_label)
    sub_real_params_layout.addRow("Pixel sigma (x,y):", calibration_sigma_label)
    controls_panel_layout.addWidget(sub_real_params_group)

    ads_real_params_group = QGroupBox("Adsorbate Real Space Parameters")
    ads_real_params_layout = QFormLayout(ads_real_params_group)
    ads_set_combo_vis = QComboBox()
    ads_real_params_layout.addRow("Select Adsorbate Set:", ads_set_combo_vis)
    ads_real_a1_label = QLabel("- nm")
    ads_real_a2_label = QLabel("- nm")
    ads_real_alpha_label = QLabel("- deg")
    ads_real_params_layout.addRow("|a1|:", ads_real_a1_label)
    ads_real_params_layout.addRow("|a2|:", ads_real_a2_label)
    ads_real_params_layout.addRow("Angle (a1,a2) [deg]:", ads_real_alpha_label)
    angle_sub_ads_label = QLabel("- deg")
    ads_real_params_layout.addRow("Sub-Ads Angle:", angle_sub_ads_label)
    calculate_sub_ads_angle_button = QPushButton("Calculate Sub-Ads Angle")
    ads_real_params_layout.addRow(calculate_sub_ads_angle_button)
    controls_panel_layout.addWidget(ads_real_params_group)

    controls_panel_layout.addStretch(1)

    close_button = QPushButton("Close")
    button_layout_final = QHBoxLayout()
    button_layout_final.addStretch(1)
    button_layout_final.addWidget(close_button)
    controls_panel_layout.addLayout(button_layout_final)

    main_splitter.addWidget(controls_panel_widget)
    main_splitter.setSizes([500, 400, 300])
    main_splitter.setStretchFactor(0, 1)
    main_splitter.setStretchFactor(1, 1)
    main_splitter.setStretchFactor(2, 0)

    return RealSpaceVisualizerWidgets(
        main_splitter=main_splitter,
        fft_panel_widget=fft_panel_widget,
        fft_view_box=fft_view_box,
        fft_image_item_vis=fft_image_item_vis,
        real_space_plot_widget=real_space_plot_widget,
        real_space_view_box=real_space_view_box,
        display_options_form=display_options_form,
        cb_show_substrate_real_lattice=cb_show_substrate_real_lattice,
        adsorbate_display_checkbox_layout=adsorbate_display_checkbox_layout,
        custom_adsorbate_visibility_checkbox=custom_adsorbate_visibility_checkbox,
        adsorbate_sets_checkbox_layout=adsorbate_sets_checkbox_layout,
        cb_show_g_substrate_fft=cb_show_g_substrate_fft,
        cb_show_g_adsorbate_fft=cb_show_g_adsorbate_fft,
        cb_visual_align=cb_visual_align,
        substrate_lattice_cells_spin=substrate_lattice_cells_spin,
        adsorbate_lattice_cells_spin=adsorbate_lattice_cells_spin,
        substrate_atom_size_spin=substrate_atom_size_spin,
        adsorbate_atom_size_spin=adsorbate_atom_size_spin,
        adsorbate_marker_alpha_spin=adsorbate_marker_alpha_spin,
        lattice_outline_width_spin=lattice_outline_width_spin,
        lattice_outline_alpha_spin=lattice_outline_alpha_spin,
        supercell_size_spinbox=supercell_size_spinbox,
        custom_a1_x_spin=custom_a1_x_spin,
        custom_a1_y_spin=custom_a1_y_spin,
        custom_a2_x_spin=custom_a2_x_spin,
        custom_a2_y_spin=custom_a2_y_spin,
        custom_offset_x_spin=custom_offset_x_spin,
        custom_offset_y_spin=custom_offset_y_spin,
        custom_a1_length_spin=custom_a1_length_spin,
        custom_a2_length_spin=custom_a2_length_spin,
        custom_angle_a1_spin=custom_angle_a1_spin,
        custom_angle_between_spin=custom_angle_between_spin,
        custom_symbol_combo=custom_symbol_combo,
        custom_length_convert_button=custom_length_convert_button,
        custom_adsorbate_apply_button=custom_adsorbate_apply_button,
        custom_adsorbate_clear_button=custom_adsorbate_clear_button,
        substrate_offset_x_spin=substrate_offset_x_spin,
        substrate_offset_y_spin=substrate_offset_y_spin,
        adsorbate_offset_x_spin=adsorbate_offset_x_spin,
        adsorbate_offset_y_spin=adsorbate_offset_y_spin,
        launch_3d_button=launch_3d_button,
        info_sub_rot_label=info_sub_rot_label,
        info_sub_scale_label=info_sub_scale_label,
        info_sub_rmse_label=info_sub_rmse_label,
        sub_real_a1_label=sub_real_a1_label,
        sub_real_a2_label=sub_real_a2_label,
        sub_real_alpha_label=sub_real_alpha_label,
        calibration_sigma_label=calibration_sigma_label,
        ads_set_combo_vis=ads_set_combo_vis,
        ads_real_a1_label=ads_real_a1_label,
        ads_real_a2_label=ads_real_a2_label,
        ads_real_alpha_label=ads_real_alpha_label,
        angle_sub_ads_label=angle_sub_ads_label,
        calculate_sub_ads_angle_button=calculate_sub_ads_angle_button,
        close_button=close_button,
        extra_references={
            "controls_panel_widget": controls_panel_widget,
            "display_options_group": display_options_group,
            "transform_info_group": transform_info_group,
            "sub_real_params_group": sub_real_params_group,
            "ads_real_params_group": ads_real_params_group,
        },
    )
