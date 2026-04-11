"""Preprocessing dialog skeleton for AtomMapper."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .models import LoadedImage
from .preprocessing import apply_blur, apply_bm3d, apply_non_local_means, is_bm3d_available
from .preprocessing_preview import PreprocessingImagePreview
from .preprocessing_state import (
    BM3DParameters,
    BlurParameters,
    NonLocalMeansParameters,
    PreprocessingMethod,
    PreprocessingPreviewRequest,
    PreprocessingPreviewResult,
    PreviewViewport,
    PreprocessingState,
)


class PreprocessingDialog(QDialog):
    """Skeleton dialog for future preprocessing preview and parameter editing."""

    def __init__(self, loaded_image: LoadedImage, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("atommapper_preprocessing_dialog")
        self.loaded_image = loaded_image
        self.preprocessing_state = PreprocessingState()
        self.latest_preview_result: Optional[PreprocessingPreviewResult] = None
        self.bm3d_available = is_bm3d_available()
        self.preview_viewport = PreviewViewport(
            x=0,
            y=0,
            width=self.loaded_image.pixels_x,
            height=self.loaded_image.pixels_y,
        ).normalized()

        self.setWindowTitle(f"Preprocessing - {loaded_image.display_name}")
        self.resize(1100, 720)
        self.setModal(True)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(14)

        self.header_label = QLabel(f"Preprocessing: {loaded_image.display_name}")
        self.header_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.subtitle_label = QLabel(
            "Preview skeleton. Applying parameters will be enabled in the next step."
        )
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        content_widget = QWidget(self)
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)

        preview_layout = QHBoxLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(16)
        preview_layout.addWidget(self._build_original_preview_group(), 1)
        preview_layout.addWidget(self._build_processed_preview_group(), 1)

        preview_container = QWidget(self)
        preview_container.setLayout(preview_layout)

        self.parameters_group = self._build_parameters_group()
        self.parameters_group.setMinimumWidth(280)
        self.parameters_group.setMaximumWidth(360)

        content_layout.addWidget(preview_container, 1)
        content_layout.addWidget(self.parameters_group)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        self.apply_button = self.button_box.button(QDialogButtonBox.StandardButton.Apply)
        self.cancel_button = self.button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if self.apply_button is not None:
            self.apply_button.setObjectName("atommapper_preprocessing_apply_button")
            self.apply_button.setEnabled(False)
            self.apply_button.setToolTip("Apply will be enabled after preview integration.")
            self.apply_button.clicked.connect(self.accept)
        if self.cancel_button is not None:
            self.cancel_button.setObjectName("atommapper_preprocessing_cancel_button")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self.blur_sigma_spinbox.valueChanged.connect(self._on_blur_sigma_changed)
        self.nlm_h_spinbox.valueChanged.connect(self._on_nlm_parameters_changed)
        self.nlm_patch_size_spinbox.valueChanged.connect(self._on_nlm_parameters_changed)
        self.nlm_patch_distance_spinbox.valueChanged.connect(self._on_nlm_parameters_changed)
        self.nlm_fast_mode_checkbox.toggled.connect(self._on_nlm_parameters_changed)
        self.bm3d_sigma_spinbox.valueChanged.connect(self._on_bm3d_sigma_changed)

        root_layout.addWidget(self.header_label)
        root_layout.addWidget(self.subtitle_label)
        root_layout.addWidget(content_widget, 1)
        root_layout.addWidget(self.button_box)
        self._sync_method_availability()
        self._sync_widgets_from_state()
        self._refresh_original_preview()
        self._refresh_processed_preview()

    def _build_original_preview_group(self) -> QGroupBox:
        group = QGroupBox("Original", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.original_preview = PreprocessingImagePreview(
            "Original image preview.",
            group,
        )
        self.original_preview_label = self.original_preview.image_label
        layout.addWidget(self.original_preview, 1)
        return group

    def _build_processed_preview_group(self) -> QGroupBox:
        group = QGroupBox("Processed", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.processed_preview = PreprocessingImagePreview(
            "Processed preview is currently showing the same viewport as the source image.",
            group,
        )
        self.processed_preview_label = self.processed_preview.status_label
        layout.addWidget(self.processed_preview, 1)
        return group

    def _build_parameters_group(self) -> QGroupBox:
        group = QGroupBox("Parameters", self)
        layout = QGridLayout(group)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)

        method_label = QLabel("Method", group)
        self.method_combo = QComboBox(group)
        self.method_combo.addItem("Blur", userData=PreprocessingMethod.BLUR)
        self.method_combo.addItem("Non-local means", userData=PreprocessingMethod.NLM)
        self.method_combo.addItem("BM3D", userData=PreprocessingMethod.BM3D)

        blur_sigma_label = QLabel("Blur sigma [px]", group)
        self.blur_sigma_spinbox = QDoubleSpinBox(group)
        self.blur_sigma_spinbox.setDecimals(2)
        self.blur_sigma_spinbox.setRange(0.05, 50.0)
        self.blur_sigma_spinbox.setSingleStep(0.05)

        nlm_h_label = QLabel("NLM h", group)
        self.nlm_h_spinbox = QDoubleSpinBox(group)
        self.nlm_h_spinbox.setDecimals(3)
        self.nlm_h_spinbox.setRange(0.001, 50.0)
        self.nlm_h_spinbox.setSingleStep(0.01)

        nlm_patch_size_label = QLabel("NLM patch size", group)
        self.nlm_patch_size_spinbox = QSpinBox(group)
        self.nlm_patch_size_spinbox.setRange(3, 99)
        self.nlm_patch_size_spinbox.setSingleStep(2)

        nlm_patch_distance_label = QLabel("NLM patch distance", group)
        self.nlm_patch_distance_spinbox = QSpinBox(group)
        self.nlm_patch_distance_spinbox.setRange(1, 99)
        self.nlm_patch_distance_spinbox.setSingleStep(1)

        self.nlm_fast_mode_checkbox = QCheckBox("Use fast mode", group)

        bm3d_sigma_label = QLabel("BM3D sigma_psd", group)
        self.bm3d_sigma_spinbox = QDoubleSpinBox(group)
        self.bm3d_sigma_spinbox.setDecimals(3)
        self.bm3d_sigma_spinbox.setRange(0.001, 50.0)
        self.bm3d_sigma_spinbox.setSingleStep(0.01)

        self.bm3d_availability_label = QLabel(group)
        self.bm3d_availability_label.setWordWrap(True)
        self.bm3d_availability_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        self.parameters_placeholder_label = QLabel(
            "Parameter controls will appear here in the next step.",
            group,
        )
        self.parameters_placeholder_label.setWordWrap(True)
        self.parameters_placeholder_label.setStyleSheet("color: palette(mid);")

        layout.addWidget(method_label, 0, 0)
        layout.addWidget(self.method_combo, 0, 1)
        layout.addWidget(blur_sigma_label, 1, 0)
        layout.addWidget(self.blur_sigma_spinbox, 1, 1)
        layout.addWidget(nlm_h_label, 2, 0)
        layout.addWidget(self.nlm_h_spinbox, 2, 1)
        layout.addWidget(nlm_patch_size_label, 3, 0)
        layout.addWidget(self.nlm_patch_size_spinbox, 3, 1)
        layout.addWidget(nlm_patch_distance_label, 4, 0)
        layout.addWidget(self.nlm_patch_distance_spinbox, 4, 1)
        layout.addWidget(self.nlm_fast_mode_checkbox, 5, 0, 1, 2)
        layout.addWidget(bm3d_sigma_label, 6, 0)
        layout.addWidget(self.bm3d_sigma_spinbox, 6, 1)
        layout.addWidget(self.bm3d_availability_label, 7, 0, 1, 2)
        layout.addWidget(self.parameters_placeholder_label, 8, 0, 1, 2)
        layout.setRowStretch(9, 1)
        return group

    def _sync_method_availability(self) -> None:
        """Update method availability state in the combo box and helper labels."""

        model = self.method_combo.model()
        bm3d_index = self.method_combo.findData(PreprocessingMethod.BM3D)
        if bm3d_index >= 0:
            item = model.item(bm3d_index)
            if item is not None:
                item.setEnabled(self.bm3d_available)
            tooltip = (
                "BM3D backend available."
                if self.bm3d_available
                else "BM3D backend unavailable. Install the 'bm3d' package to enable this method."
            )
            self.method_combo.setItemData(
                bm3d_index,
                tooltip,
                int(Qt.ItemDataRole.ToolTipRole),
            )
        self.bm3d_availability_label.setText(
            "BM3D backend available."
            if self.bm3d_available
            else "BM3D backend unavailable. Install the 'bm3d' package to enable preview and apply."
        )

    def current_preview_request(self) -> PreprocessingPreviewRequest:
        """Return the current normalized preview request."""

        return PreprocessingPreviewRequest(
            image_id=self.loaded_image.image_id,
            source_group_id=self.loaded_image.source_group_id,
            state=self.preprocessing_state,
            viewport=self.preview_viewport,
        ).normalized()

    def _sync_widgets_from_state(self) -> None:
        """Synchronize widget values from the current preprocessing state."""

        self.preprocessing_state = self.preprocessing_state.normalized()
        active_method = self.preprocessing_state.method
        index = self.method_combo.findData(active_method)
        if index >= 0 and index != self.method_combo.currentIndex():
            self.method_combo.setCurrentIndex(index)
        if self.blur_sigma_spinbox.value() != self.preprocessing_state.blur.sigma_px:
            self.blur_sigma_spinbox.blockSignals(True)
            self.blur_sigma_spinbox.setValue(self.preprocessing_state.blur.sigma_px)
            self.blur_sigma_spinbox.blockSignals(False)
        if self.nlm_h_spinbox.value() != self.preprocessing_state.nlm.h:
            self.nlm_h_spinbox.blockSignals(True)
            self.nlm_h_spinbox.setValue(self.preprocessing_state.nlm.h)
            self.nlm_h_spinbox.blockSignals(False)
        if self.nlm_patch_size_spinbox.value() != self.preprocessing_state.nlm.patch_size:
            self.nlm_patch_size_spinbox.blockSignals(True)
            self.nlm_patch_size_spinbox.setValue(self.preprocessing_state.nlm.patch_size)
            self.nlm_patch_size_spinbox.blockSignals(False)
        if self.nlm_patch_distance_spinbox.value() != self.preprocessing_state.nlm.patch_distance:
            self.nlm_patch_distance_spinbox.blockSignals(True)
            self.nlm_patch_distance_spinbox.setValue(self.preprocessing_state.nlm.patch_distance)
            self.nlm_patch_distance_spinbox.blockSignals(False)
        if self.nlm_fast_mode_checkbox.isChecked() != self.preprocessing_state.nlm.fast_mode:
            self.nlm_fast_mode_checkbox.blockSignals(True)
            self.nlm_fast_mode_checkbox.setChecked(self.preprocessing_state.nlm.fast_mode)
            self.nlm_fast_mode_checkbox.blockSignals(False)
        if self.bm3d_sigma_spinbox.value() != self.preprocessing_state.bm3d.sigma_psd:
            self.bm3d_sigma_spinbox.blockSignals(True)
            self.bm3d_sigma_spinbox.setValue(self.preprocessing_state.bm3d.sigma_psd)
            self.bm3d_sigma_spinbox.blockSignals(False)
        self._update_parameter_placeholder()
        self._update_apply_button_state()

    def _update_parameter_placeholder(self) -> None:
        """Refresh the parameter placeholder based on active state."""

        active_parameters = self.preprocessing_state.active_parameters
        method_label = self.preprocessing_state.method.label
        if self.preprocessing_state.method is PreprocessingMethod.BLUR:
            self.parameters_placeholder_label.setText(
                f"Active method: {method_label}. "
                f"Normalized parameters: {active_parameters}. "
                "Preview updates live after sigma changes."
            )
            self.blur_sigma_spinbox.setEnabled(True)
            self.nlm_h_spinbox.setEnabled(False)
            self.nlm_patch_size_spinbox.setEnabled(False)
            self.nlm_patch_distance_spinbox.setEnabled(False)
            self.nlm_fast_mode_checkbox.setEnabled(False)
            self.nlm_fast_mode_checkbox.setText("Use fast mode")
            return

        if self.preprocessing_state.method is PreprocessingMethod.NLM:
            self.parameters_placeholder_label.setText(
                f"Active method: {method_label}. "
                f"Normalized parameters: {active_parameters}. "
                "Preview updates live after NLM parameter changes."
            )
            self.blur_sigma_spinbox.setEnabled(False)
            self.nlm_h_spinbox.setEnabled(True)
            self.nlm_patch_size_spinbox.setEnabled(True)
            self.nlm_patch_distance_spinbox.setEnabled(True)
            self.nlm_fast_mode_checkbox.setEnabled(True)
            self.nlm_fast_mode_checkbox.setText("Use fast mode")
            self.bm3d_sigma_spinbox.setEnabled(False)
            return

        if self.preprocessing_state.method is PreprocessingMethod.BM3D:
            self.parameters_placeholder_label.setText(
                f"Active method: {method_label}. "
                f"Normalized parameters: {active_parameters}. "
                + (
                    "Preview updates live after BM3D sigma changes."
                    if self.bm3d_available
                    else "This method is unavailable because the optional BM3D backend is missing."
                )
            )
            self.blur_sigma_spinbox.setEnabled(False)
            self.nlm_h_spinbox.setEnabled(False)
            self.nlm_patch_size_spinbox.setEnabled(False)
            self.nlm_patch_distance_spinbox.setEnabled(False)
            self.nlm_fast_mode_checkbox.setEnabled(False)
            self.nlm_fast_mode_checkbox.setText("Use fast mode")
            self.bm3d_sigma_spinbox.setEnabled(self.bm3d_available)
            return

        self.blur_sigma_spinbox.setEnabled(False)
        self.nlm_h_spinbox.setEnabled(False)
        self.nlm_patch_size_spinbox.setEnabled(False)
        self.nlm_patch_distance_spinbox.setEnabled(False)
        self.nlm_fast_mode_checkbox.setEnabled(False)
        self.nlm_fast_mode_checkbox.setText("Use fast mode")
        self.bm3d_sigma_spinbox.setEnabled(False)
        self.parameters_placeholder_label.setText(
            f"Active method: {method_label}. "
            f"Normalized parameters: {active_parameters}. "
            "This method will be enabled in a later step."
        )

    def _update_apply_button_state(self) -> None:
        """Enable Apply only when the active method is implemented."""

        if self.apply_button is None:
            return

        can_apply = self.preprocessing_state.method in {
            PreprocessingMethod.BLUR,
            PreprocessingMethod.NLM,
        } or (
            self.preprocessing_state.method is PreprocessingMethod.BM3D and self.bm3d_available
        )
        self.apply_button.setEnabled(can_apply)
        if self.preprocessing_state.method is PreprocessingMethod.BLUR:
            self.apply_button.setToolTip("Create a blur-based image variant from the current image.")
        elif self.preprocessing_state.method is PreprocessingMethod.NLM:
            self.apply_button.setToolTip("Create an NLM-denoised image variant from the current image.")
        elif self.preprocessing_state.method is PreprocessingMethod.BM3D and self.bm3d_available:
            self.apply_button.setToolTip("Create a BM3D-denoised image variant from the current image.")
        else:
            self.apply_button.setToolTip(
                "Apply is unavailable because the selected method is not implemented or missing an optional dependency."
            )

    def _refresh_original_preview(self) -> None:
        """Render the original preview using the current shared viewport."""

        self.original_preview.set_preview_image(
            self.loaded_image.image_data,
            viewport=self.preview_viewport,
            status_text="Original image preview.",
        )

    def _refresh_processed_preview(self) -> None:
        """Render the processed-side preview for the current state."""

        request = self.current_preview_request()

        try:
            if self.preprocessing_state.method is PreprocessingMethod.BLUR:
                processed_image = apply_blur(
                    self.loaded_image.image_data,
                    sigma_px=self.preprocessing_state.blur.sigma_px,
                    mode=self.preprocessing_state.blur.mode,
                )
                status_message = (
                    f"Blur preview ready. sigma_px={self.preprocessing_state.blur.sigma_px:.2f}"
                )
            elif self.preprocessing_state.method is PreprocessingMethod.NLM:
                processed_image = apply_non_local_means(
                    self.loaded_image.image_data,
                    h=self.preprocessing_state.nlm.h,
                    patch_size=self.preprocessing_state.nlm.patch_size,
                    patch_distance=self.preprocessing_state.nlm.patch_distance,
                    fast_mode=self.preprocessing_state.nlm.fast_mode,
                )
                status_message = (
                    "NLM preview ready. "
                    f"h={self.preprocessing_state.nlm.h:.3f}, "
                    f"patch={self.preprocessing_state.nlm.patch_size}, "
                    f"distance={self.preprocessing_state.nlm.patch_distance}, "
                    f"fast={self.preprocessing_state.nlm.fast_mode}"
                )
            elif self.preprocessing_state.method is PreprocessingMethod.BM3D:
                if not self.bm3d_available:
                    raise RuntimeError("BM3D backend unavailable. Install the 'bm3d' package.")
                processed_image = apply_bm3d(
                    self.loaded_image.image_data,
                    sigma_psd=self.preprocessing_state.bm3d.sigma_psd,
                )
                status_message = (
                    f"BM3D preview ready. sigma_psd={self.preprocessing_state.bm3d.sigma_psd:.3f}"
                )
            else:
                raise NotImplementedError(
                    f"{self.preprocessing_state.method.label} preview is not implemented yet."
                )
            self.latest_preview_result = PreprocessingPreviewResult.from_success(
                request,
                processed_image,
                status_message=status_message,
            )
            self.processed_preview.set_preview_image(
                processed_image,
                viewport=self.preview_viewport,
                status_text=self.latest_preview_result.status_message,
            )
        except Exception as exc:  # pragma: no cover - guarded by preprocessing tests
            self.latest_preview_result = PreprocessingPreviewResult.from_failure(
                request,
                str(exc),
                status_message=(
                    str(exc)
                    if self.preprocessing_state.method is PreprocessingMethod.BM3D
                    and not self.bm3d_available
                    else (
                        f"{self.preprocessing_state.method.label} preview failed."
                        if self.preprocessing_state.method in {
                            PreprocessingMethod.BLUR,
                            PreprocessingMethod.NLM,
                            PreprocessingMethod.BM3D,
                        }
                        else f"{self.preprocessing_state.method.label} preview is not implemented yet."
                    )
                ),
            )
            self.processed_preview.set_preview_image(
                self.loaded_image.image_data,
                viewport=self.preview_viewport,
                status_text=self.latest_preview_result.status_message,
            )

    def _on_method_changed(self, _index: int) -> None:
        """Keep the dialog state aligned with the method selector."""

        selected_method = self.method_combo.currentData()
        if selected_method is None:
            return

        self.preprocessing_state = self.preprocessing_state.with_method(selected_method)
        self._update_parameter_placeholder()
        self._update_apply_button_state()
        self._refresh_processed_preview()

    def _on_blur_sigma_changed(self, value: float) -> None:
        """Update blur parameters and refresh only the processed preview."""

        self.preprocessing_state = PreprocessingState(
            method=self.preprocessing_state.method,
            blur=BlurParameters(
                sigma_px=float(value),
                mode=self.preprocessing_state.blur.mode,
            ),
            nlm=self.preprocessing_state.nlm,
            bm3d=self.preprocessing_state.bm3d,
        ).normalized()
        self._update_parameter_placeholder()
        self._refresh_processed_preview()

    def _on_nlm_parameters_changed(self, *_args: object) -> None:
        """Update NLM parameters and refresh only the processed preview."""

        self.preprocessing_state = PreprocessingState(
            method=self.preprocessing_state.method,
            blur=self.preprocessing_state.blur,
            nlm=NonLocalMeansParameters(
                h=float(self.nlm_h_spinbox.value()),
                patch_size=int(self.nlm_patch_size_spinbox.value()),
                patch_distance=int(self.nlm_patch_distance_spinbox.value()),
                fast_mode=bool(self.nlm_fast_mode_checkbox.isChecked()),
            ),
            bm3d=self.preprocessing_state.bm3d,
        ).normalized()
        self._update_parameter_placeholder()
        self._refresh_processed_preview()

    def _on_bm3d_sigma_changed(self, value: float) -> None:
        """Update BM3D parameters and refresh only the processed preview."""

        self.preprocessing_state = PreprocessingState(
            method=self.preprocessing_state.method,
            blur=self.preprocessing_state.blur,
            nlm=self.preprocessing_state.nlm,
            bm3d=BM3DParameters(
                sigma_psd=float(value),
                stage=self.preprocessing_state.bm3d.stage,
            ),
        ).normalized()
        self._update_parameter_placeholder()
        self._refresh_processed_preview()
