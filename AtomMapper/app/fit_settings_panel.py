"""Dock-friendly editor widget for AtomMapper local-fit settings."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .fit_models import LocalFitModelType
from .fit_settings import (
    CommonFitSettings,
    FitParameterTier,
    FitSettingsState,
    GaussianFitSettings,
    LorentzianFitSettings,
    ParameterBounds,
    VoigtFitSettings,
)
from .models import LoadedImage, ROIState


def _normalize_optional_float_text(text: str) -> float | None:
    value = text.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class OptionalFloatLineEdit(QLineEdit):
    """QLineEdit that represents an optional float value."""

    value_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setPlaceholderText("auto")
        self.editingFinished.connect(self._emit_value_changed)

    def set_optional_value(self, value: float | None) -> None:
        self.blockSignals(True)
        self.setText("" if value is None else f"{float(value):.6g}")
        self.blockSignals(False)

    def optional_value(self) -> float | None:
        return _normalize_optional_float_text(self.text())

    def _emit_value_changed(self) -> None:
        self.value_changed.emit(self.optional_value())


class ParameterBoundsEditor(QWidget):
    """Inline editor for lower/upper optional bounds."""

    value_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.lower_edit = OptionalFloatLineEdit(self)
        self.lower_edit.setObjectName("fit_bounds_lower_edit")
        self.lower_edit.setPlaceholderText("min")
        self.upper_edit = OptionalFloatLineEdit(self)
        self.upper_edit.setObjectName("fit_bounds_upper_edit")
        self.upper_edit.setPlaceholderText("max")

        layout.addWidget(self.lower_edit)
        layout.addWidget(QLabel("..", self))
        layout.addWidget(self.upper_edit)

        self.lower_edit.value_changed.connect(self._emit_value_changed)
        self.upper_edit.value_changed.connect(self._emit_value_changed)

    def set_value(self, value: ParameterBounds) -> None:
        self.lower_edit.set_optional_value(value.lower)
        self.upper_edit.set_optional_value(value.upper)

    def value(self) -> ParameterBounds:
        return ParameterBounds(
            lower=self.lower_edit.optional_value(),
            upper=self.upper_edit.optional_value(),
        ).normalized()

    def _emit_value_changed(self, _value: object) -> None:
        self.value_changed.emit(self.value())


class FitSettingsPanelWidget(QWidget):
    """Non-modal editor for fit-model selection and parameter changes."""

    fit_settings_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("atommapper_fit_settings_panel")
        self.fit_settings_state = FitSettingsState()
        self._updating_ui = False
        self._section_editors: dict[str, dict[str, QWidget]] = {}
        self._model_page_index: dict[LocalFitModelType, int] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Fit Settings", self)
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.context_label = QLabel("No active STM image selected.", self)
        self.context_label.setWordWrap(True)
        self.context_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        self.model_combo = QComboBox(self)
        self.model_combo.setObjectName("atommapper_fit_model_combo")
        for model in LocalFitModelType:
            self.model_combo.addItem(model.value.capitalize(), model)

        self.common_group = self._build_section_group("Common", "common", CommonFitSettings())

        self.model_stack = QStackedWidget(self)
        self.model_stack.setObjectName("atommapper_fit_model_stack")
        self._add_model_page(LocalFitModelType.GAUSSIAN, "Gaussian", "gaussian", GaussianFitSettings())
        self._add_model_page(
            LocalFitModelType.LORENTZIAN,
            "Lorentzian",
            "lorentzian",
            LorentzianFitSettings(),
        )
        self._add_model_page(LocalFitModelType.VOIGT, "Voigt", "voigt", VoigtFitSettings())

        layout.addWidget(title)
        layout.addWidget(self.context_label)
        layout.addWidget(QLabel("Model", self))
        layout.addWidget(self.model_combo)
        layout.addWidget(self.common_group)
        layout.addWidget(self.model_stack, 1)

        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        self.set_fit_settings_state(self.fit_settings_state)

    def set_fit_settings_state(self, state: FitSettingsState) -> None:
        """Push a new fit-settings state into the panel without re-emitting signals."""

        self.fit_settings_state = state.normalized()
        self._updating_ui = True
        try:
            model_index = self.model_combo.findData(self.fit_settings_state.model)
            if model_index >= 0:
                self.model_combo.setCurrentIndex(model_index)
            self.model_stack.setCurrentIndex(self._model_page_index[self.fit_settings_state.model])
            self._apply_section_state("common", self.fit_settings_state.common)
            self._apply_section_state("gaussian", self.fit_settings_state.gaussian)
            self._apply_section_state("lorentzian", self.fit_settings_state.lorentzian)
            self._apply_section_state("voigt", self.fit_settings_state.voigt)
        finally:
            self._updating_ui = False

    def set_context(self, active_image: LoadedImage | None, roi_state: ROIState | None) -> None:
        """Refresh the passive context summary for the current image and ROI."""

        if active_image is None:
            self.context_label.setText("No active STM image selected.")
            return

        image_summary = (
            f"Active image: {active_image.display_name} "
            f"({active_image.pixels_x}x{active_image.pixels_y} px)"
        )
        if roi_state is None:
            self.context_label.setText(f"{image_summary}\nROI: none")
            return

        self.context_label.setText(
            f"{image_summary}\n"
            f"ROI: x={roi_state.x}, y={roi_state.y}, "
            f"{roi_state.width}x{roi_state.height} px"
        )

    def _add_model_page(
        self,
        model: LocalFitModelType,
        title: str,
        section_name: str,
        section_state: Any,
    ) -> None:
        page = QWidget(self)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)
        page_layout.addWidget(self._build_section_group(title, section_name, section_state))
        page_layout.addStretch(1)
        index = self.model_stack.addWidget(page)
        self._model_page_index[model] = index

    def _build_section_group(self, title: str, section_name: str, section_state: Any) -> QGroupBox:
        group = QGroupBox(title, self)
        group.setObjectName(f"atommapper_fit_section_{section_name}")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        basic_form = QFormLayout()
        advanced_form = QFormLayout()
        basic_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        basic_form.setSpacing(8)
        advanced_form.setSpacing(8)

        for field_info in fields(type(section_state)):
            label = field_info.metadata.get("label")
            tier = field_info.metadata.get("tier")
            if label is None or tier is None:
                continue
            editor = self._create_editor(section_name, field_info.name, getattr(section_state, field_info.name))
            self._section_editors.setdefault(section_name, {})[field_info.name] = editor
            target_form = basic_form if FitParameterTier(tier) is FitParameterTier.BASIC else advanced_form
            target_form.addRow(str(label), editor)

        if basic_form.rowCount() > 0:
            basic_box = QGroupBox("Basic", group)
            basic_box.setLayout(basic_form)
            layout.addWidget(basic_box)
        if advanced_form.rowCount() > 0:
            advanced_box = QGroupBox("Advanced", group)
            advanced_box.setLayout(advanced_form)
            layout.addWidget(advanced_box)
        return group

    def _create_editor(self, section_name: str, field_name: str, value: Any) -> QWidget:
        if isinstance(value, bool):
            editor = QCheckBox(self)
            editor.setObjectName(f"atommapper_fit_{section_name}_{field_name}_checkbox")
            editor.toggled.connect(
                lambda checked, s=section_name, f=field_name: self._on_field_changed(s, f, bool(checked))
            )
            return editor

        if isinstance(value, int):
            editor = QSpinBox(self)
            editor.setObjectName(f"atommapper_fit_{section_name}_{field_name}_spinbox")
            editor.setRange(10, 1_000_000)
            editor.valueChanged.connect(
                lambda new_value, s=section_name, f=field_name: self._on_field_changed(s, f, int(new_value))
            )
            return editor

        if isinstance(value, ParameterBounds):
            editor = ParameterBoundsEditor(self)
            editor.setObjectName(f"atommapper_fit_{section_name}_{field_name}_bounds")
            editor.value_changed.connect(
                lambda new_value, s=section_name, f=field_name: self._on_field_changed(s, f, new_value)
            )
            return editor

        editor = OptionalFloatLineEdit(self)
        editor.setObjectName(f"atommapper_fit_{section_name}_{field_name}_lineedit")
        editor.value_changed.connect(
            lambda new_value, s=section_name, f=field_name: self._on_field_changed(s, f, new_value)
        )
        return editor

    def _apply_section_state(self, section_name: str, section_state: Any) -> None:
        for field_info in fields(type(section_state)):
            editor = self._section_editors.get(section_name, {}).get(field_info.name)
            if editor is None:
                continue
            value = getattr(section_state, field_info.name)
            if isinstance(editor, QCheckBox):
                editor.blockSignals(True)
                editor.setChecked(bool(value))
                editor.blockSignals(False)
            elif isinstance(editor, QSpinBox):
                editor.blockSignals(True)
                editor.setValue(int(value))
                editor.blockSignals(False)
            elif isinstance(editor, ParameterBoundsEditor):
                editor.blockSignals(True)
                editor.set_value(value)
                editor.blockSignals(False)
            elif isinstance(editor, OptionalFloatLineEdit):
                editor.set_optional_value(value)

    def _on_model_combo_changed(self, index: int) -> None:
        if self._updating_ui or index < 0:
            return
        model = self.model_combo.itemData(index)
        if model is None:
            return
        self.fit_settings_state = self.fit_settings_state.with_model(model)
        self.model_stack.setCurrentIndex(self._model_page_index[self.fit_settings_state.model])
        self.fit_settings_changed.emit(self.fit_settings_state)

    def _on_field_changed(self, section_name: str, field_name: str, value: Any) -> None:
        if self._updating_ui:
            return
        current_section = getattr(self.fit_settings_state, section_name)
        updated_section = replace(current_section, **{field_name: value}).normalized()
        self.fit_settings_state = replace(self.fit_settings_state.normalized(), **{section_name: updated_section})
        self.fit_settings_changed.emit(self.fit_settings_state)
