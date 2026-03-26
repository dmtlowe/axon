"""
Dock widgets for the Axon Analysis Review GUI.

Three widget groups:
A. Navigation & QC — prev/next, accept/reject, progress
B. Parameter sliders — per-panel pipeline param adjustment
C. MDI painting controls — brush size, clear mask
"""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTextEdit, QGroupBox, QProgressBar, QSlider,
    QDoubleSpinBox, QSpinBox, QFrame, QCheckBox,
)
from qtpy.QtCore import Qt, Signal

from .data_model import DEFAULT_PARAMS, SCALE_PX_PER_UM


# ─────────────────────────────────────────────────────────────────
# A. Navigation & QC Widget
# ─────────────────────────────────────────────────────────────────

class NavigationWidget(QWidget):
    """
    Panel/neuron navigation, QC flagging, and progress display.
    """

    # Signals emitted when user interacts
    navigate_next = Signal()
    navigate_prev = Signal()
    qc_accepted = Signal()
    qc_rejected = Signal()
    qc_modified = Signal()
    jump_requested = Signal(str, int)  # panel_name, neuron_idx
    adjust_start_requested = Signal()
    reset_start_requested = Signal()
    export_requested = Signal()
    colourblind_toggled = Signal(bool)
    channel_toggled = Signal(str, bool)  # channel_name, visible

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Header info ──
        self.panel_label = QLabel("Panel: —")
        self.panel_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self.panel_label)

        self.neuron_label = QLabel("Neuron: — / —")
        layout.addWidget(self.neuron_label)

        self.length_label = QLabel("Axon length: —")
        layout.addWidget(self.length_label)

        self.qc_status_label = QLabel("Status: pending")
        self.qc_status_label.setStyleSheet("font-style: italic;")
        layout.addWidget(self.qc_status_label)

        # ── Colourblind toggle ──
        self.colourblind_cb = QCheckBox("Colourblind-friendly mode")
        self.colourblind_cb.setToolTip(
            "Switches overlay colours to a palette safe for\n"
            "protanopia, deuteranopia, and tritanopia.\n"
            "Uses blue/orange/yellow/purple instead of red/green/cyan."
        )
        self.colourblind_cb.toggled.connect(self.colourblind_toggled.emit)
        layout.addWidget(self.colourblind_cb)

        # ── Channel view ──
        chan_group = QGroupBox("Channel View")
        chan_layout = QVBoxLayout(chan_group)

        self.composite_cb = QCheckBox("Composite (RGB)")
        self.composite_cb.setChecked(False)
        self.red_cb = QCheckBox("Red — Tubulin")
        self.red_cb.setChecked(True)
        self.green_cb = QCheckBox("Green — HRP")
        self.green_cb.setChecked(True)
        self.blue_cb = QCheckBox("Blue — DAPI")
        self.blue_cb.setChecked(True)
        self.grey_cb = QCheckBox("Greyscale")

        self.composite_cb.toggled.connect(
            lambda v: self.channel_toggled.emit("composite", v))
        self.red_cb.toggled.connect(
            lambda v: self.channel_toggled.emit("red", v))
        self.green_cb.toggled.connect(
            lambda v: self.channel_toggled.emit("green", v))
        self.blue_cb.toggled.connect(
            lambda v: self.channel_toggled.emit("blue", v))
        self.grey_cb.toggled.connect(
            lambda v: self.channel_toggled.emit("grey", v))

        chan_layout.addWidget(self.composite_cb)
        chan_layout.addWidget(self.red_cb)
        chan_layout.addWidget(self.green_cb)
        chan_layout.addWidget(self.blue_cb)
        chan_layout.addWidget(self.grey_cb)
        layout.addWidget(chan_group)

        # ── Progress ──
        progress_group = QGroupBox("Progress")
        pg_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        pg_layout.addWidget(self.progress_bar)
        self.progress_detail = QLabel("0 / 0 neurons reviewed")
        pg_layout.addWidget(self.progress_detail)
        layout.addWidget(progress_group)

        # ── Navigation buttons ──
        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("< Prev")
        self.next_btn = QPushButton("Next >")
        self.prev_btn.clicked.connect(self.navigate_prev.emit)
        self.next_btn.clicked.connect(self.navigate_next.emit)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.next_btn)
        layout.addLayout(nav_row)

        # ── QC buttons ──
        qc_group = QGroupBox("Quality Control")
        qc_layout = QVBoxLayout(qc_group)

        qc_row = QHBoxLayout()
        self.accept_btn = QPushButton("Accept")
        self.reject_btn = QPushButton("Reject")
        self.modified_btn = QPushButton("Modified")

        self.accept_btn.setStyleSheet(
            "QPushButton { background-color: #2d8a4e; color: white; "
            "font-weight: bold; padding: 6px; }"
        )
        self.reject_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; "
            "font-weight: bold; padding: 6px; }"
        )
        self.modified_btn.setStyleSheet(
            "QPushButton { background-color: #d4a017; color: white; "
            "font-weight: bold; padding: 6px; }"
        )

        self.accept_btn.clicked.connect(self.qc_accepted.emit)
        self.reject_btn.clicked.connect(self.qc_rejected.emit)
        self.modified_btn.clicked.connect(self.qc_modified.emit)

        qc_row.addWidget(self.accept_btn)
        qc_row.addWidget(self.reject_btn)
        qc_row.addWidget(self.modified_btn)
        qc_layout.addLayout(qc_row)

        # Notes
        qc_layout.addWidget(QLabel("Notes:"))
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(60)
        self.notes_edit.setPlaceholderText("Optional reviewer notes...")
        qc_layout.addWidget(self.notes_edit)

        layout.addWidget(qc_group)

        # ── Axon start adjustment ──
        start_group = QGroupBox("Axon Start Point")
        start_layout = QVBoxLayout(start_group)

        self.adjust_start_btn = QPushButton("Adjust Start Point")
        self.adjust_start_btn.setStyleSheet(
            "QPushButton { background-color: #e67e22; color: white; "
            "font-weight: bold; padding: 6px; }"
        )
        self.adjust_start_btn.setToolTip(
            "Click this, then click on the axon where the\n"
            "true start point should be (where axon leaves soma)."
        )
        self.adjust_start_btn.clicked.connect(self.adjust_start_requested.emit)
        start_layout.addWidget(self.adjust_start_btn)

        self.reset_start_btn = QPushButton("Reset to Original")
        self.reset_start_btn.clicked.connect(self.reset_start_requested.emit)
        start_layout.addWidget(self.reset_start_btn)

        self.start_status_label = QLabel("")
        self.start_status_label.setStyleSheet(
            "color: #e67e22; font-style: italic; font-size: 11px;"
        )
        start_layout.addWidget(self.start_status_label)

        layout.addWidget(start_group)

        # ── Jump to panel ──
        jump_group = QGroupBox("Jump to Panel")
        jump_layout = QHBoxLayout(jump_group)
        self.panel_combo = QComboBox()
        self.panel_combo.setMinimumWidth(150)
        jump_btn = QPushButton("Go")
        jump_btn.clicked.connect(self._on_jump)
        jump_layout.addWidget(self.panel_combo)
        jump_layout.addWidget(jump_btn)
        layout.addWidget(jump_group)

        # ── Export ──
        self.export_btn = QPushButton("Export Dataset")
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #16a085; color: white; "
            "font-weight: bold; padding: 8px; font-size: 13px; }"
        )
        self.export_btn.setToolTip(
            "Export all accepted/modified neurons as an\n"
            "ML-ready dataset (images, masks, CSV)."
        )
        self.export_btn.clicked.connect(self.export_requested.emit)
        layout.addWidget(self.export_btn)

        layout.addStretch()

    def update_display(self, panel_name, neuron_idx, neuron_data,
                       neuron_count, progress):
        """Update all display elements for the current neuron."""
        self.panel_label.setText(f"Panel: {panel_name or '—'}")
        self.neuron_label.setText(
            f"Neuron: {neuron_idx + 1} / {neuron_count}"
            if neuron_count > 0 else "Neuron: — / —"
        )

        if neuron_data:
            length_px = neuron_data.get("axon_length_px", 0)
            length_um = length_px / SCALE_PX_PER_UM
            self.length_label.setText(
                f"Axon length: {length_px:.0f} px ({length_um:.1f} \u00b5m)"
            )
            flag = neuron_data.get("qc_flag", "pending")
            self.qc_status_label.setText(f"Status: {flag}")
            self.notes_edit.setText(neuron_data.get("notes", ""))
        else:
            self.length_label.setText("Axon length: —")
            self.qc_status_label.setText("Status: —")
            self.notes_edit.clear()

        # Progress bar
        total = progress.get("total_neurons", 0)
        reviewed = (progress.get("accepted", 0) +
                    progress.get("rejected", 0) +
                    progress.get("modified", 0))
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(reviewed)
        self.progress_detail.setText(
            f"{reviewed} / {total} neurons reviewed  "
            f"({progress.get('with_mdi_mask', 0)} with MDI masks)"
        )

    def populate_panel_combo(self, panel_names):
        """Fill the jump-to-panel dropdown."""
        self.panel_combo.clear()
        self.panel_combo.addItems(panel_names)

    def get_notes(self):
        """Return current notes text."""
        return self.notes_edit.toPlainText()

    def _on_jump(self):
        panel_name = self.panel_combo.currentText()
        if panel_name:
            self.jump_requested.emit(panel_name, 0)


# ─────────────────────────────────────────────────────────────────
# B. Parameter Sliders Widget
# ─────────────────────────────────────────────────────────────────

class ParameterWidget(QWidget):
    """
    Pipeline parameter sliders with Re-run and Reset buttons.
    """

    rerun_requested = Signal()
    reset_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Store references to all spinboxes
        self._spinboxes = {}

        # ── Nucleus Detection ──
        nuc_group = QGroupBox("Nucleus Detection")
        nuc_layout = QVBoxLayout(nuc_group)
        self._add_int_slider(nuc_layout, "nucleus.min_size",
                             "Min size", 50, 2000, 100)
        self._add_float_slider(nuc_layout, "nucleus.blur_sigma",
                               "Blur sigma", 1.0, 30.0, 10.0, 0.5)
        self._add_float_slider(nuc_layout, "nucleus.hrp_threshold_factor",
                               "HRP threshold", 1.0, 5.0, 2.0, 0.1)
        layout.addWidget(nuc_group)

        # ── Neuron Tracing ──
        trace_group = QGroupBox("Neuron Tracing")
        trace_layout = QVBoxLayout(trace_group)
        self._add_float_slider(trace_layout, "trace.threshold_pct",
                               "Threshold %", 0.02, 0.50, 0.10, 0.01)
        self._add_float_slider(trace_layout, "trace.min_reach_ratio",
                               "Min reach ratio", 1.0, 15.0, 6.0, 0.5)
        self._add_int_slider(trace_layout, "trace.border_margin",
                             "Border margin", 0, 800, 350)
        self._add_int_slider(trace_layout, "trace.min_mask_size",
                             "Min mask size", 5000, 500000, 90000, step=5000)
        self._add_int_slider(trace_layout, "trace.min_seed_intensity",
                             "Min seed intensity", 5, 100, 15)
        layout.addWidget(trace_group)

        # ── Axon Measurement ──
        axon_group = QGroupBox("Axon Measurement")
        axon_layout = QVBoxLayout(axon_group)
        self._add_float_slider(axon_layout, "axon.intensity_weight",
                               "Intensity weight", 0.5, 5.0, 2.0, 0.1)
        self._add_int_slider(axon_layout, "axon.nucleus_snap_distance",
                             "Nucleus snap dist", 5, 150, 30)
        layout.addWidget(axon_group)

        # ── Buttons ──
        btn_row = QHBoxLayout()
        self.rerun_btn = QPushButton("Re-run Panel")
        self.rerun_btn.setStyleSheet(
            "QPushButton { background-color: #2980b9; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self.reset_btn = QPushButton("Reset to Defaults")

        self.rerun_btn.clicked.connect(self.rerun_requested.emit)
        self.reset_btn.clicked.connect(self._reset_and_emit)

        btn_row.addWidget(self.rerun_btn)
        btn_row.addWidget(self.reset_btn)
        layout.addLayout(btn_row)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def _add_float_slider(self, layout, key, label, min_val, max_val,
                          default, step=0.1):
        """Add a labelled float spinbox."""
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(default)
        spin.setDecimals(2)
        row.addWidget(lbl)
        row.addWidget(spin)
        layout.addLayout(row)
        self._spinboxes[key] = spin

    def _add_int_slider(self, layout, key, label, min_val, max_val,
                        default, step=1):
        """Add a labelled integer spinbox."""
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(default)
        row.addWidget(lbl)
        row.addWidget(spin)
        layout.addLayout(row)
        self._spinboxes[key] = spin

    def get_params(self):
        """
        Read current slider values and return as nested param dict.

        Returns dict like: {"nucleus": {...}, "trace": {...}, "axon": {...}}
        """
        params = {"nucleus": {}, "trace": {}, "axon": {}}
        for key, spin in self._spinboxes.items():
            section, param = key.split(".", 1)
            params[section][param] = spin.value()
        return params

    def set_params(self, params):
        """Set slider values from a param dict."""
        for section, values in params.items():
            for param, value in values.items():
                key = f"{section}.{param}"
                if key in self._spinboxes:
                    self._spinboxes[key].setValue(value)

    def _reset_and_emit(self):
        """Reset all sliders to defaults and emit signal."""
        self.set_params(DEFAULT_PARAMS)
        self.reset_requested.emit()


# ─────────────────────────────────────────────────────────────────
# C. MDI Painting Controls
# ─────────────────────────────────────────────────────────────────

class MDIPaintWidget(QWidget):
    """
    Controls for MDI mask painting.
    """

    clear_mask_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        group = QGroupBox("MDI Mask Painting")
        g_layout = QVBoxLayout(group)

        # Brush size
        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Brush size:"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 100)
        self.brush_spin.setValue(15)
        brush_row.addWidget(self.brush_spin)
        g_layout.addLayout(brush_row)

        # Clear button
        self.clear_btn = QPushButton("Clear MDI Mask")
        self.clear_btn.setStyleSheet(
            "QPushButton { background-color: #8e44ad; color: white; "
            "padding: 6px; }"
        )
        self.clear_btn.clicked.connect(self.clear_mask_requested.emit)
        g_layout.addWidget(self.clear_btn)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        g_layout.addWidget(line)

        # Instructions
        instructions = QLabel(
            "How to paint MDI regions:\n"
            "1. Select the 'MDI Mask' layer\n"
            "2. Press 2 for paint brush\n"
            "3. Paint over disorganised regions\n"
            "4. Use [ ] to resize brush\n"
            "5. Press 1 for erase mode\n"
            "\n"
            "Label 1 = MDI region\n"
            "Masks auto-save on navigation."
        )
        instructions.setStyleSheet("color: #888; font-size: 11px;")
        instructions.setWordWrap(True)
        g_layout.addWidget(instructions)

        layout.addWidget(group)
        layout.addStretch()
