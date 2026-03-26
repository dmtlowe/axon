"""
Axon Analysis Review GUI — main entry point.

Launch with:
    python -m review_app.app

Or from the project root:
    python review_app/app.py
"""

import sys
from pathlib import Path

# Ensure parent dir is on path for pipeline imports
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import napari
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSpinBox, QLineEdit, QGroupBox, QMessageBox,
    QProgressDialog, QApplication,
)
from qtpy.QtCore import Qt

from .data_model import ProjectState
from .batch_processor import batch_process, reprocess_panel
from .viewer_controller import ViewerController
from .widgets import NavigationWidget, ParameterWidget, MDIPaintWidget
from .exporter import export_analysis_csv, export_ml_dataset


# ─────────────────────────────────────────────────────────────────
# Startup Dialog
# ─────────────────────────────────────────────────────────────────

class StartupDialog(QDialog):
    """Dialog to create a new project or open an existing one."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Axon Analysis — Project Setup")
        self.setMinimumWidth(500)

        self.project_dir = None
        self.source_tif = None
        self.grid_rows = 6
        self.grid_cols = 6
        self.is_new = False

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Open existing ──
        open_group = QGroupBox("Open Existing Project")
        open_layout = QVBoxLayout(open_group)
        open_btn = QPushButton("Select project folder...")
        open_btn.clicked.connect(self._open_existing)
        open_layout.addWidget(open_btn)
        layout.addWidget(open_group)

        # ── Or create new ──
        new_group = QGroupBox("Create New Project")
        new_layout = QVBoxLayout(new_group)

        # Source TIF
        tif_row = QHBoxLayout()
        tif_row.addWidget(QLabel("Source TIF:"))
        self.tif_edit = QLineEdit()
        self.tif_edit.setPlaceholderText("Path to multi-panel TIF...")
        tif_browse = QPushButton("Browse...")
        tif_browse.clicked.connect(self._browse_tif)
        tif_row.addWidget(self.tif_edit)
        tif_row.addWidget(tif_browse)
        new_layout.addLayout(tif_row)

        # Grid dims
        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("Grid rows:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 50)
        self.rows_spin.setValue(6)
        grid_row.addWidget(self.rows_spin)
        grid_row.addWidget(QLabel("Grid cols:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 50)
        self.cols_spin.setValue(6)
        grid_row.addWidget(self.cols_spin)
        new_layout.addLayout(grid_row)

        # Project folder
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Project folder:"))
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Where to save the project...")
        dir_browse = QPushButton("Browse...")
        dir_browse.clicked.connect(self._browse_dir)
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(dir_browse)
        new_layout.addLayout(dir_row)

        create_btn = QPushButton("Create Project & Process")
        create_btn.setStyleSheet(
            "QPushButton { background-color: #2980b9; color: white; "
            "font-weight: bold; padding: 10px; font-size: 14px; }"
        )
        create_btn.clicked.connect(self._create_new)
        new_layout.addWidget(create_btn)

        layout.addWidget(new_group)

    def _browse_tif(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Source TIF", "", "TIF Files (*.tif *.tiff)"
        )
        if path:
            self.tif_edit.setText(path)

    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Project Folder"
        )
        if path:
            self.dir_edit.setText(path)

    def _open_existing(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Existing Project Folder"
        )
        if path:
            state_file = Path(path) / "state.json"
            if not state_file.exists():
                QMessageBox.warning(
                    self, "Not a project",
                    f"No state.json found in:\n{path}"
                )
                return
            self.project_dir = path
            self.is_new = False
            self.accept()

    def _create_new(self):
        tif = self.tif_edit.text().strip()
        proj_dir = self.dir_edit.text().strip()

        if not tif or not Path(tif).exists():
            QMessageBox.warning(self, "Error", "Please select a valid TIF file.")
            return
        if not proj_dir:
            QMessageBox.warning(self, "Error", "Please select a project folder.")
            return

        self.source_tif = tif
        self.project_dir = proj_dir
        self.grid_rows = self.rows_spin.value()
        self.grid_cols = self.cols_spin.value()
        self.is_new = True
        self.accept()


# ─────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────

class ReviewApp:
    """
    Wires together the napari viewer, state, controller, and widgets.
    """

    def __init__(self, viewer, state):
        self.viewer = viewer
        self.state = state
        self.controller = ViewerController(viewer, state)

        # Create widgets
        self.nav_widget = NavigationWidget()
        self.param_widget = ParameterWidget()
        self.mdi_widget = MDIPaintWidget()

        # Add dock widgets
        viewer.window.add_dock_widget(
            self.nav_widget, name="Navigation & QC", area="right"
        )
        viewer.window.add_dock_widget(
            self.param_widget, name="Parameters", area="right"
        )
        viewer.window.add_dock_widget(
            self.mdi_widget, name="MDI Painting", area="right"
        )

        # Wire signals
        self.nav_widget.navigate_next.connect(self._on_next)
        self.nav_widget.navigate_prev.connect(self._on_prev)
        self.nav_widget.qc_accepted.connect(lambda: self._set_qc("accepted"))
        self.nav_widget.qc_rejected.connect(lambda: self._set_qc("rejected"))
        self.nav_widget.qc_modified.connect(lambda: self._set_qc("modified"))
        self.nav_widget.jump_requested.connect(self._on_jump)

        self.param_widget.rerun_requested.connect(self._on_rerun)
        self.param_widget.reset_requested.connect(self._on_reset_params)

        self.nav_widget.adjust_start_requested.connect(self._on_adjust_start)
        self.nav_widget.reset_start_requested.connect(self._on_reset_start)
        self.nav_widget.export_requested.connect(self._on_export)
        self.nav_widget.colourblind_toggled.connect(
            self.controller.set_colourblind_mode
        )
        self.nav_widget.channel_toggled.connect(
            self.controller.set_channel_visibility
        )

        self.mdi_widget.clear_mask_requested.connect(
            self.controller.clear_mdi_mask
        )
        self.mdi_widget.brush_spin.valueChanged.connect(
            self._on_brush_size_changed
        )

        # Keyboard shortcuts
        self._bind_shortcuts()

        # Populate panel combo
        self.nav_widget.populate_panel_combo(state.panel_names())

        # Show initial neuron
        self._show_current()

    def _show_current(self):
        """Display the current neuron from state cursor."""
        panel_name = self.state.current_panel_name()
        neuron_idx = self.state.current_neuron_idx

        if panel_name is None:
            return

        # Load and set slider values for this panel
        params = self.state.get_params_for_panel(panel_name)
        self.param_widget.set_params(params)

        # Show in viewer
        self.controller.show_neuron(panel_name, neuron_idx)

        # Update navigation display
        self._update_nav_display()

    def _update_nav_display(self):
        """Refresh the navigation widget with current state."""
        panel_name = self.state.current_panel_name()
        neuron_idx = self.state.current_neuron_idx

        if panel_name is None:
            return

        neuron_data = self.state.neuron_data(panel_name, neuron_idx)
        neuron_count = self.state.neuron_count(panel_name)
        progress = self.state.progress_summary()

        self.nav_widget.update_display(
            panel_name, neuron_idx, neuron_data,
            neuron_count, progress
        )

    def _save_notes(self):
        """Save current notes text."""
        panel_name = self.state.current_panel_name()
        neuron_idx = self.state.current_neuron_idx
        if panel_name:
            notes = self.nav_widget.get_notes()
            self.state.set_notes(panel_name, neuron_idx, notes)

    def _on_next(self):
        """Navigate to next neuron."""
        self._save_notes()
        result = self.state.next_neuron()
        if result:
            self._show_current()

    def _on_prev(self):
        """Navigate to previous neuron."""
        self._save_notes()
        result = self.state.prev_neuron()
        if result:
            self._show_current()

    def _set_qc(self, flag):
        """Set QC flag for current neuron and auto-advance."""
        panel_name = self.state.current_panel_name()
        neuron_idx = self.state.current_neuron_idx
        if panel_name:
            self._save_notes()
            self.state.set_qc_flag(panel_name, neuron_idx, flag)
            self._update_nav_display()
            # Auto-advance to next neuron after flagging
            self._on_next()

    def _on_jump(self, panel_name, neuron_idx):
        """Jump to a specific panel."""
        self._save_notes()
        names = self.state.panel_names()
        if panel_name in names:
            panel_idx = names.index(panel_name)
            self.state.jump_to(panel_idx, neuron_idx)
            self._show_current()

    def _on_adjust_start(self):
        """Enter start-point adjustment mode."""
        self.nav_widget.start_status_label.setText(
            "Click on the axon where the start should be..."
        )
        self.nav_widget.adjust_start_btn.setEnabled(False)

        # Store a reference so we can detect when adjustment completes
        original_callback_count = len(self.viewer.mouse_drag_callbacks)

        self.controller.enter_adjust_start_mode()

        # Poll for completion — when the mouse callback is removed,
        # the adjustment is done. We use a simple timer.
        from qtpy.QtCore import QTimer

        def _check_done():
            if not self.controller._adjusting_start:
                timer.stop()
                self.nav_widget.adjust_start_btn.setEnabled(True)
                self.nav_widget.start_status_label.setText("")
                self._update_nav_display()

        timer = QTimer()
        timer.timeout.connect(_check_done)
        timer.start(200)  # check every 200ms
        self._adjust_timer = timer  # prevent garbage collection

    def _on_reset_start(self):
        """Reset axon start to original pipeline result."""
        self.controller.cancel_adjust_mode()
        self.controller.reset_start_point()
        self.nav_widget.start_status_label.setText("")
        self.nav_widget.adjust_start_btn.setEnabled(True)
        self._update_nav_display()

    def _on_rerun(self):
        """Re-run pipeline for current panel with slider params."""
        panel_name = self.state.current_panel_name()
        if not panel_name:
            return

        params = self.param_widget.get_params()
        self.param_widget.status_label.setText("Reprocessing...")
        self.param_widget.rerun_btn.setEnabled(False)

        try:
            # Store param overrides
            self.state.set_param_overrides(panel_name, params)

            # Re-run pipeline
            reprocess_panel(self.state, panel_name)

            # Reload current panel at neuron 0
            names = self.state.panel_names()
            panel_idx = names.index(panel_name)
            self.state.jump_to(panel_idx, 0)

            # Force reload
            self.controller._current_panel = None
            self._show_current()

            self.param_widget.status_label.setText("Done!")
        except Exception as e:
            self.param_widget.status_label.setText(f"Error: {e}")
        finally:
            self.param_widget.rerun_btn.setEnabled(True)

    def _on_export(self):
        """Export lab analysis CSV and ML training dataset."""
        # Save current state first
        self._save_notes()
        self.controller.save_current_mdi_mask()

        # Ask where to export
        output_dir = QFileDialog.getExistingDirectory(
            None, "Select Export Folder"
        )
        if not output_dir:
            return

        # Show progress
        progress = self.state.progress_summary()
        total = progress["total_neurons"]
        accepted = progress["accepted"]
        modified = progress["modified"]
        rejected = progress["rejected"]
        pending = progress["pending"]
        with_mdi = progress["with_mdi_mask"]

        # Derive CSV name for display
        source_tif = self.state._state.get("source_tif", "unknown")
        from pathlib import Path as _P
        tif_stem = _P(source_tif).stem
        csv_name = f"{tif_stem}_MDI_analysis.csv"

        # Confirm with user
        reply = QMessageBox.question(
            None, "Export Dataset",
            f"Export summary:\n\n"
            f"  Total neurons: {total}\n"
            f"  Accepted: {accepted}\n"
            f"  Modified: {modified}\n"
            f"  Rejected: {rejected} (excluded)\n"
            f"  Pending: {pending} (excluded)\n"
            f"  With MDI masks: {with_mdi}\n\n"
            f"Exporting {accepted + modified} neurons to:\n{output_dir}\n\n"
            f"Outputs:\n"
            f"  {csv_name} — lab analysis results\n"
            f"  ml_dataset/ — training image/mask pairs\n\n"
            f"Proceed?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )

        if reply != QMessageBox.Yes:
            return

        try:
            analysis_df = export_analysis_csv(self.state, output_dir)
            ml_df = export_ml_dataset(self.state, output_dir, crop_padding=50)

            QMessageBox.information(
                None, "Export Complete",
                f"Export finished!\n\n"
                f"Lab Analysis ({len(analysis_df)} neurons):\n"
                f"  {csv_name}\n"
                f"  Columns: axon length, MDI area, MDI ratio, etc.\n\n"
                f"ML Dataset ({len(ml_df)} crops):\n"
                f"  ml_dataset/images/  — cropped neuron images\n"
                f"  ml_dataset/masks/   — neuron + MDI masks\n"
                f"  ml_dataset/axon_paths/ — path coordinates\n"
                f"  ml_dataset/manifest.csv"
            )
        except Exception as e:
            QMessageBox.critical(
                None, "Export Failed",
                f"Error during export:\n{e}"
            )

    def _on_reset_params(self):
        """Reset params display (sliders already reset by widget)."""
        self.param_widget.status_label.setText("Params reset to defaults")

    def _on_brush_size_changed(self, size):
        """Update MDI mask layer brush size."""
        if self.controller.mdi_mask_layer is not None:
            self.controller.mdi_mask_layer.brush_size = size

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        @self.viewer.bind_key("a")
        def _accept(viewer):
            self._set_qc("accepted")

        @self.viewer.bind_key("r")
        def _reject(viewer):
            self._set_qc("rejected")

        @self.viewer.bind_key("n")
        def _next(viewer):
            self._on_next()

        @self.viewer.bind_key("p")
        def _prev(viewer):
            self._on_prev()


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    """Launch the Review GUI."""
    viewer = napari.Viewer(title="Axon Analysis Review")

    # Show startup dialog
    dialog = StartupDialog()
    result = dialog.exec_()

    if result != QDialog.Accepted:
        viewer.close()
        return

    state = ProjectState(dialog.project_dir)

    if dialog.is_new:
        # Create new project and batch process with progress dialog
        state.create_new(dialog.source_tif, dialog.grid_rows, dialog.grid_cols)

        progress_dlg = QProgressDialog(
            "Processing panels...", "Cancel", 0, 100,
        )
        progress_dlg.setWindowTitle("Batch Processing")
        progress_dlg.setWindowModality(Qt.WindowModal)
        progress_dlg.setMinimumWidth(400)
        progress_dlg.setAutoClose(True)
        progress_dlg.show()

        def _progress_cb(idx, total, panel_name):
            progress_dlg.setMaximum(total)
            progress_dlg.setValue(idx)
            progress_dlg.setLabelText(
                f"Processing panel {idx + 1} / {total}\n{panel_name}"
            )
            QApplication.processEvents()

        batch_process(state, progress_callback=_progress_cb)
        progress_dlg.setValue(progress_dlg.maximum())

    else:
        state.load()

    # Launch the review app
    app = ReviewApp(viewer, state)

    napari.run()


if __name__ == "__main__":
    main()
