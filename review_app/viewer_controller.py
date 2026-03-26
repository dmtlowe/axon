"""
Napari viewer controller for the Review GUI.

Manages all napari layers and handles loading/displaying neuron data.
Layers are created once and their data is swapped when navigating.
"""

import numpy as np
from pathlib import Path
from skimage import io

from .batch_processor import load_pipeline_cache


# ── Colour palettes ──────────────────────────────────────────────
# Standard palette (default)
PALETTE_STANDARD = {
    "centroid": [0, 1, 1, 1],          # cyan
    "tip": [1, 0, 0, 1],              # red
    "start": [0, 0.5, 1, 1],          # blue
    "axon_path": [1.0, 1.0, 0.0, 1],    # bright yellow
    "neuron_mask_colormap": None,       # napari default labels colormap
    "mdi_mask_colormap": None,          # napari default labels colormap
}

# Colourblind-safe palette (Wong, 2011 — safe for protan/deutan/tritan)
# Uses blue, orange, yellow — avoids red-green contrast entirely
PALETTE_COLOURBLIND = {
    "centroid": [0.90, 0.62, 0, 1],    # orange
    "tip": [0.80, 0.47, 0.65, 1],     # reddish purple
    "start": [0, 0.45, 0.70, 1],      # blue
    "axon_path": [0.94, 0.89, 0.26],   # yellow
    "neuron_mask_colormap": None,       # labels colormaps handled separately
    "mdi_mask_colormap": None,
}


class ViewerController:
    """
    Bridge between ProjectState and napari viewer.

    Creates and manages 5 layer types:
    - Image: panel RGB image
    - Labels (neuron mask): algorithm trace, read-only visual
    - Labels (MDI mask): user-painted MDI regions, editable
    - Shapes: axon path polyline
    - Points: centroid, tip, start markers
    """

    def __init__(self, viewer, state):
        """
        Parameters
        ----------
        viewer : napari.Viewer
        state : ProjectState
        """
        self.viewer = viewer
        self.state = state

        # Current state
        self._current_panel = None
        self._current_neuron = None
        self._cache = None  # loaded pipeline.npz data
        self._img = None    # current panel image

        # Full (untrimmed) axon path for the current neuron
        self._full_axon_path = None
        self._full_axon_length = None

        # Start-point adjustment mode
        self._adjusting_start = False
        self._mouse_callback = None

        # Colour palette
        self._colourblind = False
        self._palette = PALETTE_STANDARD

        # Layer references (created on first show_neuron call)
        self.img_layer = None
        self.red_layer = None
        self.green_layer = None
        self.blue_layer = None
        self.grey_layer = None
        self.neuron_mask_layer = None
        self.mdi_mask_layer = None
        self.axon_path_layer = None
        self.markers_layer = None

    def show_neuron(self, panel_name, neuron_idx):
        """
        Display a specific neuron in the viewer.

        Saves the current MDI mask first, then loads the new
        panel/neuron data and updates all layers.
        """
        # Save current MDI mask before switching
        self.save_current_mdi_mask()

        # Load panel data if switching panels
        if panel_name != self._current_panel:
            self._load_panel(panel_name)

        self._current_neuron = neuron_idx

        if self._cache is None or self._cache["n_neurons"] == 0:
            self._show_empty()
            return

        if neuron_idx >= self._cache["n_neurons"]:
            self._show_empty()
            return

        # Extract neuron-specific data from cache
        trace_mask = self._cache[f"trace_mask_{neuron_idx}"]
        axon_path = self._cache[f"axon_path_{neuron_idx}"]
        tip_point = self._cache[f"tip_point_{neuron_idx}"]
        original_start = self._cache[f"start_point_{neuron_idx}"]
        kept_centroids = self._cache["kept_centroids"]
        centroid = kept_centroids[neuron_idx] if neuron_idx < len(kept_centroids) else tip_point

        # Store full path for potential start-point adjustment
        self._full_axon_path = np.array(axon_path, dtype=float)
        self._full_axon_length = float(self._cache[f"axon_length_{neuron_idx}"][0])

        # Check if there's a custom start trim
        neuron_info = self.state.neuron_data(panel_name, neuron_idx)
        trim_idx = neuron_info.get("start_trim_index", None)

        if trim_idx is not None and trim_idx < len(axon_path):
            display_path = axon_path[:trim_idx + 1]
            start_point = display_path[-1]
        else:
            display_path = axon_path
            start_point = original_start

        # Exit adjust mode if it was active
        self._adjusting_start = False

        # Load existing MDI mask or create empty
        mdi_mask = self._load_mdi_mask(panel_name, neuron_idx)

        # Update layers
        self._update_layers(trace_mask, mdi_mask, display_path,
                            centroid, tip_point, start_point)

        # Auto-zoom to neuron region
        self._zoom_to_neuron(trace_mask, padding=100)

    def save_current_mdi_mask(self):
        """Save the current MDI mask to disk if one is being displayed."""
        if (self._current_panel is None or
                self._current_neuron is None or
                self.mdi_mask_layer is None):
            return

        mask_data = self.mdi_mask_layer.data
        has_mask = np.any(mask_data > 0)

        # Save mask
        mask_path = self.state.mdi_mask_path(
            self._current_panel, self._current_neuron
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        if has_mask:
            from skimage.io import imsave
            imsave(str(mask_path), mask_data.astype(np.uint8),
                   check_contrast=False)
        elif mask_path.exists():
            # Remove empty mask file
            mask_path.unlink()

        # Update state
        self.state.set_mdi_mask_flag(
            self._current_panel, self._current_neuron, has_mask
        )

    def clear_mdi_mask(self):
        """Clear the current MDI mask."""
        if self.mdi_mask_layer is not None:
            self.mdi_mask_layer.data = np.zeros_like(self.mdi_mask_layer.data)

    @property
    def current_panel(self):
        return self._current_panel

    @property
    def current_neuron(self):
        return self._current_neuron

    # ── Colourblind mode ────────────────────────────────────────

    def set_colourblind_mode(self, enabled):
        """
        Toggle colourblind-safe colour palette.

        Immediately recolours all existing layers.
        """
        self._colourblind = enabled
        self._palette = PALETTE_COLOURBLIND if enabled else PALETTE_STANDARD
        self._apply_palette()

    def set_channel_visibility(self, channel, visible):
        """
        Toggle individual channel layer visibility.

        Parameters
        ----------
        channel : str
            One of "red", "green", "blue", "grey", "composite".
        visible : bool
        """
        layer_map = {
            "red": self.red_layer,
            "green": self.green_layer,
            "blue": self.blue_layer,
            "grey": self.grey_layer,
            "composite": self.img_layer,
        }
        layer = layer_map.get(channel)
        if layer is not None:
            layer.visible = visible

    def _apply_palette(self):
        """Apply the current colour palette to all layers."""
        p = self._palette

        # Axon path
        if self.axon_path_layer is not None and len(self.axon_path_layer.data) > 0:
            self.axon_path_layer.edge_color = p["axon_path"]

        # Marker points
        if self.markers_layer is not None and len(self.markers_layer.data) > 0:
            colours = np.array([p["centroid"], p["tip"], p["start"]])
            self.markers_layer.face_color = colours

    # ── Start-point adjustment ───────────────────────────────────

    def enter_adjust_start_mode(self):
        """
        Enter mode where the next click on the image sets a new
        axon start point. The existing path is trimmed to the
        nearest point to the click location.
        """
        if self._full_axon_path is None or len(self._full_axon_path) < 2:
            return

        self._adjusting_start = True

        # Register mouse click callback
        @self.viewer.mouse_drag_callbacks.append
        def _on_click(viewer, event):
            if not self._adjusting_start:
                return
            if event.type != "mouse_press":
                return

            # Get click position in data coordinates
            # event.position is (y, x) in data coords
            click_pos = np.array(event.position[-2:])  # last 2 dims = y, x

            self._apply_start_trim(click_pos)

            # Remove this callback
            self._adjusting_start = False
            if _on_click in viewer.mouse_drag_callbacks:
                viewer.mouse_drag_callbacks.remove(_on_click)

        self._mouse_callback = _on_click

    def _apply_start_trim(self, click_pos):
        """Find nearest path point to click and trim the axon there."""
        path = self._full_axon_path
        if path is None or len(path) < 2:
            return

        # Find closest point on path to click
        dists = np.sqrt(np.sum((path - click_pos) ** 2, axis=1))
        trim_idx = int(np.argmin(dists))

        # Don't allow trimming to less than 10% of path
        if trim_idx < len(path) * 0.1:
            trim_idx = max(2, trim_idx)

        # Trim path (path is tip→soma, so trim_idx cuts from soma end)
        trimmed_path = path[:trim_idx + 1]

        # Recalculate length
        diffs = np.diff(trimmed_path, axis=0)
        new_length = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))

        # Update display
        new_start = trimmed_path[-1]
        tip_point = self._cache[f"tip_point_{self._current_neuron}"]
        kept = self._cache["kept_centroids"]
        centroid = (kept[self._current_neuron]
                    if self._current_neuron < len(kept)
                    else tip_point)
        trace_mask = self._cache[f"trace_mask_{self._current_neuron}"]
        mdi_mask = self.mdi_mask_layer.data if self.mdi_mask_layer else None

        self._update_layers(trace_mask, mdi_mask, trimmed_path,
                            centroid, tip_point, new_start)

        # Save trim to state
        self.state.set_start_trim(
            self._current_panel, self._current_neuron,
            trim_idx, new_length
        )

        return new_length

    def reset_start_point(self):
        """Reset axon start to the original pipeline result."""
        if (self._current_panel is None or
                self._current_neuron is None or
                self._full_axon_path is None):
            return

        # Restore original length and clear trim
        self.state.clear_start_trim(
            self._current_panel, self._current_neuron,
            self._full_axon_length
        )

        # Redisplay with full path
        original_start = self._cache[f"start_point_{self._current_neuron}"]
        tip_point = self._cache[f"tip_point_{self._current_neuron}"]
        kept = self._cache["kept_centroids"]
        centroid = (kept[self._current_neuron]
                    if self._current_neuron < len(kept)
                    else tip_point)
        trace_mask = self._cache[f"trace_mask_{self._current_neuron}"]
        mdi_mask = self.mdi_mask_layer.data if self.mdi_mask_layer else None

        self._update_layers(trace_mask, mdi_mask, self._full_axon_path,
                            centroid, tip_point, original_start)

        return self._full_axon_length

    def cancel_adjust_mode(self):
        """Cancel start-point adjustment without changing anything."""
        self._adjusting_start = False
        if (self._mouse_callback is not None and
                self._mouse_callback in self.viewer.mouse_drag_callbacks):
            self.viewer.mouse_drag_callbacks.remove(self._mouse_callback)
            self._mouse_callback = None

    # ── Private helpers ──────────────────────────────────────────

    def _load_panel(self, panel_name):
        """Load panel image and pipeline cache."""
        self._current_panel = panel_name

        img_path = self.state.panel_image_path(panel_name)
        if img_path.exists():
            self._img = io.imread(str(img_path))
        else:
            self._img = None

        self._cache = load_pipeline_cache(self.state, panel_name)

    def _load_mdi_mask(self, panel_name, neuron_idx):
        """Load existing MDI mask or return zeros."""
        mask_path = self.state.mdi_mask_path(panel_name, neuron_idx)
        if mask_path.exists():
            from skimage.io import imread
            return imread(str(mask_path))

        if self._img is not None:
            h, w = self._img.shape[:2]
        else:
            h, w = 512, 512
        return np.zeros((h, w), dtype=np.uint8)

    def _update_layers(self, trace_mask, mdi_mask, axon_path,
                       centroid, tip_point, start_point):
        """Create or update all napari layers."""
        img = self._img

        # Image layer (composite RGB)
        if self.img_layer is None:
            self.img_layer = self.viewer.add_image(
                img, name="Panel Image", visible=False
            )
        else:
            self.img_layer.data = img

        # ── Channel split layers (hidden by default) ────────────
        if img is not None and img.ndim == 3 and img.shape[2] >= 3:
            red_ch = img[:, :, 0]
            green_ch = img[:, :, 1]
            blue_ch = img[:, :, 2]
            grey_ch = np.mean(img[:, :, :3], axis=2).astype(img.dtype)

            if self.red_layer is None:
                self.red_layer = self.viewer.add_image(
                    red_ch, name="Red (Tubulin)",
                    colormap="red", blending="additive",
                    visible=True,
                )
                self.green_layer = self.viewer.add_image(
                    green_ch, name="Green (HRP)",
                    colormap="green", blending="additive",
                    visible=True,
                )
                self.blue_layer = self.viewer.add_image(
                    blue_ch, name="Blue (DAPI)",
                    colormap="blue", blending="additive",
                    visible=True,
                )
                self.grey_layer = self.viewer.add_image(
                    grey_ch, name="Greyscale",
                    colormap="gray",
                    visible=False,
                )
            else:
                self.red_layer.data = red_ch
                self.green_layer.data = green_ch
                self.blue_layer.data = blue_ch
                self.grey_layer.data = grey_ch

        # Neuron mask layer (read-only)
        mask_display = trace_mask.astype(np.int32)
        if self.neuron_mask_layer is None:
            self.neuron_mask_layer = self.viewer.add_labels(
                mask_display, name="Neuron Mask",
                opacity=0.15, visible=False
            )
        else:
            self.neuron_mask_layer.data = mask_display
        # Keep it non-editable
        self.neuron_mask_layer.editable = False

        # MDI mask layer (editable) — turquoise fill
        from napari.utils.colormaps import DirectLabelColormap
        _mdi_cmap = DirectLabelColormap(
            color_dict={None: (0, 0, 0, 0), 0: (0, 0, 0, 0),
                        1: (0.0, 0.8, 0.8, 1.0)},  # turquoise
        )
        if self.mdi_mask_layer is None:
            self.mdi_mask_layer = self.viewer.add_labels(
                mdi_mask.astype(np.int32), name="MDI Mask",
                opacity=0.35, visible=True,
                colormap=_mdi_cmap,
            )
        else:
            self.mdi_mask_layer.data = mdi_mask.astype(np.int32)
            self.mdi_mask_layer.colormap = _mdi_cmap
        self.mdi_mask_layer.contour = 0  # filled, no outline-only
        self.mdi_mask_layer.editable = True
        self.mdi_mask_layer.selected_label = 1
        self.mdi_mask_layer.brush_size = 15

        # Axon path as shapes layer
        p = self._palette
        if len(axon_path) > 1:
            # Convert (y,x) path to napari shapes format
            path_coords = np.array(axon_path, dtype=float)
            shapes_data = [path_coords]
            shape_types = ["path"]

            if self.axon_path_layer is None:
                self.axon_path_layer = self.viewer.add_shapes(
                    shapes_data, shape_type=shape_types,
                    edge_color=p["axon_path"], edge_width=2,
                    face_color="transparent",
                    name="Axon Path", visible=True
                )
            else:
                self.axon_path_layer.data = shapes_data
                self.axon_path_layer.shape_type = shape_types
                self.axon_path_layer.edge_color = p["axon_path"]
        elif self.axon_path_layer is not None:
            self.axon_path_layer.data = []

        # Points layer for markers
        points = np.array([centroid, tip_point, start_point], dtype=float)
        colours = np.array([p["centroid"], p["tip"], p["start"]])
        symbols = ["disc", "cross", "cross"]
        sizes = [12, 14, 14]

        if self.markers_layer is None:
            self.markers_layer = self.viewer.add_points(
                points, name="Markers",
                face_color=colours, symbol=symbols,
                size=sizes, visible=True
            )
        else:
            self.markers_layer.data = points
            self.markers_layer.face_color = colours
            self.markers_layer.symbol = symbols
            self.markers_layer.size = sizes

        self.markers_layer.editable = False

    def _show_empty(self):
        """Display empty state when no neurons are available."""
        if self._img is not None:
            if self.img_layer is None:
                self.img_layer = self.viewer.add_image(
                    self._img, name="Panel Image"
                )
            else:
                self.img_layer.data = self._img

        # Clear other layers
        h, w = self._img.shape[:2] if self._img is not None else (512, 512)
        empty = np.zeros((h, w), dtype=np.int32)
        empty_8 = np.zeros((h, w), dtype=np.uint8)

        if self.red_layer is not None:
            self.red_layer.data = empty_8
        if self.green_layer is not None:
            self.green_layer.data = empty_8
        if self.blue_layer is not None:
            self.blue_layer.data = empty_8
        if self.grey_layer is not None:
            self.grey_layer.data = empty_8
        if self.neuron_mask_layer is not None:
            self.neuron_mask_layer.data = empty
        if self.mdi_mask_layer is not None:
            self.mdi_mask_layer.data = empty
        if self.axon_path_layer is not None:
            self.axon_path_layer.data = []
        if self.markers_layer is not None:
            self.markers_layer.data = np.empty((0, 2))

    def _zoom_to_neuron(self, trace_mask, padding=100):
        """Auto-zoom the camera to center on the neuron."""
        ys, xs = np.where(trace_mask > 0)
        if len(ys) == 0:
            return

        y_min, y_max = ys.min() - padding, ys.max() + padding
        x_min, x_max = xs.min() - padding, xs.max() + padding

        # Clamp to image bounds
        if self._img is not None:
            h, w = self._img.shape[:2]
            y_min, y_max = max(0, y_min), min(h, y_max)
            x_min, x_max = max(0, x_min), min(w, x_max)

        centre = ((y_min + y_max) / 2, (x_min + x_max) / 2)
        extent = max(y_max - y_min, x_max - x_min)

        self.viewer.camera.center = centre
        self.viewer.camera.zoom = min(
            self.viewer.window.qt_viewer.canvas.size[0],
            self.viewer.window.qt_viewer.canvas.size[1],
        ) / max(extent, 1)
