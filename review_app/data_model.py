"""
Project state management for the Axon Analysis Review GUI.

Stores all review state (QC flags, param overrides, cursor position)
in a single state.json file that persists across sessions.
"""

import json
import datetime
import numpy as np
from pathlib import Path


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types when serialising to JSON."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Default pipeline parameters
DEFAULT_PARAMS = {
    "nucleus": {
        "min_size": 100,
        "blur_sigma": 10,
        "hrp_threshold_factor": 2.0,
    },
    "trace": {
        "channel": 0,
        "threshold_pct": 0.10,
        "min_reach_ratio": 6.0,
        "border_margin": 350,
        "min_mask_size": 90000,
        "min_seed_intensity": 15,
        "seed_search_radius": 50,
    },
    "axon": {
        "channel": 0,
        "intensity_weight": 2.0,
        "nucleus_snap_distance": 30,
    },
}

SCALE_PX_PER_UM = 29.21


class ProjectState:
    """
    Manages the review project state backed by state.json.

    Tracks which panels/neurons have been reviewed, QC flags,
    parameter overrides, and the reviewer's cursor position.
    """

    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)
        self.state_file = self.project_dir / "state.json"
        self.panels_dir = self.project_dir / "panels"
        self.results_dir = self.project_dir / "results"
        self._state = {}

    # ── Persistence ──────────────────────────────────────────────

    def create_new(self, source_tif, grid_rows, grid_cols):
        """Initialise a new project."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.panels_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        self._state = {
            "project_name": self.project_dir.name,
            "created": datetime.datetime.now().isoformat(),
            "scale_px_per_um": SCALE_PX_PER_UM,
            "source_tif": str(source_tif),
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "default_params": DEFAULT_PARAMS,
            "panels": {},
            "current_panel_idx": 0,
            "current_neuron_idx": 0,
        }
        self.save()

    def load(self):
        """Load state from disk."""
        with open(self.state_file, "r") as f:
            self._state = json.load(f)

    def save(self):
        """Write state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2, cls=_NumpyEncoder)

    def exists(self):
        """Check if a project already exists at this path."""
        return self.state_file.exists()

    # ── Panel management ─────────────────────────────────────────

    def add_panel(self, panel_name, neuron_data):
        """
        Register a panel and its neurons after batch processing.

        Parameters
        ----------
        panel_name : str
            Panel stem name, e.g. "PAVNCD1_R1_C1"
        neuron_data : list of dict
            One dict per neuron with keys: axon_length_px
        """
        neurons = {}
        for i, nd in enumerate(neuron_data):
            neurons[str(i)] = {
                "qc_flag": "pending",
                "axon_length_px": nd["axon_length_px"],
                "axon_length_um": nd["axon_length_px"] / SCALE_PX_PER_UM,
                "has_mdi_mask": False,
                "notes": "",
            }

        self._state["panels"][panel_name] = {
            "status": "pending",
            "param_overrides": {},
            "neurons": neurons,
        }
        self.save()

    def panel_names(self):
        """Return sorted list of panel names."""
        return sorted(self._state.get("panels", {}).keys())

    def panel_data(self, panel_name):
        """Return the panel dict."""
        return self._state["panels"].get(panel_name, {})

    def neuron_count(self, panel_name):
        """Number of neurons in a panel."""
        panel = self._state["panels"].get(panel_name, {})
        return len(panel.get("neurons", {}))

    def neuron_data(self, panel_name, neuron_idx):
        """Return neuron dict."""
        panel = self._state["panels"].get(panel_name, {})
        return panel.get("neurons", {}).get(str(neuron_idx), {})

    # ── Navigation ───────────────────────────────────────────────

    @property
    def current_panel_idx(self):
        return self._state.get("current_panel_idx", 0)

    @property
    def current_neuron_idx(self):
        return self._state.get("current_neuron_idx", 0)

    def current_panel_name(self):
        """Return the name of the current panel."""
        names = self.panel_names()
        if not names:
            return None
        idx = min(self.current_panel_idx, len(names) - 1)
        return names[idx]

    def jump_to(self, panel_idx, neuron_idx):
        """Set the cursor position."""
        self._state["current_panel_idx"] = panel_idx
        self._state["current_neuron_idx"] = neuron_idx
        self.save()

    def next_neuron(self):
        """
        Advance to the next neuron. Returns (panel_name, neuron_idx)
        or None if at the end.
        """
        names = self.panel_names()
        if not names:
            return None

        p_idx = self.current_panel_idx
        n_idx = self.current_neuron_idx + 1

        while p_idx < len(names):
            panel_name = names[p_idx]
            n_count = self.neuron_count(panel_name)

            if n_count == 0:
                # Panel with no neurons — skip
                p_idx += 1
                n_idx = 0
                continue

            if n_idx < n_count:
                self.jump_to(p_idx, n_idx)
                return (panel_name, n_idx)

            # Move to next panel
            p_idx += 1
            n_idx = 0

        return None  # Reached the end

    def prev_neuron(self):
        """
        Go back to the previous neuron. Returns (panel_name, neuron_idx)
        or None if at the start.
        """
        names = self.panel_names()
        if not names:
            return None

        p_idx = self.current_panel_idx
        n_idx = self.current_neuron_idx - 1

        while p_idx >= 0:
            if n_idx >= 0:
                panel_name = names[p_idx]
                if self.neuron_count(panel_name) > 0:
                    self.jump_to(p_idx, n_idx)
                    return (panel_name, n_idx)

            # Move to previous panel, last neuron
            p_idx -= 1
            if p_idx >= 0:
                n_idx = self.neuron_count(names[p_idx]) - 1

        return None  # At the start

    # ── Annotation ───────────────────────────────────────────────

    def set_qc_flag(self, panel_name, neuron_idx, flag):
        """Set QC flag for a neuron. flag: 'accepted', 'rejected', 'modified'."""
        neuron = self._state["panels"][panel_name]["neurons"][str(neuron_idx)]
        neuron["qc_flag"] = flag
        self.save()

    def set_mdi_mask_flag(self, panel_name, neuron_idx, has_mask):
        """Mark whether a neuron has an MDI mask painted."""
        neuron = self._state["panels"][panel_name]["neurons"][str(neuron_idx)]
        neuron["has_mdi_mask"] = has_mask
        self.save()

    def set_notes(self, panel_name, neuron_idx, notes):
        """Set reviewer notes for a neuron."""
        neuron = self._state["panels"][panel_name]["neurons"][str(neuron_idx)]
        neuron["notes"] = notes
        self.save()

    def set_start_trim(self, panel_name, neuron_idx, trim_index, new_length_px):
        """
        Store a custom axon start trim for a neuron.

        Parameters
        ----------
        trim_index : int
            Index into the original axon path where the user placed the
            new start point. The displayed path becomes path[:trim_index+1].
        new_length_px : float
            Recalculated axon length after trimming.
        """
        neuron = self._state["panels"][panel_name]["neurons"][str(neuron_idx)]
        neuron["start_trim_index"] = trim_index
        neuron["axon_length_px"] = new_length_px
        neuron["axon_length_um"] = new_length_px / SCALE_PX_PER_UM
        self.save()

    def clear_start_trim(self, panel_name, neuron_idx, original_length_px):
        """Remove the custom start trim, restoring the original path."""
        neuron = self._state["panels"][panel_name]["neurons"][str(neuron_idx)]
        neuron.pop("start_trim_index", None)
        neuron["axon_length_px"] = original_length_px
        neuron["axon_length_um"] = original_length_px / SCALE_PX_PER_UM
        self.save()

    def set_param_overrides(self, panel_name, params):
        """Store parameter overrides for a panel."""
        self._state["panels"][panel_name]["param_overrides"] = params
        self.save()

    def get_params_for_panel(self, panel_name):
        """
        Return effective params for a panel (defaults + overrides).
        """
        import copy
        params = copy.deepcopy(self._state.get("default_params", DEFAULT_PARAMS))
        overrides = self._state["panels"].get(panel_name, {}).get(
            "param_overrides", {}
        )
        for section, values in overrides.items():
            if section in params:
                params[section].update(values)
        return params

    def mark_panel_reviewed(self, panel_name):
        """Mark a panel as fully reviewed."""
        self._state["panels"][panel_name]["status"] = "reviewed"
        self.save()

    # ── Progress ─────────────────────────────────────────────────

    def progress_summary(self):
        """
        Return progress stats.

        Returns dict with: total_neurons, accepted, rejected, modified,
                          pending, total_panels, panels_reviewed
        """
        stats = {
            "total_neurons": 0,
            "accepted": 0,
            "rejected": 0,
            "modified": 0,
            "pending": 0,
            "with_mdi_mask": 0,
            "total_panels": 0,
            "panels_reviewed": 0,
        }

        for panel_name, panel in self._state.get("panels", {}).items():
            stats["total_panels"] += 1
            if panel.get("status") == "reviewed":
                stats["panels_reviewed"] += 1

            for neuron_idx, neuron in panel.get("neurons", {}).items():
                stats["total_neurons"] += 1
                flag = neuron.get("qc_flag", "pending")
                if flag in stats:
                    stats[flag] += 1
                if neuron.get("has_mdi_mask"):
                    stats["with_mdi_mask"] += 1

        return stats

    # ── Paths ────────────────────────────────────────────────────

    def panel_image_path(self, panel_name):
        """Path to the panel TIF."""
        return self.panels_dir / f"{panel_name}.tif"

    def panel_results_dir(self, panel_name):
        """Directory for a panel's pipeline results."""
        return self.results_dir / panel_name

    def pipeline_cache_path(self, panel_name):
        """Path to the cached pipeline.npz for a panel."""
        return self.panel_results_dir(panel_name) / "pipeline.npz"

    def neuron_dir(self, panel_name, neuron_idx):
        """Directory for a specific neuron's data."""
        return self.panel_results_dir(panel_name) / f"neuron_{neuron_idx:03d}"

    def mdi_mask_path(self, panel_name, neuron_idx):
        """Path to the MDI mask PNG for a neuron."""
        return self.neuron_dir(panel_name, neuron_idx) / "mdi_mask.png"
