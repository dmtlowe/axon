"""
Batch processing for the Review GUI.

Runs the existing pipeline (panel split → nucleus detection → neuron
tracing → axon measurement) over an entire dataset and caches all
intermediate results as pipeline.npz files for fast GUI loading.
"""

import sys
import numpy as np
from pathlib import Path
from skimage import io

# Add parent dir so we can import the pipeline modules
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from panel_splitter import split_panels
from nucleus_detector import detect_nuclei
from neuron_trace import trace_all_neurons
from measure_axon import measure_axon


def batch_process(state, progress_callback=None):
    """
    Run full pipeline on all panels and cache results.

    Parameters
    ----------
    state : ProjectState
        Initialised project state (must have source_tif, grid info set).
    progress_callback : callable, optional
        Called with (panel_idx, total_panels, panel_name) for progress.
    """
    source_tif = state._state["source_tif"]
    grid_rows = state._state["grid_rows"]
    grid_cols = state._state["grid_cols"]
    params = state._state.get("default_params", {})

    # Step 0: split panels
    split_panels(source_tif, grid_rows, grid_cols,
                 output_dir=str(state.panels_dir))

    # Get list of panel files
    panel_files = sorted(state.panels_dir.glob("*.tif"))
    total = len(panel_files)
    print(f"Processing {total} panels...")

    for idx, panel_path in enumerate(panel_files):
        panel_name = panel_path.stem

        if progress_callback:
            progress_callback(idx, total, panel_name)

        # Skip if already cached
        cache_path = state.pipeline_cache_path(panel_name)
        if cache_path.exists() and panel_name in state._state.get("panels", {}):
            print(f"  [{idx+1}/{total}] {panel_name} — cached, skipping")
            continue

        print(f"  [{idx+1}/{total}] {panel_name}")

        neuron_data = _process_single_panel(
            panel_path, params, state.panel_results_dir(panel_name)
        )

        # Register in state
        state.add_panel(panel_name, neuron_data)

    print(f"Batch processing complete. {total} panels processed.")


def reprocess_panel(state, panel_name):
    """
    Re-run pipeline for a single panel with its current params.

    Parameters
    ----------
    state : ProjectState
    panel_name : str

    Returns
    -------
    list of dict : neuron data (one per detected neuron)
    """
    panel_path = state.panel_image_path(panel_name)
    params = state.get_params_for_panel(panel_name)
    results_dir = state.panel_results_dir(panel_name)

    print(f"Reprocessing {panel_name}...")
    neuron_data = _process_single_panel(panel_path, params, results_dir)

    # Update state — re-register the panel (resets QC flags)
    state.add_panel(panel_name, neuron_data)

    print(f"  Reprocessed: {len(neuron_data)} neurons found")
    return neuron_data


def _process_single_panel(panel_path, params, results_dir):
    """
    Run pipeline on one panel and save pipeline.npz.

    Returns list of neuron_data dicts with axon_length_px.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    img = io.imread(str(panel_path))

    # Nucleus detection
    nuc_params = params.get("nucleus", {})
    centroids, labeled = detect_nuclei(
        img,
        min_size=nuc_params.get("min_size", 100),
        blur_sigma=nuc_params.get("blur_sigma", 10),
        hrp_threshold_factor=nuc_params.get("hrp_threshold_factor", 2.0),
    )

    if len(centroids) == 0:
        _save_empty_cache(results_dir, img.shape)
        return []

    # Neuron tracing
    trace_params = params.get("trace", {})
    traces, kept_centroids = trace_all_neurons(
        img, centroids, labeled,
        channel=trace_params.get("channel", 0),
        threshold_pct=trace_params.get("threshold_pct", 0.10),
        min_reach_ratio=trace_params.get("min_reach_ratio", 6.0),
        border_margin=trace_params.get("border_margin", 350),
        min_mask_size=trace_params.get("min_mask_size", 90000),
        min_seed_intensity=trace_params.get("min_seed_intensity", 15),
        seed_search_radius=trace_params.get("seed_search_radius", 50),
        verbose=False,
    )

    if len(traces) == 0:
        _save_empty_cache(results_dir, img.shape)
        return []

    # Axon measurement for each neuron
    axon_params = params.get("axon", {})
    neuron_data = []
    trace_masks = []
    axon_paths = []
    axon_lengths = []
    tip_points = []
    start_points = []

    for i, (trace, centroid) in enumerate(zip(traces, kept_centroids)):
        label_at_seed = labeled[centroid[0], centroid[1]]
        if label_at_seed > 0:
            nuc_mask = labeled == label_at_seed
        else:
            h, w = labeled.shape
            yy, xx = np.ogrid[:h, :w]
            nuc_mask = ((yy - centroid[0])**2 + (xx - centroid[1])**2) < 15**2

        result = measure_axon(
            img,
            trace["mask"],
            centroid,
            nuc_mask,
            channel=axon_params.get("channel", 0),
            intensity_weight=axon_params.get("intensity_weight", 2.0),
            nucleus_snap_distance=axon_params.get("nucleus_snap_distance", 30),
        )

        trace_masks.append(trace["mask"].astype(np.uint8))
        axon_paths.append(np.array(result["path"]))
        axon_lengths.append(result["length_px"])
        tip_points.append(result["tip_point"])
        start_points.append(result["start_point"])

        neuron_data.append({
            "axon_length_px": result["length_px"],
        })

        # Create neuron subdirectory
        neuron_dir = results_dir / f"neuron_{i:03d}"
        neuron_dir.mkdir(exist_ok=True)

    # Save cache
    cache = {
        "centroids": np.array(centroids),
        "kept_centroids": np.array(kept_centroids),
        "labeled": labeled,
        "n_neurons": len(traces),
    }

    # Store per-neuron arrays with indexed keys
    for i in range(len(traces)):
        cache[f"trace_mask_{i}"] = trace_masks[i]
        cache[f"axon_path_{i}"] = axon_paths[i]
        cache[f"axon_length_{i}"] = np.array([axon_lengths[i]])
        cache[f"tip_point_{i}"] = np.array(tip_points[i])
        cache[f"start_point_{i}"] = np.array(start_points[i])

    np.savez_compressed(str(results_dir / "pipeline.npz"), **cache)

    return neuron_data


def _save_empty_cache(results_dir, img_shape):
    """Save a minimal cache for panels with no neurons."""
    np.savez_compressed(
        str(results_dir / "pipeline.npz"),
        centroids=np.array([]),
        kept_centroids=np.array([]),
        labeled=np.zeros(img_shape[:2], dtype=np.int32),
        n_neurons=np.array([0]),
    )


def load_pipeline_cache(state, panel_name):
    """
    Load cached pipeline results for a panel.

    Returns
    -------
    dict with: centroids, kept_centroids, labeled, n_neurons,
               and per-neuron: trace_mask_N, axon_path_N,
               axon_length_N, tip_point_N, start_point_N
    Returns None if cache doesn't exist.
    """
    cache_path = state.pipeline_cache_path(panel_name)
    if not cache_path.exists():
        return None

    data = dict(np.load(str(cache_path), allow_pickle=True))

    # Convert n_neurons to int (stored as array)
    n = data.get("n_neurons", 0)
    if hasattr(n, "__len__"):
        n = int(n.item()) if n.size == 1 else int(n[0]) if len(n) > 0 else 0
    data["n_neurons"] = n

    return data
