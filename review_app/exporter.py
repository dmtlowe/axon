"""
Export functions for the Axon Analysis Review GUI.

Two export targets:
1. Lab analysis CSV — MDI area, axon length, ratio for quantification
2. ML dataset — paired image/mask crops for segmentation model training
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io as skio

from .data_model import ProjectState, SCALE_PX_PER_UM
from .batch_processor import load_pipeline_cache


# ─────────────────────────────────────────────────────────────────
# Goal 1: Lab Analysis CSV
# ─────────────────────────────────────────────────────────────────

def export_analysis_csv(state, output_dir):
    """
    Export quantification results as {TIF_stem}_MDI_analysis.csv.

    Computes MDI mask area from saved PNGs and combines with
    axon length measurements. Only includes accepted + modified neurons.

    Parameters
    ----------
    state : ProjectState
        Loaded project state.
    output_dir : str or Path
        Where to write the CSV.

    Returns
    -------
    pd.DataFrame
        The exported analysis data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive filename from source TIF
    source_tif = state._state.get("source_tif", "unknown")
    tif_stem = Path(source_tif).stem
    csv_name = f"{tif_stem}_MDI_analysis.csv"

    rows = []

    for panel_name in state.panel_names():
        panel_data = state.panel_data(panel_name)
        neurons = panel_data.get("neurons", {})
        if not neurons:
            continue

        # Load pipeline cache for coordinate data
        cache = load_pipeline_cache(state, panel_name)
        if cache is None:
            continue

        for neuron_idx_str, neuron_info in neurons.items():
            neuron_idx = int(neuron_idx_str)
            qc_flag = neuron_info.get("qc_flag", "pending")

            # Only export accepted + modified
            if qc_flag not in ("accepted", "modified"):
                continue

            # Axon length
            axon_length_px = float(neuron_info.get("axon_length_px", 0))
            axon_length_um = axon_length_px / SCALE_PX_PER_UM

            # MDI mask area
            mdi_mask_path = state.mdi_mask_path(panel_name, neuron_idx)
            if mdi_mask_path.exists():
                mdi_mask = skio.imread(str(mdi_mask_path))
                mdi_area_px = int(np.count_nonzero(mdi_mask))
            else:
                mdi_area_px = 0

            mdi_area_um2 = mdi_area_px / (SCALE_PX_PER_UM ** 2)
            mdi_ratio = (mdi_area_um2 / axon_length_um
                         if axon_length_um > 0 else 0.0)

            # Coordinates from cache
            tip_key = f"tip_point_{neuron_idx}"
            start_key = f"start_point_{neuron_idx}"
            tip_point = cache.get(tip_key, np.array([0, 0]))
            start_point = cache.get(start_key, np.array([0, 0]))

            kept = cache.get("kept_centroids", np.array([]))
            if neuron_idx < len(kept):
                centroid = kept[neuron_idx]
            else:
                centroid = np.array([0, 0])

            rows.append({
                "panel": panel_name,
                "neuron_idx": neuron_idx,
                "qc_flag": qc_flag,
                "axon_length_px": axon_length_px,
                "axon_length_um": round(axon_length_um, 2),
                "mdi_area_px": mdi_area_px,
                "mdi_area_um2": round(mdi_area_um2, 4),
                "mdi_ratio": round(mdi_ratio, 6),
                "has_mdi_mask": mdi_area_px > 0,
                "notes": neuron_info.get("notes", ""),
                "centroid_y": float(centroid[0]),
                "centroid_x": float(centroid[1]),
                "tip_y": float(tip_point[0]),
                "tip_x": float(tip_point[1]),
                "start_y": float(start_point[0]),
                "start_x": float(start_point[1]),
            })

    df = pd.DataFrame(rows)
    csv_path = output_dir / csv_name
    df.to_csv(str(csv_path), index=False)

    print(f"\nAnalysis CSV: {csv_path}  ({len(df)} neurons)")
    return df


# ─────────────────────────────────────────────────────────────────
# Goal 2: ML Training Dataset
# ─────────────────────────────────────────────────────────────────

def export_ml_dataset(state, output_dir, crop_padding=50):
    """
    Export paired image/mask crops for ML training.

    Produces cropped neuron images with matching neuron masks,
    MDI masks, and axon path coordinates.

    Parameters
    ----------
    state : ProjectState
        Loaded project state.
    output_dir : str or Path
        Root output directory. Files go into ml_dataset/ subfolder.
    crop_padding : int
        Padding around neuron bounding box in pixels.

    Returns
    -------
    pd.DataFrame
        The manifest data.
    """
    output_dir = Path(output_dir)
    ml_dir = output_dir / "ml_dataset"
    (ml_dir / "images").mkdir(parents=True, exist_ok=True)
    (ml_dir / "masks").mkdir(parents=True, exist_ok=True)
    (ml_dir / "axon_paths").mkdir(parents=True, exist_ok=True)

    rows = []
    exported = 0

    for panel_name in state.panel_names():
        panel_data = state.panel_data(panel_name)
        neurons = panel_data.get("neurons", {})
        if not neurons:
            continue

        # Load panel image
        img_path = state.panel_image_path(panel_name)
        if not img_path.exists():
            continue

        img = skio.imread(str(img_path))
        h, w = img.shape[:2]

        # Load pipeline cache
        cache = load_pipeline_cache(state, panel_name)
        if cache is None:
            continue

        for neuron_idx_str, neuron_info in neurons.items():
            neuron_idx = int(neuron_idx_str)
            qc_flag = neuron_info.get("qc_flag", "pending")

            # Only export accepted + modified
            if qc_flag not in ("accepted", "modified"):
                continue

            base_name = f"{panel_name}_neuron_{neuron_idx:03d}"

            # Get neuron data from cache
            mask_key = f"trace_mask_{neuron_idx}"
            path_key = f"axon_path_{neuron_idx}"
            if mask_key not in cache:
                continue

            trace_mask = cache[mask_key]
            axon_path = cache[path_key]

            # Compute bounding box with padding
            ys, xs = np.where(trace_mask > 0)
            if len(ys) == 0:
                continue

            y_min = max(0, ys.min() - crop_padding)
            y_max = min(h, ys.max() + crop_padding + 1)
            x_min = max(0, xs.min() - crop_padding)
            x_max = min(w, xs.max() + crop_padding + 1)

            # Crop image and neuron mask
            crop_img = img[y_min:y_max, x_min:x_max]
            crop_neuron_mask = trace_mask[y_min:y_max, x_min:x_max]

            # Load and crop MDI mask
            mdi_mask_path = state.mdi_mask_path(panel_name, neuron_idx)
            if mdi_mask_path.exists():
                mdi_mask_full = skio.imread(str(mdi_mask_path))
                crop_mdi_mask = mdi_mask_full[y_min:y_max, x_min:x_max]
            else:
                crop_mdi_mask = np.zeros(
                    (y_max - y_min, x_max - x_min), dtype=np.uint8
                )

            # Offset axon path to crop coordinates
            if len(axon_path) > 0:
                offset_path = axon_path.copy().astype(float)
                offset_path[:, 0] -= y_min
                offset_path[:, 1] -= x_min
                path_list = offset_path.tolist()
            else:
                path_list = []

            # Save files
            skio.imsave(
                str(ml_dir / "images" / f"{base_name}.tif"),
                crop_img, check_contrast=False
            )
            skio.imsave(
                str(ml_dir / "masks" / f"{base_name}_neuron.png"),
                (crop_neuron_mask * 255).astype(np.uint8),
                check_contrast=False
            )
            skio.imsave(
                str(ml_dir / "masks" / f"{base_name}_mdi.png"),
                (crop_mdi_mask * 255).astype(np.uint8),
                check_contrast=False
            )

            with open(ml_dir / "axon_paths" / f"{base_name}.json", "w") as f:
                json.dump(path_list, f)

            rows.append({
                "filename": f"{base_name}.tif",
                "panel": panel_name,
                "neuron_idx": neuron_idx,
                "qc_flag": qc_flag,
                "has_mdi_mask": neuron_info.get("has_mdi_mask", False),
                "crop_y_min": y_min,
                "crop_x_min": x_min,
                "crop_y_max": y_max,
                "crop_x_max": x_max,
            })
            exported += 1

    # Save manifest
    df = pd.DataFrame(rows)
    manifest_path = ml_dir / "manifest.csv"
    df.to_csv(str(manifest_path), index=False)

    print(f"ML dataset: {ml_dir}  ({exported} neurons)")
    return df


# ─────────────────────────────────────────────────────────────────
# Backward-compatible wrapper (used by CLI)
# ─────────────────────────────────────────────────────────────────

def export_dataset(project_dir, output_dir, include_rejected=False,
                   crop_padding=50):
    """
    Export both analysis CSV and ML dataset.

    This is the main entry point for CLI usage and backward
    compatibility. Calls both export functions.

    Returns the analysis DataFrame.
    """
    project_dir = Path(project_dir)
    output_dir = Path(output_dir)

    state = ProjectState(project_dir)
    state.load()

    analysis_df = export_analysis_csv(state, output_dir)
    ml_df = export_ml_dataset(state, output_dir, crop_padding=crop_padding)

    print(f"\nExport complete: {len(analysis_df)} neurons analysed, "
          f"{len(ml_df)} crops saved")

    return analysis_df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m review_app.exporter <project_dir> <output_dir>")
        sys.exit(1)
    export_dataset(sys.argv[1], sys.argv[2])
