# Axon Analysis — Documentation

## Overview

A Python application for measuring axon length and microtubule disorganisation index (MDI) from fluorescence microscopy images. Combines automated image processing with a napari-based GUI for human review, manual MDI mask annotation, and dataset export.

**Input:** Multi-panel TIF images with 3 channels — Red (Tubulin, ch 0), Green (HRP, ch 1), Blue (DAPI, ch 2).

**Example Data:** https://drive.google.com/drive/folders/10KaK9dqEUR0_Bdvv4G4Z_btKqIFCM7Ao?usp=drive_link

**Two primary goals:**

1. **Lab quantification** — measure axon length and MDI mask area per neuron, export as a single analysis CSV
2. **ML training data** — build paired image/mask crops for training a segmentation model to predict MDI automatically

---

## System Architecture

```
review_app/              ← GUI application (napari + Qt)
    __init__.py
    __main__.py          ← python -m review_app
    app.py               ← Entry point, startup dialog, signal wiring
    data_model.py        ← ProjectState: JSON persistence, navigation
    batch_processor.py   ← Runs pipeline over all panels, caches results
    viewer_controller.py ← Napari layer management, channel split, palettes
    widgets.py           ← Qt dock widgets (navigation, params, MDI paint)
    exporter.py          ← Analysis CSV + ML dataset export

*.py (root level)        ← Pipeline algorithm modules (used by batch_processor)
    panel_splitter.py    ← Splits multi-panel TIF into individual panels
    nucleus_detector.py  ← Detects neuronal nuclei from DAPI + HRP filter
    neuron_trace.py      ← Region-grows tubulin masks, filters non-neurons
    measure_axon.py      ← Dijkstra intensity-guided axon path + length
    mdi_analysis.py      ← Computational MDI (experimental, not in active use)
    furthest_tubulin_finder.py  ← Earlier iteration, kept for reference
    border_test.py       ← Utility
    rgb.py               ← Utility
    roi_inspect.py       ← Utility
```

The root-level `.py` files contain the image analysis algorithms. The `review_app/` folder contains the GUI that orchestrates them. `batch_processor.py` imports from the root-level modules.

---

## Quick Start

### Launch the GUI

```bash
python -m review_app
```

### Create a new project

1. Select your source TIF file
2. Set grid dimensions (rows × cols)
3. Choose a project folder
4. Click "Create Project & Process" — batch processing runs automatically with a progress dialog

### Open an existing project

1. Click "Select project folder"
2. Choose a folder containing `state.json`

---

## GUI Features

### Navigation & QC

- **Prev / Next** buttons (or keyboard: `P` / `N`) to step through neurons
- **Accept / Reject / Modified** QC flags (keyboard: `A` / `R`)
- **Jump to Panel** dropdown for direct navigation
- **Progress bar** showing reviewed vs total neurons
- **Notes** text field per neuron

### Channel View

The viewer provides multiple image layers that can be toggled independently:

| Layer | Default | Description |
|-------|---------|-------------|
| Red (Tubulin) | Visible | Channel 0, red colourmap, additive blending |
| Green (HRP) | Visible | Channel 1, green colourmap, additive blending |
| Blue (DAPI) | Visible | Channel 2, blue colourmap, additive blending |
| Greyscale | Hidden | Mean of all 3 channels, grey colourmap |
| Composite (RGB) | Hidden | Original RGB image |

Channels use additive blending so you can mix and match — e.g. view just Red + Green to see tubulin/HRP without DAPI.

### MDI Mask Painting

- Select the "MDI Mask" layer in the napari layer list
- Press `2` for paint brush, `1` for eraser
- Use `[` `]` to resize brush
- Brush size is also adjustable via the MDI Painting widget
- Masks are displayed in **turquoise** at 35% opacity
- Masks auto-save when navigating between neurons
- "Clear MDI Mask" button to start fresh

### Axon Start Point Adjustment

Sometimes the automated trace extends into the cell body. The start point adjustment allows trimming:

1. Click "Adjust Start Point"
2. Click on the axon where the true start should be (where the axon leaves the soma)
3. The path is trimmed to the nearest traced point — no Dijkstra re-run needed
4. The updated axon length is shown immediately
5. "Reset to Original" restores the full traced path

### Parameter Tuning

Per-panel pipeline parameter adjustment with live sliders:

**Nucleus Detection:**
- Min size (50–2000 px)
- Blur sigma (1.0–30.0)
- HRP threshold factor (1.0–5.0)

**Neuron Tracing:**
- Threshold % (0.02–0.50)
- Min reach ratio (1.0–15.0)
- Border margin (0–800 px)
- Min mask size (5,000–500,000 px)
- Min seed intensity (5–100)

**Axon Measurement:**
- Intensity weight (0.5–5.0)
- Nucleus snap distance (5–150 px)

Click "Re-run Panel" to reprocess with updated parameters. "Reset to Defaults" restores default values.

### Accessibility

- **Colourblind-friendly mode** checkbox toggles all overlay colours to a palette based on Wong (2011), safe for protanopia, deuteranopia, and tritanopia
- Standard palette: cyan centroid, red tip, blue start, bright yellow axon path
- Colourblind palette: orange centroid, purple tip, blue start, yellow axon path

---

## Export

Click **Export Dataset** in the GUI (or run via CLI). Two outputs are produced:

### Goal 1: Lab Analysis CSV

**Filename:** `{TIF_name}_MDI_analysis.csv`

One row per accepted/modified neuron with columns:

| Column | Description |
|--------|-------------|
| `panel` | Panel name |
| `neuron_idx` | Neuron index within the panel |
| `qc_flag` | accepted / modified |
| `axon_length_px` | Axon path length in pixels |
| `axon_length_um` | Axon length in µm (÷ 29.21) |
| `mdi_area_px` | MDI mask area in pixels |
| `mdi_area_um2` | MDI mask area in µm² |
| `mdi_ratio` | MDI area (µm²) / axon length (µm) |
| `has_mdi_mask` | Whether an MDI mask was drawn |
| `notes` | Reviewer notes |
| `centroid_y/x` | Nucleus centroid coordinates |
| `tip_y/x` | Axon tip coordinates |
| `start_y/x` | Axon start coordinates |

### Goal 2: ML Training Dataset

**Folder:** `ml_dataset/` inside the export directory

```
ml_dataset/
    images/          ← Cropped neuron TIF images (RGB, padded bounding box)
    masks/
        *_neuron.png ← Binary neuron mask (trace region)
        *_mdi.png    ← Binary MDI mask (hand-drawn)
    axon_paths/
        *.json       ← Axon path coordinates (offset to crop)
    manifest.csv     ← Index mapping filenames → metadata
```

**manifest.csv** is the table of contents for ML data loaders — maps each crop to its panel, neuron index, QC flag, MDI mask presence, and crop coordinates. Use it to build a PyTorch/TensorFlow dataset without parsing filenames.

### CLI Export

```bash
python -m review_app.exporter <project_dir> <output_dir>
```

---

## Project File Structure

When you create a project, the following structure is created:

```
project_folder/
    state.json           ← Single source of truth: QC flags, params, cursor
    panels/
        panel_R1_C1.tif  ← Individual panel images
        panel_R1_C1/
            pipeline.npz ← Cached pipeline results (masks, paths, lengths)
            neuron_000/
                mdi_mask.png  ← Hand-drawn MDI mask (if any)
            neuron_001/
                mdi_mask.png
        panel_R1_C2.tif
        ...
```

**state.json** persists everything across sessions: which neurons have been reviewed, QC flags, parameter overrides, start-point trims, notes, and the reviewer's current position (panel + neuron index).

**pipeline.npz** is a numpy compressed archive caching all intermediate results per panel: trace masks, axon paths, tip/start points, centroid coordinates, axon lengths. This avoids re-running the pipeline every time the GUI loads.

---

## Pipeline Algorithm Modules

These root-level Python files contain the image analysis algorithms. They can be used independently or are orchestrated by the GUI's batch processor.

### panel_splitter.py

Splits a multi-panel TIF into individual images.

```python
from panel_splitter import split_panels
panels = split_panels("experiment.tif", rows=6, cols=6)
```

### nucleus_detector.py

Detects neuronal nuclei from DAPI staining, filtered by HRP colocalisation.

**Algorithm:** Blue channel → Gaussian blur → Otsu threshold → watershed segmentation → HRP ring filter.

```python
from nucleus_detector import detect_nuclei
centroids, labeled = detect_nuclei(img)
```

### neuron_trace.py

Region-grows tubulin masks from nucleus seeds. Filters non-neurons by geodesic reach ratio and mask size.

```python
from neuron_trace import trace_all_neurons
traces, kept_centroids = trace_all_neurons(img, centroids, labeled)
```

### measure_axon.py

Traces an intensity-guided axon path (Dijkstra) from nucleus boundary to the geodesic furthest point (axon tip). Path follows bright tubulin signal.

**Cost function:** `spatial_cost / (intensity ^ weight)` — brighter pixels are cheaper to traverse.

```python
from measure_axon import measure_axon
result = measure_axon(img, neuron_mask, centroid, nucleus_mask)
# result["length_px"], result["path"], result["tip_point"], result["start_point"]
```

### Scaling Factor

The TIF metadata gives X Resolution = 29.21 px/µm. To convert:

```python
SCALE_PX_PER_UM = 29.21
length_um = length_px / SCALE_PX_PER_UM
area_um2 = area_px / (SCALE_PX_PER_UM ** 2)
```

---

## Default Pipeline Parameters

```python
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
```

---

## Dependencies

```
napari >= 0.5
scikit-image
scipy
numpy
pandas
qtpy
magicgui
tifffile
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `A` | Accept current neuron |
| `R` | Reject current neuron |
| `N` | Next neuron |
| `P` | Previous neuron |
| `2` | Paint brush (napari labels) |
| `1` | Eraser (napari labels) |
| `[` `]` | Decrease / increase brush size |
