"""
Nucleus detector - find nuclei from DAPI (blue channel),
filtered by HRP colocalisation (green channel).

Usage:
    python nucleus_detector.py image.tif

Returns list of (y, x) centroids and labelled mask.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io, morphology, measure, filters
from pathlib import Path


def detect_nuclei(img, min_size=100, blur_sigma=10,
                  require_hrp=True, hrp_channel=1, hrp_ring_width=30,
                  hrp_threshold_factor=2.0):
    """
    Detect nuclei from the blue channel of an RGB image.
    
    Parameters
    ----------
    img : array
        RGB image.
    min_size : int
        Minimum nucleus area in pixels.
    blur_sigma : float
        Gaussian blur sigma for DAPI channel.
    close_radius : int
        Morphological closing disk radius. Merges nearby DAPI
        fragments into single nuclei. Increase if nuclei are
        still being split.
    require_hrp : bool
        If True, only keep nuclei with green signal nearby.
    hrp_channel : int
        Which channel has HRP (default 1 = green).
    hrp_ring_width : int
        Half-width of patch around centroid to check for HRP.
    hrp_threshold_factor : float
        Keep nucleus if mean green in patch > factor × median green.
    
    Returns
    -------
    centroids : list of (y, x) tuples
    labeled : 2D array, each nucleus has a unique integer label
    """
    blue = img[:, :, 2].astype(np.float64)
    
    # Blur to merge internal structure
    blurred = filters.gaussian(blue, sigma=blur_sigma)
    
    # Otsu threshold
    thresh = filters.threshold_otsu(blurred)
    mask = blurred > thresh
    
    # Clean up
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.remove_small_holes(mask, area_threshold=500)
    
    # Simple connected components
    labeled = measure.label(mask)
    
    # Green channel for HRP check
    if require_hrp:
        green = img[:, :, hrp_channel].astype(np.float64)
        bg_green = np.median(green)
        hrp_min = bg_green * hrp_threshold_factor
    
    # Filter nuclei
    centroids = []
    filtered_label = np.zeros_like(labeled)
    new_id = 1
    
    for region in measure.regionprops(labeled):
        if region.area < min_size:
            continue
        
        cy, cx = int(region.centroid[0]), int(region.centroid[1])
        
        if require_hrp:
            h, w = green.shape
            y0 = max(0, cy - hrp_ring_width)
            y1 = min(h, cy + hrp_ring_width)
            x0 = max(0, cx - hrp_ring_width)
            x1 = min(w, cx + hrp_ring_width)
            mean_green = green[y0:y1, x0:x1].mean()
            
            if mean_green < hrp_min:
                continue
        
        filtered_label[labeled == region.label] = new_id
        centroids.append((cy, cx))
        new_id += 1
    
    return centroids, filtered_label


def show_nuclei(img_path):
    """Detect and display nuclei."""
    img = io.imread(str(img_path))
    centroids, labeled = detect_nuclei(img)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img[:, :, 2], cmap="gray")
    axes[0].set_title("Blue channel (DAPI)")
    axes[0].axis("off")
    
    axes[1].imshow(labeled, cmap="nipy_spectral")
    axes[1].set_title(f"Labelled nuclei ({labeled.max()} found)")
    axes[1].axis("off")
    
    axes[2].imshow(img)
    for i, (cy, cx) in enumerate(centroids):
        axes[2].plot(cx, cy, 'c+', markersize=10, markeredgewidth=1.5)
        axes[2].text(cx + 5, cy - 5, str(i+1), color='cyan', fontsize=7)
    axes[2].set_title(f"Centroids ({len(centroids)} nuclei)")
    axes[2].axis("off")
    
    plt.tight_layout()
    out = Path(img_path).with_suffix(".nuclei.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Found {len(centroids)} nuclei")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python nucleus_detector.py image.tif")
        sys.exit(1)
    show_nuclei(sys.argv[1])