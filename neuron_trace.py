"""
Neuron tracer - region grow from seed points, skeletonize, filter non-neurons.

Functions:
    trace_neuron(img, seed, channel=0, threshold_pct=0.10, min_absolute=10)
    trace_all_neurons(img, centroids, nuclei_labeled, ...)
    show_traces(img, traces, nuclei_labeled=None)
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure
from collections import deque


def region_grow(img_gray, seed, threshold_pct=0.10, min_absolute=10):
    """
    Grow mask from seed point based on intensity threshold.
    
    Uses the HIGHER of:
    - threshold_pct × seed intensity (relative)
    - min_absolute (absolute floor)
    """
    h, w = img_gray.shape
    img = img_gray.astype(np.float64)
    
    relative_thresh = img[seed[0], seed[1]] * threshold_pct
    min_intensity = max(relative_thresh, float(min_absolute))

    mask = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    queue = deque([seed])
    visited[seed[0], seed[1]] = True

    while queue:
        y, x = queue.popleft()
        if img[y, x] >= min_intensity:
            mask[y, x] = True
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
    return mask


def find_bright_seed(img_gray, centroid, search_radius=50):
    """
    Find the brightest pixel near a centroid.
    If the centroid itself is dim, search nearby for a better seed.
    
    Returns (y, x) of brightest pixel within search_radius, or
    the original centroid if nothing brighter is found.
    """
    h, w = img_gray.shape
    cy, cx = centroid
    
    y0 = max(0, cy - search_radius)
    y1 = min(h, cy + search_radius)
    x0 = max(0, cx - search_radius)
    x1 = min(w, cx + search_radius)
    
    patch = img_gray[y0:y1, x0:x1]
    if patch.size == 0:
        return centroid
    
    local_idx = np.unravel_index(patch.argmax(), patch.shape)
    return (y0 + local_idx[0], x0 + local_idx[1])


def geodesic_max_distance(mask, seed):
    """Quick geodesic BFS to find max distance from seed through mask."""
    h, w = mask.shape

    if not mask[seed[0], seed[1]]:
        mask_ys, mask_xs = np.where(mask)
        if len(mask_ys) == 0:
            return 0.0
        dists = (mask_ys - seed[0])**2 + (mask_xs - seed[1])**2
        nearest = np.argmin(dists)
        seed = (mask_ys[nearest], mask_xs[nearest])

    dist = np.full((h, w), np.inf)
    dist[seed[0], seed[1]] = 0
    queue = deque([seed])
    max_dist = 0.0

    neighbours = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
    ]

    while queue:
        y, x = queue.popleft()
        current_dist = dist[y, x]
        for dy, dx, cost in neighbours:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                new_dist = current_dist + cost
                if new_dist < dist[ny, nx]:
                    dist[ny, nx] = new_dist
                    if new_dist > max_dist:
                        max_dist = new_dist
                    queue.append((ny, nx))

    return max_dist


def nucleus_radius(nucleus_mask):
    """Estimate nucleus radius from mask area (assuming circular)."""
    area = nucleus_mask.sum()
    if area == 0:
        return 15.0
    return np.sqrt(area / np.pi)


def trace_neuron(img, seed, channel=0, threshold_pct=0.10, min_absolute=10,
                 min_seed_intensity=15, seed_search_radius=50):
    """
    Trace a single neuron from a seed point.

    Parameters
    ----------
    img : array
        RGB image.
    seed : tuple (y, x)
        Seed point (e.g. nucleus centroid).
    channel : int
        Channel to trace on (0=red, 1=green, 2=blue).
    threshold_pct : float
        Intensity threshold as fraction of seed value.
    min_absolute : float
        Absolute minimum intensity for region growing.
        Region grow uses max(threshold_pct * seed_intensity, min_absolute).
    min_seed_intensity : float
        If the seed pixel is dimmer than this, search nearby for a
        brighter pixel. If nothing above this value is found, return
        empty mask (no neuron here).
    seed_search_radius : int
        How far to search for a brighter seed (pixels).

    Returns
    -------
    dict with: mask, skeleton, seed, actual_seed
    """
    gray = img[:, :, channel].astype(np.float64)
    
    original_seed = seed
    
    # Check seed intensity — find brighter pixel if needed
    if gray[seed[0], seed[1]] < min_seed_intensity:
        seed = find_bright_seed(gray, seed, seed_search_radius)
        
        # If the best nearby pixel is still too dim, no neuron here
        if gray[seed[0], seed[1]] < min_seed_intensity:
            empty = np.zeros(gray.shape, dtype=bool)
            return {"mask": empty, "skeleton": empty, 
                    "seed": original_seed, "actual_seed": seed}

    mask = region_grow(gray, seed, threshold_pct, min_absolute)
    mask = morphology.remove_small_objects(mask, min_size=50)
    mask = morphology.remove_small_holes(mask, area_threshold=100)

    labeled = measure.label(mask)
    seed_label = labeled[seed[0], seed[1]]
    if seed_label > 0:
        mask = labeled == seed_label
    else:
        empty = np.zeros(gray.shape, dtype=bool)
        return {"mask": empty, "skeleton": empty, 
                "seed": original_seed, "actual_seed": seed}

    skeleton = morphology.skeletonize(mask)

    return {"mask": mask, "skeleton": skeleton, 
            "seed": original_seed, "actual_seed": seed}


def trace_all_neurons(img, centroids, nuclei_labeled, channel=0,
                      threshold_pct=0.10, min_absolute=10,
                      min_seed_intensity=15, seed_search_radius=50,
                      min_reach_ratio=2.0, min_mask_size=100000,
                      max_mask_fraction=0.15, border_margin=50,
                      verbose=True):
    """
    Trace all neurons, filtering out non-neurons.

    Filtering (in order):
    1. Border exclusion: ignore centroids near image edges
    2. Seed intensity: skip if no bright tubulin near the nucleus
    3. Mask size floor: skip if mask < min_mask_size pixels
    4. Mask size ceiling: skip if mask > max_mask_fraction of image
    5. Geodesic reach: skip if tubulin doesn't extend far from nucleus

    Parameters
    ----------
    img : array
        RGB image.
    centroids : list of (y, x)
        Seed points from nucleus_detector.
    nuclei_labeled : 2D array
        Labelled nuclei mask from nucleus_detector.
    channel : int
        Channel to trace on.
    threshold_pct : float
        Relative intensity threshold for region growing.
    min_absolute : float
        Absolute intensity floor for region growing.
    min_seed_intensity : float
        Minimum tubulin intensity at seed. If dimmer, searches nearby.
        If nothing bright found, skips this neuron entirely.
    seed_search_radius : int
        How far from centroid to search for bright tubulin (px).
    min_reach_ratio : float
        Minimum geodesic reach as multiple of nucleus radius.
    min_mask_size : int
        Minimum neuron mask area in pixels.
    max_mask_fraction : float
        Maximum mask area as fraction of total image area.
    border_margin : int
        Pixels to exclude around image edges. Centroids within this
        margin are skipped. Region grow is also blocked from entering
        this zone. Set to 0 to disable.

    Returns
    -------
    traces : list of trace dicts (only neurons that passed all filters)
    kept_centroids : list of (y, x) centroids that passed
    """
    traces = []
    kept_centroids = []
    h, w = img.shape[0], img.shape[1]
    total_pixels = h * w
    max_mask_size = int(total_pixels * max_mask_fraction)

    # Create a working copy with borders zeroed out
    if border_margin > 0:
        img_work = img.copy()
        img_work[:border_margin, :] = 0
        img_work[-border_margin:, :] = 0
        img_work[:, :border_margin] = 0
        img_work[:, -border_margin:] = 0
    else:
        img_work = img

    for i, seed in enumerate(centroids):
        # Filter 0: skip centroids near border
        if border_margin > 0:
            sy, sx = seed
            if sy < border_margin or sy >= h - border_margin or \
               sx < border_margin or sx >= w - border_margin:
                if verbose: print(f"  Neuron {i+1}: FILTERED (near border)")
                continue

        trace = trace_neuron(img_work, seed, channel, threshold_pct,
                             min_absolute, min_seed_intensity,
                             seed_search_radius)
        mask_size = trace["mask"].sum()

        # Filter 1: empty mask (seed too dim)
        if mask_size == 0:
            if verbose: print(f"  Neuron {i+1}: FILTERED (no tubulin signal at seed)")
            continue

        # Filter 2: too small
        if mask_size < min_mask_size:
            if verbose: print(f"  Neuron {i+1}: FILTERED (too small) — "
                  f"mask={mask_size} px, min={min_mask_size}")
            continue

        # Filter 3: too large (probably noise)
        if mask_size > max_mask_size:
            if verbose: print(f"  Neuron {i+1}: FILTERED (too large, likely noise) — "
                  f"mask={mask_size} px, max={max_mask_size} "
                  f"({max_mask_fraction*100:.0f}% of image)")
            continue

        # Get nucleus mask and radius
        label_at_seed = nuclei_labeled[seed[0], seed[1]]
        if label_at_seed > 0:
            nuc_mask = nuclei_labeled == label_at_seed
        else:
            h, w = nuclei_labeled.shape
            yy, xx = np.ogrid[:h, :w]
            nuc_mask = ((yy - seed[0])**2 + (xx - seed[1])**2) < 15**2

        nuc_r = nucleus_radius(nuc_mask)
        min_reach = nuc_r * min_reach_ratio

        # Filter 4: geodesic reach
        actual_seed = trace.get("actual_seed", seed)
        reach = geodesic_max_distance(trace["mask"], actual_seed)

        if reach >= min_reach:
            traces.append(trace)
            kept_centroids.append(seed)
            if verbose: print(f"  Neuron {i+1}: KEPT — mask={mask_size} px, "
                  f"reach={reach:.0f} px, threshold={min_reach:.0f} px")
        else:
            if verbose: print(f"  Neuron {i+1}: FILTERED (low reach) — "
                  f"mask={mask_size} px, reach={reach:.0f} px, "
                  f"threshold={min_reach:.0f} px")

    if verbose: print(f"\n  Kept {len(traces)}/{len(centroids)} neurons")
    return traces, kept_centroids


def show_traces(img, traces, nuclei_labeled=None, save_path=None):
    """Visualise all traced neurons, optionally with nuclei overlay."""
    gray = img[:, :, 0].astype(np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original with seeds and nuclei
    axes[0].imshow(img)
    if nuclei_labeled is not None:
        axes[0].contour(nuclei_labeled > 0, colors='cyan', linewidths=0.8)
    for i, t in enumerate(traces):
        y, x = t["seed"]
        axes[0].plot(x, y, 'c+', markersize=10, markeredgewidth=1.5)
        axes[0].text(x + 5, y - 5, str(i + 1), color='cyan', fontsize=7)
    axes[0].set_title(f"Seeds + nuclei ({len(traces)} neurons)")
    axes[0].axis("off")

    # All masks with nuclei
    axes[1].imshow(gray, cmap="gray")
    if nuclei_labeled is not None:
        axes[1].contour(nuclei_labeled > 0, colors='cyan', linewidths=0.8)
    colours = plt.cm.Set1(np.linspace(0, 1, max(len(traces), 1)))
    for i, t in enumerate(traces):
        if t["mask"].any():
            axes[1].contour(t["mask"], colors=[colours[i % len(colours)]], linewidths=0.5)
    axes[1].set_title("Masks + nuclei")
    axes[1].axis("off")

    # Skeletons with nuclei
    skel_overlay = np.stack([gray / max(gray.max(), 1)] * 3, axis=-1)
    if nuclei_labeled is not None:
        from skimage.segmentation import find_boundaries
        nuclei_border = find_boundaries(nuclei_labeled, mode='outer')
        skel_overlay[nuclei_border, 0] = 0
        skel_overlay[nuclei_border, 1] = 1
        skel_overlay[nuclei_border, 2] = 1
    for i, t in enumerate(traces):
        c = colours[i % len(colours)]
        skel_overlay[t["skeleton"], 0] = c[0]
        skel_overlay[t["skeleton"], 1] = c[1]
        skel_overlay[t["skeleton"], 2] = c[2]
    axes[2].imshow(skel_overlay)
    axes[2].set_title("Skeletons + nuclei")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()