"""
Measure axon length by tracing the tubulin intensity ridge
from nucleus boundary to the axon tip.

Uses geodesic furthest point to define the destination,
then traces an intensity-weighted path to get there.

Functions:
    measure_axon(img, neuron_mask, nucleus_centroid, nucleus_mask, channel=0)
    measure_all_axons(img, traces, centroids, nuclei_labeled, channel=0)
    show_axon(img, axon_result, nucleus_mask=None)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq


def geodesic_distance(mask, seed):
    """Geodesic distance from seed through mask via BFS."""
    h, w = mask.shape
    dist = np.full((h, w), np.inf)

    if not mask[seed[0], seed[1]]:
        mask_ys, mask_xs = np.where(mask)
        if len(mask_ys) == 0:
            return dist
        dists = (mask_ys - seed[0])**2 + (mask_xs - seed[1])**2
        nearest = np.argmin(dists)
        seed = (mask_ys[nearest], mask_xs[nearest])

    dist[seed[0], seed[1]] = 0
    queue = deque([seed])

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
                    queue.append((ny, nx))

    return dist


def find_start_point(nucleus_mask, neuron_mask, tip, distance_map):
    """
    Find where the axon meets the nucleus boundary.
    
    Uses the geodesic distance map (from the centroid) to find the
    nucleus boundary pixel with the highest geodesic distance that
    is still connected to the neuron mask. This is where the axon
    actually leaves the soma, not just the nearest point to the tip.
    
    Parameters
    ----------
    nucleus_mask : 2D bool array
    neuron_mask : 2D bool array
    tip : tuple (y, x), the axon tip
    distance_map : 2D float array, geodesic distance from centroid
    """
    from skimage.segmentation import find_boundaries
    boundary = find_boundaries(nucleus_mask, mode='outer')
    
    # Boundary pixels that are also in the neuron mask
    candidates = boundary & neuron_mask
    
    if not candidates.any():
        candidates = boundary
    
    if not candidates.any():
        ys, xs = np.where(nucleus_mask)
        return (int(ys.mean()), int(xs.mean()))
    
    # Pick the boundary pixel on the path toward the tip:
    # highest geodesic distance from centroid = furthest along the neuron
    # but still on the nucleus edge
    ys, xs = np.where(candidates)
    geo_dists = distance_map[ys, xs]
    
    # Filter out inf values
    valid = np.isfinite(geo_dists)
    if not valid.any():
        # Fallback to Euclidean closest to tip
        ty, tx = tip
        dists = (ys - ty)**2 + (xs - tx)**2
        best = np.argmin(dists)
        return (ys[best], xs[best])
    
    best = np.argmax(geo_dists[valid])
    valid_ys = ys[valid]
    valid_xs = xs[valid]
    return (valid_ys[best], valid_xs[best])


def trace_intensity_path(gray, start, end, neuron_mask, intensity_weight=2.0):
    """
    Trace a path from start to end following high intensity signal,
    constrained to the neuron mask.
    
    Uses Dijkstra's algorithm where the cost of each step is:
        spatial_cost / (intensity ^ weight)
    
    This means brighter pixels are cheaper to traverse,
    so the path follows the intensity ridge.
    
    Parameters
    ----------
    gray : 2D float array
        Tubulin channel intensity.
    start : tuple (y, x)
    end : tuple (y, x)
    neuron_mask : 2D bool array
    intensity_weight : float
        How strongly intensity pulls the path. Higher = sticks
        more tightly to bright signal.
    
    Returns
    -------
    path : list of (y, x) tuples
    length_px : float, path length in pixels
    """
    h, w = gray.shape
    
    # Normalise intensity to 0-1 within the mask
    mask_vals = gray[neuron_mask]
    if mask_vals.max() > mask_vals.min():
        norm = (gray - mask_vals.min()) / (mask_vals.max() - mask_vals.min())
    else:
        norm = np.ones_like(gray)
    norm = np.clip(norm, 0.01, 1.0)  # avoid division by zero
    
    # Cost: spatial distance / intensity^weight
    # Bright pixels = low cost, dark pixels = high cost
    neighbours = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
    ]
    
    # Dijkstra
    cost = np.full((h, w), np.inf)
    cost[start[0], start[1]] = 0
    prev = np.full((h, w, 2), -1, dtype=np.int32)
    visited = np.zeros((h, w), dtype=bool)
    
    # Priority queue: (cost, y, x)
    pq = [(0.0, start[0], start[1])]
    
    while pq:
        c, y, x = heapq.heappop(pq)
        
        if visited[y, x]:
            continue
        visited[y, x] = True
        
        # Reached the end
        if y == end[0] and x == end[1]:
            break
        
        for dy, dx, spatial_cost in neighbours:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and neuron_mask[ny, nx] and not visited[ny, nx]:
                # Average intensity of current and next pixel
                avg_intensity = (norm[y, x] + norm[ny, nx]) / 2.0
                step_cost = spatial_cost / (avg_intensity ** intensity_weight)
                new_cost = c + step_cost
                
                if new_cost < cost[ny, nx]:
                    cost[ny, nx] = new_cost
                    prev[ny, nx] = [y, x]
                    heapq.heappush(pq, (new_cost, ny, nx))
    
    # Reconstruct path
    path = []
    y, x = end
    while y >= 0 and x >= 0:
        path.append((y, x))
        py, px = prev[y, x]
        y, x = int(py), int(px)
    path.reverse()
    
    # Calculate actual path length in pixels
    length_px = 0.0
    for i in range(1, len(path)):
        dy = path[i][0] - path[i-1][0]
        dx = path[i][1] - path[i-1][1]
        length_px += np.sqrt(dy**2 + dx**2)
    
    return path, length_px


def measure_axon(img, neuron_mask, nucleus_centroid, nucleus_mask, 
                 channel=0, intensity_weight=2.0, nucleus_snap_distance=30):
    """
    Full axon measurement pipeline.
    
    1. Find furthest point (geodesic) = axon tip
    2. Trace intensity-guided path from tip toward the nucleus
    3. Path stops when it gets within nucleus_snap_distance (geodesic)
       of the nucleus mask
    
    Parameters
    ----------
    img : array, RGB image
    neuron_mask : 2D bool array
    nucleus_centroid : tuple (y, x)
    nucleus_mask : 2D bool array
    channel : int, which channel has tubulin (default 0 = red)
    intensity_weight : float, how strongly path follows bright signal
    nucleus_snap_distance : int
        When the path gets within this many geodesic pixels of the
        nucleus mask, it stops. Prevents path from looping around soma.
    
    Returns
    -------
    dict with: path, length_px, start_point, tip_point, distance_map
    """
    gray = img[:, :, channel].astype(np.float64)
    
    # Step 1: find axon tip via geodesic
    geo = geodesic_distance(neuron_mask, nucleus_centroid)
    reachable = geo.copy()
    reachable[~np.isfinite(reachable)] = -1
    tip_idx = np.argmax(reachable)
    tip = np.unravel_index(tip_idx, geo.shape)
    
    # Step 2: compute geodesic distance from nucleus mask boundary
    # (so we know how far each pixel is from the nucleus)
    nuc_geo = geodesic_distance(neuron_mask, nucleus_centroid)
    
    # Step 3: trace intensity path from tip toward nucleus centroid
    path, _ = trace_intensity_path(
        gray, tip, nucleus_centroid, neuron_mask, intensity_weight
    )
    
    # Step 4: trim path — stop when within snap distance of nucleus
    trimmed_path = []
    for point in path:
        trimmed_path.append(point)
        if nuc_geo[point[0], point[1]] <= nucleus_snap_distance:
            break
        if nucleus_mask[point[0], point[1]]:
            break
    
    # Calculate length of trimmed path
    length_px = 0.0
    for i in range(1, len(trimmed_path)):
        dy = trimmed_path[i][0] - trimmed_path[i-1][0]
        dx = trimmed_path[i][1] - trimmed_path[i-1][1]
        length_px += np.sqrt(dy**2 + dx**2)
    
    start = trimmed_path[-1] if trimmed_path else nucleus_centroid
    
    return {
        "path": trimmed_path,
        "length_px": length_px,
        "start_point": start,
        "tip_point": tip,
        "distance_map": geo,
    }


def measure_all_axons(img, traces, centroids, nuclei_labeled, 
                      channel=0, intensity_weight=2.0):
    """
    Measure axons for all traced neurons.
    
    Returns list of axon result dicts.
    """
    results = []
    for i, (trace, centroid) in enumerate(zip(traces, centroids)):
        label_at_seed = nuclei_labeled[centroid[0], centroid[1]]
        if label_at_seed > 0:
            nuc_mask = nuclei_labeled == label_at_seed
        else:
            h, w = nuclei_labeled.shape
            yy, xx = np.ogrid[:h, :w]
            nuc_mask = ((yy - centroid[0])**2 + (xx - centroid[1])**2) < 15**2
        
        result = measure_axon(img, trace["mask"], centroid, nuc_mask, 
                              channel, intensity_weight)
        results.append(result)
        print(f"  Neuron {i+1}: length = {result['length_px']:.1f} px, "
              f"path points = {len(result['path'])}")
    
    return results


def show_axon(img, axon_result, nucleus_mask=None, save_path=None, show=True):
    """
    Visualise the measured axon path.
    """
    path = axon_result["path"]
    start = axon_result["start_point"]
    tip = axon_result["tip_point"]
    
    path_arr = np.array(path)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    ax.imshow(img)
    ax.plot(path_arr[:, 1], path_arr[:, 0], 'lime', linewidth=1.5, label='Axon')
    ax.plot(tip[1], tip[0], 'r+', markersize=15, markeredgewidth=2, label='Tip')
    ax.plot(start[1], start[0], 'c+', markersize=15, markeredgewidth=2, label='Start')
    if nucleus_mask is not None:
        ax.contour(nucleus_mask, colors='cyan', linewidths=0.8)
    ax.legend(fontsize=9)
    ax.set_title(f"Axon: {axon_result['length_px']:.1f} px")
    ax.axis("off")
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)