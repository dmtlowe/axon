"""
Furthest tubulin finder + axon path tracer.

Functions:
    find_furthest_point(neuron_mask, nucleus_centroid)
    trace_axon_path(distance_map, furthest_point, nucleus_mask)
    measure_axon(neuron_mask, nucleus_centroid, nucleus_mask)
    show_geodesic(img, neuron_mask, nucleus_centroid, furthest_point, distance_map)
    show_axon_path(img, axon_result)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def geodesic_distance(mask, seed):
    """
    Compute geodesic distance from seed through mask.
    BFS with diagonal cost sqrt(2), cardinal cost 1.
    """
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


def find_furthest_point(neuron_mask, nucleus_centroid):
    """
    Find the furthest point in neuron mask from nucleus centroid,
    measured by geodesic distance.
    """
    dist = geodesic_distance(neuron_mask, nucleus_centroid)

    reachable = dist.copy()
    reachable[~np.isfinite(reachable)] = -1

    furthest_idx = np.argmax(reachable)
    fy, fx = np.unravel_index(furthest_idx, dist.shape)

    return {
        "furthest_point": (fy, fx),
        "geodesic_dist": dist[fy, fx],
        "distance_map": dist,
    }


def trace_axon_path(distance_map, furthest_point, nucleus_mask):
    """
    Trace the axon path from the furthest point back to the nucleus
    by gradient descent through the geodesic distance map.
    
    Stops when the path hits the nucleus mask.
    
    Parameters
    ----------
    distance_map : 2D float array
        Geodesic distance from nucleus centroid.
    furthest_point : tuple (y, x)
    nucleus_mask : 2D bool array
    
    Returns
    -------
    path : list of (y, x) tuples, from axon tip to nucleus boundary
    length_px : float, path length in pixels (with diagonal cost)
    """
    h, w = distance_map.shape
    y, x = furthest_point
    path = [(y, x)]
    
    neighbours = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
    ]
    
    length_px = 0.0
    visited = set()
    visited.add((y, x))
    
    while True:
        # Stop if we hit the nucleus
        if nucleus_mask[y, x]:
            break
        
        # Find neighbour with lowest geodesic distance
        best_dist = distance_map[y, x]
        best_pos = None
        best_cost = 0
        
        for dy, dx, cost in neighbours:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                if np.isfinite(distance_map[ny, nx]) and distance_map[ny, nx] < best_dist:
                    best_dist = distance_map[ny, nx]
                    best_pos = (ny, nx)
                    best_cost = cost
        
        if best_pos is None:
            # Stuck — no lower neighbour found
            break
        
        y, x = best_pos
        visited.add((y, x))
        path.append((y, x))
        length_px += best_cost
    
    return path, length_px


def measure_axon(neuron_mask, nucleus_centroid, nucleus_mask):
    """
    Full measurement: find furthest point, trace path, measure length.
    
    Parameters
    ----------
    neuron_mask : 2D bool array
    nucleus_centroid : tuple (y, x)
    nucleus_mask : 2D bool array
        Mask for this specific nucleus.
    
    Returns
    -------
    dict with: furthest_point, path, length_px, distance_map
    """
    result = find_furthest_point(neuron_mask, nucleus_centroid)
    path, length_px = trace_axon_path(
        result["distance_map"], 
        result["furthest_point"], 
        nucleus_mask
    )
    
    return {
        "furthest_point": result["furthest_point"],
        "geodesic_dist": result["geodesic_dist"],
        "distance_map": result["distance_map"],
        "path": path,
        "length_px": length_px,
    }


def measure_all_axons(traces, centroids, nuclei_labeled):
    """
    Measure axon length for all traced neurons.
    
    Parameters
    ----------
    traces : list of trace dicts from neuron_trace
    centroids : list of (y, x) from nucleus_detector
    nuclei_labeled : 2D array, labelled nuclei mask
    
    Returns
    -------
    list of result dicts from measure_axon
    """
    results = []
    for i, (trace, centroid) in enumerate(zip(traces, centroids)):
        # Get the nucleus mask for this specific neuron
        label_at_seed = nuclei_labeled[centroid[0], centroid[1]]
        if label_at_seed > 0:
            nuc_mask = nuclei_labeled == label_at_seed
        else:
            # Fallback: small circle around centroid
            h, w = nuclei_labeled.shape
            yy, xx = np.ogrid[:h, :w]
            nuc_mask = ((yy - centroid[0])**2 + (xx - centroid[1])**2) < 15**2
        
        result = measure_axon(trace["mask"], centroid, nuc_mask)
        results.append(result)
        fy, fx = result["furthest_point"]
        print(f"  Neuron {i+1}: length = {result['length_px']:.1f} px, "
              f"tip = ({fy}, {fx}), path points = {len(result['path'])}")
    
    return results


def show_geodesic(img, neuron_mask, nucleus_centroid, furthest_point,
                  distance_map=None, save_path=None):
    """Visualise geodesic distance heatmap."""
    if distance_map is None:
        distance_map = geodesic_distance(neuron_mask, nucleus_centroid)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    cy, cx = nucleus_centroid
    fy, fx = furthest_point
    axes[0].plot(cx, cy, 'c+', markersize=15, markeredgewidth=2, label='Nucleus')
    axes[0].plot(fx, fy, 'r+', markersize=15, markeredgewidth=2, label='Furthest')
    axes[0].legend(fontsize=9)
    axes[0].set_title("Nucleus → Furthest point")
    axes[0].axis("off")

    display_dist = distance_map.copy()
    display_dist[~np.isfinite(display_dist)] = np.nan
    display_dist[~neuron_mask] = np.nan
    axes[1].imshow(img[:, :, 0], cmap="gray", alpha=0.3)
    heatmap = axes[1].imshow(display_dist, cmap="hot", alpha=0.8)
    axes[1].plot(cx, cy, 'c+', markersize=15, markeredgewidth=2)
    axes[1].plot(fx, fy, 'w+', markersize=15, markeredgewidth=2)
    plt.colorbar(heatmap, ax=axes[1], label="Geodesic distance (px)", shrink=0.8)
    axes[1].set_title("Geodesic distance from nucleus")
    axes[1].axis("off")

    axes[2].imshow(img)
    axes[2].contour(neuron_mask, colors='lime', linewidths=0.5)
    axes[2].plot(cx, cy, 'c+', markersize=15, markeredgewidth=2)
    axes[2].plot(fx, fy, 'r+', markersize=15, markeredgewidth=2)
    euclidean = np.sqrt((fy - cy)**2 + (fx - cx)**2)
    axes[2].set_title(f"Geodesic: {distance_map[fy, fx]:.0f} px | Euclidean: {euclidean:.0f} px")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def show_axon_path(img, axon_result, nucleus_mask=None, save_path=None):
    """
    Visualise the traced axon path.
    
    Parameters
    ----------
    img : array, RGB image
    axon_result : dict from measure_axon
    nucleus_mask : 2D bool array, optional
    save_path : str, optional
    """
    path = axon_result["path"]
    distance_map = axon_result["distance_map"]
    furthest = axon_result["furthest_point"]
    
    path_arr = np.array(path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Path on original image
    axes[0].imshow(img)
    axes[0].plot(path_arr[:, 1], path_arr[:, 0], 'lime', linewidth=1.5, label='Axon path')
    axes[0].plot(path_arr[0, 1], path_arr[0, 0], 'r+', markersize=15, 
                 markeredgewidth=2, label='Tip')
    axes[0].plot(path_arr[-1, 1], path_arr[-1, 0], 'c+', markersize=15, 
                 markeredgewidth=2, label='Nucleus boundary')
    if nucleus_mask is not None:
        axes[0].contour(nucleus_mask, colors='cyan', linewidths=0.8)
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"Axon path — {axon_result['length_px']:.1f} px")
    axes[0].axis("off")
    
    # Path on geodesic heatmap
    display_dist = distance_map.copy()
    display_dist[~np.isfinite(display_dist)] = np.nan
    axes[1].imshow(img[:, :, 0], cmap="gray", alpha=0.3)
    heatmap = axes[1].imshow(display_dist, cmap="hot", alpha=0.7)
    axes[1].plot(path_arr[:, 1], path_arr[:, 0], 'lime', linewidth=1.5)
    axes[1].plot(path_arr[0, 1], path_arr[0, 0], 'w+', markersize=15, markeredgewidth=2)
    axes[1].plot(path_arr[-1, 1], path_arr[-1, 0], 'c+', markersize=15, markeredgewidth=2)
    if nucleus_mask is not None:
        axes[1].contour(nucleus_mask, colors='cyan', linewidths=0.8)
    plt.colorbar(heatmap, ax=axes[1], label="Geodesic distance (px)", shrink=0.8)
    axes[1].set_title("Path on geodesic map")
    axes[1].axis("off")
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()