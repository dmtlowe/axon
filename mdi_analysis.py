"""
Microtubule Disorganisation Index (MDI) analysis.

Measures axon width along the traced path using intensity-weighted
second moment (spread = 2*sigma) and detects regions of anomalous
swelling that indicate microtubule disorganisation.

Functions:
    analyse_mdi(img, neuron_mask, axon_result, nucleus_mask, ...)
    analyse_all_mdi(img, traces, axon_results, kept_centroids, nuclei_labeled, ...)
    show_mdi(img, mdi_result, nucleus_mask=None, save_path=None, show=True)
    show_mdi_debug(img, mdi_result, neuron_mask, nucleus_mask=None, ...)
"""

import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────
# 1. Path resampling
# ──────────────────────────────────────────────────────────────────

def resample_path(path, step_size=5.0):
    """
    Resample a dense pixel path at regular arc-length intervals.

    Parameters
    ----------
    path : list of (y, x) tuples
        Dense path from measure_axon. Tip at index 0, soma at index -1.
    step_size : float
        Spacing between sampled points in pixels (arc-length).
        Default 5 px ≈ 0.17 µm at 29.21 px/µm.

    Returns
    -------
    sampled_points : list of (float, float)
        Evenly spaced (y, x) points along the path.
    cumulative_dist : list of float
        Arc-length distance from index 0 for each sampled point.
    """
    if len(path) < 2:
        return list(path), [0.0]

    # Build cumulative arc-length array
    arc = [0.0]
    for i in range(1, len(path)):
        dy = path[i][0] - path[i - 1][0]
        dx = path[i][1] - path[i - 1][1]
        arc.append(arc[-1] + np.sqrt(dy**2 + dx**2))

    total_length = arc[-1]
    if total_length < step_size:
        return list(path), arc

    # Place sample points at regular arc-length intervals
    sampled = []
    cum_dist = []
    target = 0.0
    j = 0  # index into original path

    while target <= total_length:
        # Advance j so arc[j] <= target < arc[j+1]
        while j < len(arc) - 2 and arc[j + 1] < target:
            j += 1

        if j >= len(arc) - 1:
            break

        # Interpolate between path[j] and path[j+1]
        seg_len = arc[j + 1] - arc[j]
        if seg_len < 1e-9:
            alpha = 0.0
        else:
            alpha = (target - arc[j]) / seg_len

        y = path[j][0] + alpha * (path[j + 1][0] - path[j][0])
        x = path[j][1] + alpha * (path[j + 1][1] - path[j][1])
        sampled.append((y, x))
        cum_dist.append(target)

        target += step_size

    return sampled, cum_dist


# ──────────────────────────────────────────────────────────────────
# 2. Tangent and normal computation
# ──────────────────────────────────────────────────────────────────

def compute_tangents(sampled_points, window=5):
    """
    Compute local tangent and normal directions at each sampled point.

    The tangent at point i is the direction from point[i-k] to
    point[i+k], normalised to unit length. The normal is the
    90° rotation: normal = (-tx, ty).

    Parameters
    ----------
    sampled_points : list of (y, x) float tuples
    window : int
        Points on each side for tangent estimation. Default 5.

    Returns
    -------
    tangents : list of (ty, tx) unit vectors
    normals : list of (ny, nx) unit vectors perpendicular to tangent
    """
    n = len(sampled_points)
    tangents = []
    normals = []

    for i in range(n):
        i_back = max(0, i - window)
        i_fwd = min(n - 1, i + window)

        dy = sampled_points[i_fwd][0] - sampled_points[i_back][0]
        dx = sampled_points[i_fwd][1] - sampled_points[i_back][1]
        length = np.sqrt(dy**2 + dx**2)

        if length < 1e-8:
            ty, tx = 0.0, 1.0
        else:
            ty, tx = dy / length, dx / length

        tangents.append((ty, tx))
        # 90° CCW rotation in image coords: (ty, tx) → (-tx, ty)
        normals.append((-tx, ty))

    return tangents, normals


# ──────────────────────────────────────────────────────────────────
# 3. Cross-section measurement (intensity-weighted second moment)
# ──────────────────────────────────────────────────────────────────

def sample_perpendicular(point, normal, tubulin_gray, max_half_width=80):
    """
    Sample tubulin intensity along a perpendicular ray.

    Casts a ray in both ±normal directions from the point, sampling
    the raw tubulin channel intensity at each pixel.

    Parameters
    ----------
    point : (y, x) float tuple
    normal : (ny, nx) float unit vector
    tubulin_gray : 2D float array
    max_half_width : int
        Maximum distance to walk in each direction. Default 80 px.

    Returns
    -------
    positions : 1D array of float
        Signed distance from centre along the normal (-max to +max).
        Negative = -normal direction, positive = +normal direction.
    intensities : 1D array of float
        Tubulin intensity at each sampled position.
    """
    h, w = tubulin_gray.shape
    py, px = point
    ny, nx = normal

    positions = []
    intensities = []

    # Walk from -max_half_width to +max_half_width
    for t in range(-max_half_width, max_half_width + 1):
        ry = int(round(py + t * ny))
        rx = int(round(px + t * nx))
        if 0 <= ry < h and 0 <= rx < w:
            positions.append(float(t))
            intensities.append(tubulin_gray[ry, rx])

    return np.array(positions), np.array(intensities)


def measure_spread(positions, intensities):
    """
    Measure the intensity-weighted spread (second moment) of a
    cross-section profile.

    Uses the intensity-weighted standard deviation of position:
        σ = sqrt( Σ(w_i × (x_i - x_mean)²) / Σ(w_i) )
    where w_i = max(0, intensity_i - background).

    This captures the TOTAL spread of signal including splayed
    multi-peaked profiles (MDI regions where tubulin fans out
    with holes in between). Unlike FWHM, it doesn't stop at
    the first valley — all signal contributes to the spread.

    Reports spread = 2σ as the effective width (analogous to a
    Gaussian where 2σ ≈ FWHM/1.18).

    Parameters
    ----------
    positions : 1D array, signed distances from centre
    intensities : 1D array, tubulin intensities

    Returns
    -------
    dict with:
        spread : float, 2σ effective width in pixels.
                 0.0 if no signal detected.
        sigma : float, intensity-weighted std of position.
        peak_intensity : float, background-subtracted peak.
        background : float, estimated background level.
        weighted_centre : float, intensity-weighted mean position.
    """
    if len(intensities) < 3:
        return {"spread": 0.0, "sigma": 0.0, "peak_intensity": 0.0,
                "background": 0.0, "weighted_centre": 0.0}

    # Background = minimum intensity along the ray
    background = float(np.min(intensities))
    weights = np.maximum(intensities - background, 0.0)

    total_weight = np.sum(weights)
    peak_val = float(np.max(weights))

    if total_weight < 1.0 or peak_val < 1.0:
        return {"spread": 0.0, "sigma": 0.0, "peak_intensity": 0.0,
                "background": background, "weighted_centre": 0.0}

    # Intensity-weighted mean position
    weighted_centre = np.sum(weights * positions) / total_weight

    # Intensity-weighted variance
    variance = np.sum(weights * (positions - weighted_centre)**2) / total_weight
    sigma = np.sqrt(variance)

    # Effective width = 2σ
    spread = 2.0 * sigma

    return {
        "spread": spread,
        "sigma": sigma,
        "peak_intensity": peak_val,
        "background": background,
        "weighted_centre": float(weighted_centre),
    }


def measure_cross_section(point, normal, tubulin_gray, max_half_width=80):
    """
    Measure the axon cross-section using intensity-weighted spread.

    No binary mask involved — works directly on continuous
    tubulin intensity. Captures the total spread of signal
    including splayed multi-peaked profiles in MDI regions.

    Parameters
    ----------
    point : (y, x) float tuple
    normal : (ny, nx) float unit vector
    tubulin_gray : 2D float array
    max_half_width : int
        Maximum distance in each direction. Default 80 px.

    Returns
    -------
    dict with spread, sigma, peak_intensity, background,
         weighted_centre, positions, intensities
    """
    positions, intensities = sample_perpendicular(
        point, normal, tubulin_gray, max_half_width
    )

    spread_result = measure_spread(positions, intensities)

    return {
        **spread_result,
        "positions": positions,
        "intensities": intensities,
    }


# ──────────────────────────────────────────────────────────────────
# 4. Width profile
# ──────────────────────────────────────────────────────────────────

def build_width_profile(sampled_points, normals, cumulative_dist,
                        tubulin_gray, max_half_width=80, soma_skip=0):
    """
    Build 1D spread-width profile along the axon.

    Parameters
    ----------
    sampled_points : list of (y, x) float tuples
    normals : list of (ny, nx) float tuples
    cumulative_dist : list of float
    tubulin_gray : 2D float array
    max_half_width : int
    soma_skip : int
        Number of points to skip at the soma end (end of list).

    Returns
    -------
    dict with arc_lengths, widths (2*sigma spread values), peak_intensities,
         cross_sections (list of per-point dicts)
    """
    end = len(sampled_points) - soma_skip if soma_skip > 0 else len(sampled_points)
    if end < 1:
        end = 1

    arc_lengths = []
    widths = []
    peak_intensities = []
    cross_sections = []

    for i in range(end):
        cs = measure_cross_section(
            sampled_points[i], normals[i],
            tubulin_gray, max_half_width
        )
        arc_lengths.append(cumulative_dist[i])
        widths.append(cs["spread"])
        peak_intensities.append(cs["peak_intensity"])
        cross_sections.append(cs)

    return {
        "arc_lengths": np.array(arc_lengths),
        "widths": np.array(widths, dtype=float),
        "peak_intensities": np.array(peak_intensities),
        "cross_sections": cross_sections,
    }


# ──────────────────────────────────────────────────────────────────
# 5. Anomaly detection
# ──────────────────────────────────────────────────────────────────

def detect_mdi_regions(widths, arc_lengths,
                       rolling_window_frac=0.25, threshold_k=2.5,
                       min_region_len=3, merge_gap=3):
    """
    Detect disorganised regions from the width profile.

    Uses rolling median + MAD baseline to flag anomalous widths.

    Parameters
    ----------
    widths : 1D array
    arc_lengths : 1D array
    rolling_window_frac : float
        Window size as fraction of profile length. Default 0.25.
    threshold_k : float
        MADs above rolling median to flag. Default 2.5.
    min_region_len : int
        Minimum consecutive flagged samples. Default 3.
    merge_gap : int
        Maximum gap between flagged runs to merge. Default 3.

    Returns
    -------
    dict with is_flagged, rolling_median, rolling_mad, threshold, regions
    """
    n = len(widths)

    if n < 3:
        return {
            "is_flagged": np.zeros(n, dtype=bool),
            "rolling_median": widths.copy(),
            "rolling_mad": np.ones(n),
            "threshold": widths.copy(),
            "regions": [],
        }

    # Rolling window size (ensure odd)
    win = max(3, int(n * rolling_window_frac))
    if win % 2 == 0:
        win += 1

    half = win // 2

    rolling_median = np.zeros(n)
    rolling_mad = np.zeros(n)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_vals = widths[lo:hi]
        med = np.median(window_vals)
        rolling_median[i] = med
        rolling_mad[i] = np.median(np.abs(window_vals - med))

    # Floor MAD to avoid zero-threshold
    rolling_mad = np.maximum(rolling_mad, 1.0)

    threshold = rolling_median + threshold_k * rolling_mad
    is_flagged = widths > threshold

    # Merge nearby flagged runs
    if merge_gap > 0 and np.any(is_flagged):
        flagged = is_flagged.copy()
        last_true = -merge_gap - 1
        for i in range(n):
            if flagged[i]:
                # Fill gap if it's small enough
                if 0 < (i - last_true - 1) <= merge_gap:
                    flagged[last_true + 1:i] = True
                last_true = i
        is_flagged = flagged

    # Extract contiguous runs
    regions = []
    in_region = False
    start_idx = 0

    for i in range(n):
        if is_flagged[i] and not in_region:
            start_idx = i
            in_region = True
        elif not is_flagged[i] and in_region:
            _add_region(regions, widths, arc_lengths, rolling_median,
                        start_idx, i - 1, min_region_len)
            in_region = False

    # Close final region
    if in_region:
        _add_region(regions, widths, arc_lengths, rolling_median,
                    start_idx, n - 1, min_region_len)

    return {
        "is_flagged": is_flagged,
        "rolling_median": rolling_median,
        "rolling_mad": rolling_mad,
        "threshold": threshold,
        "regions": regions,
    }


def _add_region(regions, widths, arc_lengths, rolling_median,
                start_idx, end_idx, min_region_len):
    """Helper: create a region dict if the run is long enough."""
    run_len = end_idx - start_idx + 1
    if run_len < min_region_len:
        return

    region_widths = widths[start_idx:end_idx + 1]
    baseline = rolling_median[start_idx:end_idx + 1]
    mean_baseline = float(np.mean(baseline))

    regions.append({
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_arc": float(arc_lengths[start_idx]),
        "end_arc": float(arc_lengths[end_idx]),
        "extent_px": float(arc_lengths[end_idx] - arc_lengths[start_idx]),
        "mean_width": float(np.mean(region_widths)),
        "max_width": float(np.max(region_widths)),
        "baseline_width": mean_baseline,
        "severity": float((np.mean(region_widths) - mean_baseline) / mean_baseline)
                    if mean_baseline > 0 else 0.0,
    })


# ──────────────────────────────────────────────────────────────────
# 6. Main entry point
# ──────────────────────────────────────────────────────────────────

def analyse_mdi(img, neuron_mask, axon_result, nucleus_mask,
                channel=0, step_size=5.0, tangent_window=5,
                max_half_width=80, soma_skip_px=50,
                rolling_window_frac=0.25, threshold_k=2.5,
                min_region_len=3, merge_gap=3,
                debug=False, debug_save_path=None):
    """
    Measure microtubule disorganisation along an axon.

    Uses intensity-weighted second moment (no binary mask) to measure
    axon width at regular intervals along the path, then flags regions
    where width deviates significantly from the local baseline.

    Parameters
    ----------
    img : array, RGB image
    neuron_mask : 2D bool array from trace_neuron
    axon_result : dict from measure_axon (must contain 'path')
    nucleus_mask : 2D bool array
    channel : int, tubulin channel (default 0 = red)
    step_size : float, arc-length sampling interval (px). Default 5.
    tangent_window : int, samples each side for tangent. Default 5.
    max_half_width : int, max perpendicular ray distance. Default 80.
    soma_skip_px : float, arc-length to skip at soma end (px). Default 50.
    rolling_window_frac : float, baseline window fraction. Default 0.25.
    threshold_k : float, anomaly threshold in MADs. Default 2.5.
    min_region_len : int, min flagged samples for a region. Default 3.
    merge_gap : int, max gap to merge flagged runs. Default 3.
    debug : bool, if True produce debug visualisation. Default False.
    debug_save_path : str, path to save debug figure. Default None.

    Returns
    -------
    dict with mdi_ratio, disorganised_length_px, analysed_length_px,
         n_regions, regions, profile, sampled_points, normals
    """
    path = axon_result["path"]
    tubulin_gray = img[:, :, channel].astype(np.float64)

    # Too short to analyse
    if len(path) < 20:
        return _empty_result(axon_result)

    # Step 1: resample path
    sampled_points, cumulative_dist = resample_path(path, step_size)

    if len(sampled_points) < 5:
        return _empty_result(axon_result)

    # Step 2: compute tangents and normals
    tangents, normals = compute_tangents(sampled_points, tangent_window)

    # Step 3: determine soma skip
    # Walk backwards from soma end, find last point inside nucleus mask
    h, w = nucleus_mask.shape
    nucleus_exit_idx = len(sampled_points)  # default: no nucleus overlap
    for i in range(len(sampled_points) - 1, -1, -1):
        ry = int(round(sampled_points[i][0]))
        rx = int(round(sampled_points[i][1]))
        if 0 <= ry < h and 0 <= rx < w and nucleus_mask[ry, rx]:
            nucleus_exit_idx = i
            break

    # Skip at least soma_skip_px past the nucleus, or to the nucleus boundary
    skip_from_end = len(sampled_points) - nucleus_exit_idx
    skip_from_px = int(np.ceil(soma_skip_px / step_size))
    soma_skip = max(skip_from_end, skip_from_px)

    if soma_skip >= len(sampled_points) - 5:
        return _empty_result(axon_result)

    # Step 4: build spread-width profile (no mask dependency)
    profile = build_width_profile(
        sampled_points, normals, cumulative_dist,
        tubulin_gray, max_half_width, soma_skip
    )

    # Step 5: detect MDI regions
    detection = detect_mdi_regions(
        profile["widths"], profile["arc_lengths"],
        rolling_window_frac, threshold_k, min_region_len, merge_gap
    )

    # Compute summary stats
    analysed_length = float(
        profile["arc_lengths"][-1] - profile["arc_lengths"][0]
    ) if len(profile["arc_lengths"]) > 1 else 0.0

    disorganised_length = sum(r["extent_px"] for r in detection["regions"])
    mdi_ratio = disorganised_length / analysed_length if analysed_length > 0 else 0.0

    # Add spatial coordinates to each region
    for region in detection["regions"]:
        si = region["start_idx"]
        ei = region["end_idx"]
        region["start_point"] = (
            int(round(sampled_points[si][0])),
            int(round(sampled_points[si][1]))
        )
        region["end_point"] = (
            int(round(sampled_points[ei][0])),
            int(round(sampled_points[ei][1]))
        )

    # Number of analysed points (before soma skip)
    n_analysed = len(sampled_points) - soma_skip

    result = {
        "mdi_ratio": mdi_ratio,
        "total_axon_length_px": axon_result["length_px"],
        "analysed_length_px": analysed_length,
        "disorganised_length_px": disorganised_length,
        "n_regions": len(detection["regions"]),
        "regions": detection["regions"],
        "profile": {**profile, **detection},
        "sampled_points": sampled_points[:n_analysed],
        "normals": normals[:n_analysed],
        "soma_skip": soma_skip,
    }

    if debug:
        show_mdi_debug(img, result, neuron_mask, nucleus_mask,
                       save_path=debug_save_path, show=(debug_save_path is None))

    return result


def _empty_result(axon_result):
    """Return a minimal result dict when analysis isn't possible."""
    return {
        "mdi_ratio": 0.0,
        "total_axon_length_px": axon_result.get("length_px", 0.0),
        "analysed_length_px": 0.0,
        "disorganised_length_px": 0.0,
        "n_regions": 0,
        "regions": [],
        "profile": {},
        "sampled_points": [],
        "normals": [],
        "soma_skip": 0,
        "path_too_short": True,
    }


# ──────────────────────────────────────────────────────────────────
# 7. Batch wrapper
# ──────────────────────────────────────────────────────────────────

def analyse_all_mdi(img, traces, axon_results, kept_centroids,
                    nuclei_labeled, channel=0, **kwargs):
    """
    Analyse MDI for all traced neurons.

    Parameters
    ----------
    img : array, RGB image
    traces : list of trace dicts from trace_all_neurons
    axon_results : list of dicts from measure_all_axons
    kept_centroids : list of (y, x) centroids
    nuclei_labeled : 2D array, labelled nuclei mask
    channel : int
    **kwargs : passed to analyse_mdi

    Returns
    -------
    list of MDI result dicts
    """
    results = []
    for i, (trace, axon_res, centroid) in enumerate(
            zip(traces, axon_results, kept_centroids)):
        label_at_seed = nuclei_labeled[centroid[0], centroid[1]]
        if label_at_seed > 0:
            nuc_mask = nuclei_labeled == label_at_seed
        else:
            h, w = nuclei_labeled.shape
            yy, xx = np.ogrid[:h, :w]
            nuc_mask = ((yy - centroid[0])**2 + (xx - centroid[1])**2) < 15**2

        mdi = analyse_mdi(img, trace["mask"], axon_res, nuc_mask,
                          channel, **kwargs)
        results.append(mdi)
        print(f"  Neuron {i+1}: MDI ratio = {mdi['mdi_ratio']:.3f}, "
              f"{mdi['n_regions']} regions, "
              f"disorg = {mdi['disorganised_length_px']:.0f} px")

    return results


# ──────────────────────────────────────────────────────────────────
# 8. Visualisation — production
# ──────────────────────────────────────────────────────────────────

def show_mdi(img, mdi_result, nucleus_mask=None, save_path=None, show=True):
    """
    Visualise MDI analysis results (2-panel figure).

    Left:  1D spread-width profile with baseline, threshold, flagged regions.
    Right: spatial overlay — axon path green/red, nucleus cyan.
    """
    profile = mdi_result.get("profile", {})
    if not profile or "widths" not in profile:
        print("  No MDI profile data to plot.")
        return

    arc = profile["arc_lengths"]
    widths = profile["widths"]
    rolling_med = profile.get("rolling_median", widths)
    threshold = profile.get("threshold", widths)
    is_flagged = profile.get("is_flagged", np.zeros(len(widths), dtype=bool))
    regions = mdi_result.get("regions", [])
    sampled_pts = mdi_result.get("sampled_points", [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left panel: spread profile ──
    ax = axes[0]
    ax.plot(arc, widths, 'steelblue', linewidth=1, label='Spread (2\u03c3)')
    ax.plot(arc, rolling_med, 'orange', linewidth=1.5, linestyle='--',
            label='Rolling median')
    ax.plot(arc, threshold, 'red', linewidth=1, linestyle='--',
            alpha=0.7, label='Threshold')

    for region in regions:
        ax.axvspan(region["start_arc"], region["end_arc"],
                   color='red', alpha=0.15)

    ax.set_xlabel("Arc-length from tip (px)")
    ax.set_ylabel("Spread width (px)")
    ax.set_title(f"Spread profile \u2014 MDI ratio: {mdi_result['mdi_ratio']:.2%}")
    ax.legend(fontsize=8)

    # ── Right panel: spatial overlay ──
    ax = axes[1]
    ax.imshow(img)

    if len(sampled_pts) > 0:
        pts = np.array(sampled_pts)

        for i in range(len(pts) - 1):
            colour = 'red' if is_flagged[i] else 'lime'
            ax.plot(pts[i:i+2, 1], pts[i:i+2, 0],
                    colour, linewidth=1.5)

        ax.plot(pts[0, 1], pts[0, 0], 'r+', markersize=12,
                markeredgewidth=2, label='Tip')
        ax.plot(pts[-1, 1], pts[-1, 0], 'c+', markersize=12,
                markeredgewidth=2, label='Start')

    if nucleus_mask is not None:
        ax.contour(nucleus_mask, colors='cyan', linewidths=0.8)

    ax.set_title(f"MDI: {mdi_result['n_regions']} regions, "
                 f"{mdi_result['disorganised_length_px']:.0f} px disorganised")
    ax.legend(fontsize=8)
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# 9. Visualisation — debug
# ──────────────────────────────────────────────────────────────────

def show_mdi_debug(img, mdi_result, neuron_mask, nucleus_mask=None,
                   save_path=None, show=True):
    """
    Multi-panel debug visualisation for sanity-checking MDI analysis.

    Panel 1: Resampled path with normal vectors on the image
    Panel 2: Example intensity profiles at selected cross-sections
    Panel 3: Spread width + peak intensity profiles with baseline
    Panel 4: Spatial flagging — path coloured green/red, spread extent
             drawn as perpendicular lines at each sample point
    """
    profile = mdi_result.get("profile", {})
    if not profile or "widths" not in profile:
        print("  No MDI profile data for debug plot.")
        return

    sampled_pts = mdi_result.get("sampled_points", [])
    normals_list = mdi_result.get("normals", [])
    arc = profile["arc_lengths"]
    widths = profile["widths"]
    rolling_med = profile.get("rolling_median", widths)
    threshold = profile.get("threshold", widths)
    is_flagged = profile.get("is_flagged", np.zeros(len(widths), dtype=bool))
    peak_intensities = profile.get("peak_intensities", np.zeros(len(widths)))
    cross_sections = profile.get("cross_sections", [])
    regions = mdi_result.get("regions", [])

    tubulin_gray = img[:, :, 0].astype(np.float64)
    pts = np.array(sampled_pts) if len(sampled_pts) > 0 else np.empty((0, 2))

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # ── Panel 1: resampled path + normals ──
    ax = axes[0, 0]
    ax.imshow(img)
    if len(pts) > 0:
        ax.plot(pts[:, 1], pts[:, 0], 'lime', linewidth=1, alpha=0.7)
        ax.scatter(pts[:, 1], pts[:, 0], c='lime', s=4, zorder=5)

        # Draw normal vectors every 5th point
        ray_len = 15
        for i in range(0, len(pts), 5):
            if i < len(normals_list):
                ny, nx = normals_list[i]
                y0, x0 = pts[i]
                ax.plot([x0 - ray_len*nx, x0 + ray_len*nx],
                        [y0 - ray_len*ny, y0 + ray_len*ny],
                        'yellow', linewidth=0.5, alpha=0.6)

    if nucleus_mask is not None:
        ax.contour(nucleus_mask, colors='cyan', linewidths=0.8)
    ax.set_title(f"Resampled path ({len(pts)} points) + normal vectors")
    ax.axis("off")

    # ── Panel 2: example intensity profiles ──
    ax = axes[0, 1]
    if len(cross_sections) > 0:
        # Pick ~6 evenly spaced cross-sections to plot
        n_examples = min(6, len(cross_sections))
        indices = np.linspace(0, len(cross_sections) - 1, n_examples, dtype=int)
        colours = plt.cm.viridis(np.linspace(0, 1, n_examples))

        for j, idx in enumerate(indices):
            cs = cross_sections[idx]
            positions = cs.get("positions", np.array([]))
            intensities = cs.get("intensities", np.array([]))
            if len(positions) == 0:
                continue

            bg = cs.get("background", 0)
            profile_vals = intensities - bg

            ax.plot(positions, profile_vals, color=colours[j],
                    linewidth=1, alpha=0.8,
                    label=f"arc={arc[idx]:.0f}")

            # Mark spread extent (weighted centre ± sigma)
            sigma = cs.get("sigma", 0)
            centre = cs.get("weighted_centre", 0)
            if sigma > 0:
                peak = cs.get("peak_intensity", 0)
                marker_level = peak * 0.4  # draw at 40% peak height
                ax.plot([centre - sigma, centre + sigma],
                        [marker_level, marker_level],
                        color=colours[j], linewidth=2, alpha=0.6)
                ax.axvline(centre, color=colours[j], linewidth=0.5,
                           linestyle=':', alpha=0.4)

        ax.set_xlabel("Distance from axon centre (px)")
        ax.set_ylabel("Intensity (bg-subtracted)")
        ax.set_title("Perpendicular intensity profiles (\u03c3 extent marked)")
        ax.legend(fontsize=6, ncol=2)
    else:
        ax.set_title("No cross-section data")

    # ── Panel 3: spread + peak intensity profiles ──
    ax = axes[1, 0]
    ax.plot(arc, widths, 'steelblue', linewidth=1, label='Spread (2\u03c3)')
    ax.plot(arc, rolling_med, 'orange', linewidth=1.5, linestyle='--',
            label='Rolling median')
    ax.plot(arc, threshold, 'red', linewidth=1, linestyle='--',
            alpha=0.7, label='Threshold')

    for region in regions:
        ax.axvspan(region["start_arc"], region["end_arc"],
                   color='red', alpha=0.15)

    ax.set_xlabel("Arc-length from tip (px)")
    ax.set_ylabel("Spread (px)", color='steelblue')
    ax.legend(fontsize=7, loc='upper left')

    # Secondary y-axis for peak intensity
    if len(peak_intensities) > 0 and np.any(peak_intensities > 0):
        ax2 = ax.twinx()
        ax2.plot(arc, peak_intensities, 'gray', linewidth=0.8, alpha=0.6,
                 label='Peak intensity')
        ax2.set_ylabel("Peak intensity", color='gray')
        ax2.legend(fontsize=7, loc='upper right')

    ax.set_title(f"Profiles — MDI ratio: {mdi_result['mdi_ratio']:.2%}")

    # ── Panel 4: spatial flagging with spread extent lines ──
    ax = axes[1, 1]
    ax.imshow(tubulin_gray, cmap="gray")

    if len(pts) > 0 and len(normals_list) > 0:
        # Draw spread extent as perpendicular lines at each sample point
        for i in range(len(pts)):
            if i >= len(cross_sections) or i >= len(normals_list):
                break

            cs = cross_sections[i]
            spread = cs.get("spread", 0)
            if spread < 1:
                continue

            ny, nx = normals_list[i]
            y0, x0 = pts[i]
            half_w = spread / 2.0

            # Draw the spread extent line
            is_flag = is_flagged[i] if i < len(is_flagged) else False
            colour = 'red' if is_flag else 'lime'
            alpha = 0.6 if is_flag else 0.3

            ax.plot([x0 - half_w*nx, x0 + half_w*nx],
                    [y0 - half_w*ny, y0 + half_w*ny],
                    colour, linewidth=0.8, alpha=alpha)

        # Draw path on top
        for i in range(len(pts) - 1):
            colour = 'red' if (i < len(is_flagged) and is_flagged[i]) else 'lime'
            ax.plot(pts[i:i+2, 1], pts[i:i+2, 0], colour, linewidth=1.5)

        # Tip and start markers
        ax.plot(pts[0, 1], pts[0, 0], 'r+', markersize=12,
                markeredgewidth=2)
        ax.plot(pts[-1, 1], pts[-1, 0], 'c+', markersize=12,
                markeredgewidth=2)

    if nucleus_mask is not None:
        ax.contour(nucleus_mask, colors='cyan', linewidths=0.8)

    ax.set_title(f"Spread extent (green=normal, red=flagged): "
                 f"{len(regions)} regions")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved debug: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
