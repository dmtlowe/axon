import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import tifffile
import roifile

# ── USER DEFINED PATHS ────────────────────────────────────────────────────────
IMAGE_PATH = r"C:\Uni\Projects\Axon_Analysis\data\PAVNCD1_1-5-_8bit_panel.tif"
ROI_FOLDER = r"C:\Uni\Projects\Axon_Analysis\data\PAVNCD1_1-5-_ROI"
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path):
    img = tifffile.imread(path)
    # If multichannel, take max projection or select channel
    if img.ndim == 3 and img.shape[0] < 10:
        # Likely (channels, H, W) — take first channel or adjust as needed
        img = img[0]
    elif img.ndim == 3:
        # Likely (H, W, channels)
        img = img[:, :, 0]
    return img

def normalise(img):
    """Stretch to 0-1 for display."""
    img = img.astype(float)
    return (img - img.min()) / (img.max() - img.min())

def load_all_rois(folder):
    roi_files = sorted([
        f for f in os.listdir(folder) if f.endswith('.roi')
    ])
    rois = []
    for fname in roi_files:
        try:
            roi = roifile.roiread(os.path.join(folder, fname))
            rois.append((fname, roi))
        except Exception as e:
            print(f"  Could not load {fname}: {e}")
    return rois

def plot_roi(ax, roi, color):
    try:
        coords = roi.coordinates()
        # coords are (x, y) — matplotlib wants x on horizontal axis
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1.5)
    except Exception:
        # Some ROIs (e.g. straight lines) expose differently
        try:
            x1, y1, x2, y2 = roi.left, roi.top, roi.left + roi.width, roi.top + roi.height
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.5)
        except Exception as e:
            print(f"  Could not plot ROI: {e}")

def main():
    print(f"Loading image: {IMAGE_PATH}")
    img = normalise(load_image(IMAGE_PATH))

    print(f"Loading ROIs from: {ROI_FOLDER}")
    rois = load_all_rois(ROI_FOLDER)
    print(f"  Found {len(rois)} ROI files")

    # Generate a colour per ROI for distinguishability
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(rois))]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray', interpolation='none')

    for i, (fname, roi) in enumerate(rois):
        plot_roi(ax, roi, color=colors[i])

    ax.set_title(f"{os.path.basename(IMAGE_PATH)}\n{len(rois)} ROIs overlaid", fontsize=11)
    ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()