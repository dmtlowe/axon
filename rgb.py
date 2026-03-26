"""
Show R, G, B channels of an image side by side.

Usage:
    python show_channels.py image.tif
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io
from pathlib import Path


def show_channels(img_path):
    img = io.imread(str(img_path))
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img)
    axes[0].set_title("Composite")
    
    for i, name in enumerate(["Red", "Green", "Blue"]):
        axes[i+1].imshow(img[:, :, i], cmap="gray")
        axes[i+1].set_title(name)
    
    for ax in axes:
        ax.axis("off")
    
    plt.tight_layout()
    
    out = Path(img_path).with_suffix(".channels.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_channels.py image.tif")
        sys.exit(1)
    show_channels(sys.argv[1])