"""
Split a multi-panel TIF into individual panels based on user-defined grid.
"""

import numpy as np
import tifffile
from pathlib import Path

def split_panels(filepath, rows, cols, output_dir=None):
    """
    Split a TIF image into a grid of panels and save each one.
    """
    filepath = Path(filepath)
    img = tifffile.imread(str(filepath))
    
    # Get dimensions
    h, w = img.shape[:2]  # Handles both grayscale and multi-channel
    ph = h // rows
    pw = w // cols
    
    if output_dir is None:
        output_dir = filepath.parent / f"{filepath.stem}_panels"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing: {filepath.name}")
    print(f"Image Size: {w}x{h} px")
    print(f"Grid: {rows} rows × {cols} columns")
    print(f"Panel Size: {pw}x{ph} px")
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            # Slicing the numpy array for the specific panel
            panel = img[r*ph : (r+1)*ph, c*pw : (c+1)*pw]
            
            # Using row/col indices in filename often makes it easier to find specific spots
            out_path = output_dir / f"{filepath.stem}_R{r+1}_C{c+1}.tif"
            tifffile.imwrite(str(out_path), panel)
            count += 1
    
    print(f"Successfully saved {count} panels to: {output_dir}")

if __name__ == "__main__":
    import sys
    
    # Check if a filename was provided as an argument
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        
        try:
            # User Inputs
            num_rows = int(input("Enter number of rows: "))
            num_cols = int(input("Enter number of columns: "))
            
            split_panels(target_file, num_rows, num_cols)
            
        except ValueError:
            print("Error: Please enter whole numbers for rows and columns.")
    else:
        print("Usage: python panel_splitter.py image.tif")