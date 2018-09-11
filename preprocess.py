"""Preprocess LUNA16 data."""
import argparse, os, glob
import numpy as np
import pandas as pd
from preprocessing import preprocess_scan

def extract_patches(image, annotations, patch_size=64):
    patches = []
    for _, row in annotations.iterrows():
        z, y, x = int(row['coordZ']), int(row['coordY']), int(row['coordX'])
        radius = int(row['diameter_mm'] / 2) + 5
        z1 = max(0, z - patch_size // 2)
        y1 = max(0, y - patch_size // 2)
        x1 = max(0, x - patch_size // 2)
        patch = image[z1:z1+patch_size, y1:y1+patch_size, x1:x1+patch_size]
        if patch.shape == (patch_size, patch_size, patch_size):
            mask = np.zeros_like(patch)
            zz, yy, xx = np.ogrid[:patch_size, :patch_size, :patch_size]
            ctr = patch_size // 2
            mask[np.sqrt((zz-ctr)**2 + (yy-ctr)**2 + (xx-ctr)**2) <= radius] = 1
            patches.append((patch, mask))
    return patches

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/luna16')
    p.add_argument('--output', default='data/processed')
    args = p.parse_args()
    for split in ['train', 'val']:
        os.makedirs(os.path.join(args.output, split), exist_ok=True)
    annotations = pd.read_csv(os.path.join(args.data, 'annotations.csv'))
    scans = glob.glob(os.path.join(args.data, '*.mhd'))
    print(f"Found {len(scans)} scans, {len(annotations)} annotations")
    for i, path in enumerate(scans):
        uid = os.path.basename(path).replace('.mhd', '')
        ann = annotations[annotations['seriesuid'] == uid]
        if len(ann) == 0: continue
        image = preprocess_scan(path)
        patches = extract_patches(image, ann)
        split = 'val' if i % 5 == 0 else 'train'
        for j, (patch, mask) in enumerate(patches):
            np.savez(os.path.join(args.output, split, f'{uid}_{j}.npz'), image=patch, mask=mask)
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(scans)}")

if __name__ == '__main__':
    main()
