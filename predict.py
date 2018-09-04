"""Run inference on a CT scan."""
import argparse
import numpy as np
import torch
from model import UNet3D
from preprocessing import preprocess_scan

def predict(scan_path, model_path='checkpoints/best.pt', patch_size=64, stride=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    image = preprocess_scan(scan_path)
    print(f"Scan shape: {image.shape}")
    d, h, w = image.shape
    output = np.zeros_like(image)
    counts = np.zeros_like(image)
    # TODO: proper sliding window with overlap averaging
    for z in range(0, d - patch_size + 1, stride):
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                with torch.no_grad():
                    pred = model(torch.from_numpy(patch[np.newaxis, np.newaxis]).float().to(device)).cpu().numpy()[0, 0]
                output[z:z+patch_size, y:y+patch_size, x:x+patch_size] += pred
                counts[z:z+patch_size, y:y+patch_size, x:x+patch_size] += 1
    counts[counts == 0] = 1
    binary = ((output / counts) > 0.5).astype(np.uint8)
    print(f"Found {binary.sum()} positive voxels")
    return binary
# TODO: add FROC metric evaluation

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--model', default='checkpoints/best.pt')
    a = p.parse_args()
    predict(a.input, a.model)
