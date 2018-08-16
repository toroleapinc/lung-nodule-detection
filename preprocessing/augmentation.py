"""Data augmentation for 3D medical images."""
import numpy as np
from scipy.ndimage import rotate, shift

def random_flip(image, mask, axis=0):
    if np.random.random() > 0.5:
        image = np.flip(image, axis=axis)
        mask = np.flip(mask, axis=axis)
    return image.copy(), mask.copy()

def random_rotate(image, mask, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    image = rotate(image, angle, axes=(1, 2), reshape=False, order=1)
    mask = rotate(mask, angle, axes=(1, 2), reshape=False, order=0)
    return image, mask

def augment(image, mask):
    image, mask = random_flip(image, mask, 0)
    image, mask = random_flip(image, mask, 1)
    if np.random.random() > 0.5:
        image, mask = random_rotate(image, mask)
    return image, mask
