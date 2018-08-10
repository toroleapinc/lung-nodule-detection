"""CT scan preprocessing."""
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

def load_scan(path):
    img = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(img)
    spacing = np.array(img.GetSpacing())[::-1]
    return array, spacing

def normalize_hu(image, min_hu=-1000, max_hu=400):
    image = np.clip(image, min_hu, max_hu)
    return ((image - min_hu) / (max_hu - min_hu)).astype(np.float32)

def resample(image, spacing, new_spacing=(1.0, 1.0, 1.0)):
    resize_factor = spacing / np.array(new_spacing)
    new_shape = np.round(image.shape * resize_factor).astype(int)
    return zoom(image, new_shape / np.array(image.shape), order=1)

def preprocess_scan(path, target_spacing=(1.0, 1.0, 1.0)):
    image, spacing = load_scan(path)
    image = resample(image, spacing, target_spacing)
    return normalize_hu(image)
