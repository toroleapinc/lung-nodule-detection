# Lung Nodule Detection

U-Net based segmentation for detecting lung nodules in CT scans. Uses the LUNA16 dataset.

### Setup
```
pip install -r requirements.txt
```

Download LUNA16 data from https://luna16.grand-challenge.org/ and extract to `data/luna16/`.

### Usage
```
python preprocess.py --data data/luna16/
python train.py --epochs 50
python predict.py --input scan.mhd --model checkpoints/best.pt
```

Got ~0.82 dice score on validation after 50 epochs. The 3D U-Net works better than slice-by-slice 2D.
