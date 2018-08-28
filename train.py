"""Train U-Net on LUNA16 data."""
import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import UNet3D

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        p, t = pred.view(-1), target.view(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)

class LungDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        d = os.path.join(data_dir, split)
        self.files = sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.npz')])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return torch.from_numpy(data['image'][np.newaxis].astype(np.float32)), torch.from_numpy(data['mask'][np.newaxis].astype(np.float32))

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = DiceLoss()
    train_loader = DataLoader(LungDataset(args.data, 'train'), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(LungDataset(args.data, 'val'), batch_size=1)
    os.makedirs('checkpoints', exist_ok=True)
    best_dice = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                pred = (model(imgs) > 0.5).float()
                dice = (2 * (pred * masks).sum()) / (pred.sum() + masks.sum() + 1e-8)
                val_dice += dice.item()
        avg = val_dice / max(len(val_loader), 1)
        if avg > best_dice:
            best_dice = avg
            torch.save(model.state_dict(), 'checkpoints/best.pt')
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: loss={train_loss/len(train_loader):.4f}, val_dice={avg:.4f}")
    print(f"Best val dice: {best_dice:.4f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/processed')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    train(p.parse_args())
