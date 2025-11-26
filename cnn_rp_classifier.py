import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

DEVICE = "cpu"
print("Using CPU for training.")

# ===========================
# Dataset
# ===========================

class RPImageDataset(Dataset):
    def __init__(self, img_dir_real, img_dir_fake, img_size=64):
        self.img_paths = []
        self.labels = []

        real_paths = sorted([os.path.join(img_dir_real, f) for f in os.listdir(img_dir_real) if f.endswith(".png")])
        fake_paths = sorted([os.path.join(img_dir_fake, f) for f in os.listdir(img_dir_fake) if f.endswith(".png")])

        for p in real_paths:
            self.img_paths.append(p)
            self.labels.append(0)

        for p in fake_paths:
            self.img_paths.append(p)
            self.labels.append(1)

        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("L")
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img) / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# ===========================
# CNN Model
# ===========================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


# ===========================
# Training Function
# ===========================

def train_classifier(real_dir, fake_dir, title="Model"):
    print(f"\nTraining classifier for: {title}")

    dataset = RPImageDataset(real_dir, fake_dir)
    train_data, test_data = train_test_split(list(range(len(dataset))), test_size=0.2, shuffle=True)

    train_loader = DataLoader([dataset[i] for i in train_data], batch_size=16, shuffle=True)
    test_loader = DataLoader([dataset[i] for i in test_data], batch_size=16, shuffle=False)

    model = SimpleCNN().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---- Training ----
    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/10 - Loss: {total_loss:.4f}")

    # ---- Evaluation ----
    model.eval()
    preds = []
    truths = []

    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        preds.extend(pred)
        truths.extend(y.numpy())

    acc = accuracy_score(truths, preds)
    cm = confusion_matrix(truths, preds)

    print(f"\n{title} Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return acc, cm


# ===========================
# MAIN
# ===========================

if __name__ == "__main__":

    acc_d, cm_d = train_classifier(
        "rp_diffusion_real",
        "rp_diffusion_generated",
        title="Diffusion vs Real"
    )

    acc_t, cm_t = train_classifier(
        "rp_timegan_real",
        "rp_timegan_generated",
        title="TimeGAN vs Real"
    )

    print("\nFinal Results:")
    print(f"Diffusion Accuracy: {acc_d}")
    print(f"TimeGAN Accuracy: {acc_t}")
