import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DEVICE = "cpu"
print("Using device:", DEVICE)

SEQ_LEN = 60
INPUT_LEN = SEQ_LEN - 1  # first 59 as input


# ====== Dataset ======
class SeqPredictDataset(Dataset):
    def __init__(self, seqs):
        """
        seqs: [N, T]
        Input: seq[:, :T-1]
        Label: sign(seq[:, T-1]) > 0  -> 1 else 0
        """
        if seqs.ndim == 3:
            seqs = seqs[:, :, 0]
        self.X = seqs[:, :INPUT_LEN]           # [N, 59]
        last_vals = seqs[:, -1]
        self.y = (last_vals > 0).astype(np.int64)

        # normalize X globally
        mean = self.X.mean()
        std  = self.X.std() + 1e-8
        self.X = (self.X - mean) / std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]        # [59]
        y = self.y[idx]
        # LSTM expects [T,F]. We'll treat it as [59,1]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# ====== LSTM Model ======
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: [B, T, 1]
        out, (h_n, c_n) = self.lstm(x)
        # use last hidden state
        h_last = h_n[-1]   # [B, H]
        logits = self.fc(h_last)
        return logits


def train_and_eval_predictive_score(real_file, gen_file, model_name="TimeGAN"):
    print(f"\n=== Predictive Score for {model_name} ===")

    real = np.load(real_file)
    gen  = np.load(gen_file)

    # Build datasets
    real_dataset = SeqPredictDataset(real)
    gen_dataset  = SeqPredictDataset(gen)

    # Split gen into train/val, real is only test here
    idx = np.arange(len(gen_dataset))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)

    train_subset = torch.utils.data.Subset(gen_dataset, train_idx)
    val_subset   = torch.utils.data.Subset(gen_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(real_dataset, batch_size=32, shuffle=False)

    model = LSTMPredictor().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---- Train on GEN only ----
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        avg_loss = total_loss / n

        # quick val
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                logits = model(x)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(y.numpy())
        val_acc = accuracy_score(val_true, val_preds)
        print(f"Epoch {epoch+1}/{EPOCHS} - TrainLoss: {avg_loss:.4f} - ValAcc: {val_acc:.4f}")

    # ---- Test on REAL (TSTR) ----
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(y.numpy())
    test_acc = accuracy_score(test_true, test_preds)
    test_err = 1.0 - test_acc
    print(f"TSTR Test Accuracy on REAL ({model_name} synthetic → REAL test): {test_acc:.4f}")
    print(f"Predictive Score (error, lower better): {test_err:.4f}")
    return test_err


if __name__ == "__main__":
    # TimeGAN predictive score
    ps_timegan = train_and_eval_predictive_score(
        "timegan_real_sequences.npy",
        "timegan_generated_sequences.npy",
        model_name="TimeGAN"
    )

    # Diffusion predictive score
    ps_diff = train_and_eval_predictive_score(
        "diffusion_real_sequences.npy",
        "diffusion_generated_sequences.npy",
        model_name="Diffusion"
    )

    print("\nSummary Predictive Scores (lower is better):")
    print(f"TimeGAN:  {ps_timegan:.4f}")
    print(f"Diffusion:{ps_diff:.4f}")
