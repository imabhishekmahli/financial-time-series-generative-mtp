import math
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============ CONFIG ============

ASSET_NAME_DEFAULT = "sp500"   # can override via command line argument
SEQ_LEN = 60

BATCH_SIZE = 64
EPOCHS     = 30
LR         = 1e-3
TIMESTEPS  = 100
NUM_GEN_SAMPLES = 8

DEVICE = "cpu"
print("Using device:", DEVICE)


# ============ DATASET ============

class TimeSeriesDataset(Dataset):
    def __init__(self, seqs):
        """
        seqs: [N, T]
        """
        x = seqs.astype(np.float32)
        self.mean = x.mean()
        self.std  = x.std() + 1e-8
        x_norm = (x - self.mean) / self.std
        self.x = torch.tensor(x_norm, dtype=torch.float32)  # [N, T]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]   # [T]


# ============ DIFFUSION SETUP ============

def make_beta_schedule(num_timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, num_timesteps)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        emb = self.lin1(emb)
        emb = self.act(emb)
        emb = self.lin2(emb)
        return emb


class MLPDiffusionModel(nn.Module):
    def __init__(self, seq_len, time_emb_dim=64, hidden_dim=128):
        super().__init__()
        self.seq_len = seq_len
        self.time_emb = TimeEmbedding(time_emb_dim)

        self.fc1 = nn.Linear(seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_time = nn.Linear(time_emb_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, seq_len)
        self.act = nn.SiLU()

    def forward(self, x, t):
        # x: [B, T], t: [B]
        t_emb = self.time_emb(t)           # [B, time_emb_dim]
        h = self.fc1(x)                    # [B, H]
        h = self.act(h)
        t_h = self.fc_time(t_emb)          # [B, H]
        h = h + t_h
        h = self.act(self.fc2(h))
        out = self.fc3(h)                  # [B, T]
        return out


def q_sample(x0, t_scalar, alpha_bars, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    alpha_bar_t = alpha_bars[t_scalar]
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise


# ============ TRAINING ============

def train_ddpm_for_asset(asset_name):
    seq_file = f"{asset_name}_returns_sequences.npy"
    print(f"\n=== Training DDPM for asset: {asset_name} ===")
    print(f"Loading sequences from {seq_file}")
    seqs = np.load(seq_file)   # [N, T]
    dataset = TimeSeriesDataset(seqs)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    seq_len = seqs.shape[1]

    betas = make_beta_schedule(TIMESTEPS).to(DEVICE)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    model = MLPDiffusionModel(seq_len=seq_len).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for batch in dataloader:
            batch = batch.to(DEVICE)   # [B,T]
            B = batch.size(0)

            t_scalar = torch.randint(0, TIMESTEPS, (1,), device=DEVICE).item()
            t_batch = torch.full((B,), t_scalar, device=DEVICE, dtype=torch.long)

            noise = torch.randn_like(batch)
            x_t = q_sample(batch, t_scalar, alpha_bars, noise=noise)

            noise_pred = model(x_t, t_batch)
            loss = mse(noise_pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * B
            n_samples += B

        avg_loss = total_loss / n_samples
        print(f"[{asset_name}] Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")

    return model, dataset, alpha_bars, betas, alphas


@torch.no_grad()
def p_sample(model, x_t, t_scalar, betas, alphas, alpha_bars):
    beta_t = betas[t_scalar]
    alpha_t = alphas[t_scalar]
    alpha_bar_t = alpha_bars[t_scalar]

    B = x_t.size(0)
    t_batch = torch.full((B,), t_scalar, device=x_t.device, dtype=torch.long)

    noise_pred = model(x_t, t_batch)
    x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

    mean = torch.sqrt(alpha_t) * x0_pred + torch.sqrt(1 - alpha_t) * noise_pred

    z = torch.randn_like(x_t)
    if t_scalar > 0:
        x_prev = mean + torch.sqrt(beta_t) * z
    else:
        x_prev = mean
    return x_prev


@torch.no_grad()
def generate_samples(model, seq_len, betas, alphas, alpha_bars, num_samples=NUM_GEN_SAMPLES):
    model.eval()
    x_t = torch.randn(num_samples, seq_len, device=DEVICE)
    for t in reversed(range(TIMESTEPS)):
        x_t = p_sample(model, x_t, t, betas, alphas, alpha_bars)
    return x_t.cpu().numpy()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        asset_name = sys.argv[1]
    else:
        asset_name = ASSET_NAME_DEFAULT

    model, dataset, alpha_bars, betas, alphas = train_ddpm_for_asset(asset_name)

    print(f"Generating synthetic sequences for {asset_name} ...")
    seq_len = dataset.x.shape[1]
    gen_norm = generate_samples(model, seq_len, betas, alphas, alpha_bars)
    real_norm = dataset.x[:NUM_GEN_SAMPLES].numpy()

    # Save
    real_out = f"{asset_name}_diffusion_real_sequences.npy"
    gen_out  = f"{asset_name}_diffusion_generated_sequences.npy"
    np.save(real_out, real_norm)
    np.save(gen_out, gen_norm)
    print("Saved:", real_out, "and", gen_out)

    # Plot comparison
    import numpy as _np
    fig, axes = plt.subplots(NUM_GEN_SAMPLES, 2, figsize=(10, 2*NUM_GEN_SAMPLES))
    if NUM_GEN_SAMPLES == 1:
        axes = _np.expand_dims(axes, 0)

    for i in range(NUM_GEN_SAMPLES):
        axes[i, 0].plot(real_norm[i], color="blue")
        axes[i, 0].set_title(f"{asset_name.upper()} Real seq {i+1}")
        axes[i, 1].plot(gen_norm[i], color="green")
        axes[i, 1].set_title(f"{asset_name.upper()} Diffusion seq {i+1}")

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"{asset_name}_diffusion_real_vs_generated_{ts}.png"
    plt.savefig(out_file, dpi=150)
    print("Saved plot:", out_file)
    plt.show()
