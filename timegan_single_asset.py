import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============ CONFIG ============

ASSET_NAME_DEFAULT = "sp500"   # override with argv[1]
H_DIM = 32
Z_DIM = 16
NUM_LAYERS = 1
BATCH_SIZE = 32

EPOCHS_AUTO = 10
EPOCHS_SUP  = 10
EPOCHS_ADV  = 20

LR = 1e-3
DEVICE = "cpu"
print("Using DEVICE:", DEVICE)


# ============ DATASET ============

class SeqDataset(Dataset):
    def __init__(self, seqs):
        """
        seqs: [N, T] or [N, T, 1]
        """
        if seqs.ndim == 2:
            seqs = seqs[:, :, None]
        x = seqs.astype(np.float32)
        self.mean = x.mean()
        self.std  = x.std() + 1e-8
        x_norm = (x - self.mean) / self.std
        self.x = torch.tensor(x_norm, dtype=torch.float32)  # [N,T,1]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


# ============ TIMEGAN MODULES ============

class Embedder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=x_dim, hidden_size=h_dim,
                          num_layers=num_layers, batch_first=True)
        self.lin = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        h, _ = self.gru(x)
        z = self.lin(h)
        return z


class Recovery(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=z_dim, hidden_size=h_dim,
                          num_layers=num_layers, batch_first=True)
        self.lin = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        h, _ = self.gru(z)
        x_tilde = self.lin(h)
        return x_tilde


class Generator(nn.Module):
    def __init__(self, z_dim, h_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=z_dim, hidden_size=h_dim,
                          num_layers=num_layers, batch_first=True)
        self.lin = nn.Linear(h_dim, z_dim)

    def forward(self, z):
        h, _ = self.gru(z)
        e_hat = self.lin(h)
        return e_hat


class Supervisor(nn.Module):
    def __init__(self, z_dim, h_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=z_dim, hidden_size=h_dim,
                          num_layers=num_layers, batch_first=True)
        self.lin = nn.Linear(h_dim, z_dim)

    def forward(self, z):
        h, _ = self.gru(z)
        h_hat = self.lin(h)
        return h_hat


class Discriminator(nn.Module):
    def __init__(self, z_dim, h_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=z_dim, hidden_size=h_dim,
                          num_layers=num_layers, batch_first=True)
        self.lin = nn.Linear(h_dim, 1)

    def forward(self, z):
        h, _ = self.gru(z)
        h_last = h[:, -1, :]
        logits = self.lin(h_last)
        return logits.squeeze(-1)


mse_loss = nn.MSELoss()
bce_logits = nn.BCEWithLogitsLoss()


def autoencoder_loss(embedder, recovery, x):
    e = embedder(x)
    x_tilde = recovery(e)
    return mse_loss(x_tilde, x)


def supervised_loss(supervisor, embedder, x):
    e = embedder(x)
    h_hat = supervisor(e)
    return mse_loss(h_hat[:, :-1, :], e[:, 1:, :])


# ============ TRAINING LOOP ============

def train_timegan_for_asset(asset_name):
    seq_file = f"{asset_name}_returns_sequences.npy"
    print(f"\n=== Training TimeGAN for asset: {asset_name} ===")
    print(f"Loading sequences from {seq_file}")
    seqs = np.load(seq_file)   # [N,T]
    dataset = SeqDataset(seqs)

    indices = np.arange(len(dataset))
    train_idx, _ = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

    x_dim = 1

    embedder = Embedder(x_dim, H_DIM, Z_DIM, NUM_LAYERS).to(DEVICE)
    recovery = Recovery(Z_DIM, H_DIM, x_dim, NUM_LAYERS).to(DEVICE)
    generator = Generator(Z_DIM, H_DIM, NUM_LAYERS).to(DEVICE)
    supervisor = Supervisor(Z_DIM, H_DIM, NUM_LAYERS).to(DEVICE)
    discriminator = Discriminator(Z_DIM, H_DIM, NUM_LAYERS).to(DEVICE)

    optE = torch.optim.Adam(embedder.parameters(), lr=LR)
    optR = torch.optim.Adam(recovery.parameters(), lr=LR)
    optG = torch.optim.Adam(generator.parameters(), lr=LR)
    optS = torch.optim.Adam(supervisor.parameters(), lr=LR)
    optD = torch.optim.Adam(discriminator.parameters(), lr=LR)

    # ---- Phase 1: Autoencoder ----
    print("\n[Phase 1] Autoencoder pretraining...")
    for epoch in range(EPOCHS_AUTO):
        embedder.train()
        recovery.train()
        total_loss = 0.0
        n_samples = 0
        for x in loader:
            x = x.to(DEVICE)
            loss = autoencoder_loss(embedder, recovery, x)
            optE.zero_grad()
            optR.zero_grad()
            loss.backward()
            optE.step()
            optR.step()
            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
        print(f"[{asset_name}] Auto Epoch {epoch+1}/{EPOCHS_AUTO} - Recon Loss: {total_loss/n_samples:.6f}")

    # ---- Phase 2: Supervisor ----
    print("\n[Phase 2] Supervisor pretraining...")
    for epoch in range(EPOCHS_SUP):
        embedder.train()
        supervisor.train()
        total_loss = 0.0
        n_samples = 0
        for x in loader:
            x = x.to(DEVICE)
            loss_s = supervised_loss(supervisor, embedder, x)
            optE.zero_grad()
            optS.zero_grad()
            loss_s.backward()
            optE.step()
            optS.step()
            total_loss += loss_s.item() * x.size(0)
            n_samples += x.size(0)
        print(f"[{asset_name}] Sup Epoch {epoch+1}/{EPOCHS_SUP} - Sup Loss: {total_loss/n_samples:.6f}")

    # ---- Phase 3: Adversarial / Joint ----
    print("\n[Phase 3] Adversarial / joint training...")
    for epoch in range(EPOCHS_ADV):
        for x in loader:
            x = x.to(DEVICE)
            B, T, _ = x.shape

            # --- Train Discriminator ---
            embedder.eval()
            generator.eval()
            supervisor.eval()
            discriminator.train()

            e_real = embedder(x).detach()
            z_noise = torch.randn(B, T, Z_DIM, device=DEVICE)
            e_hat = generator(z_noise).detach()
            h_hat = supervisor(e_hat).detach()

            d_real = discriminator(e_real)
            d_fake = discriminator(h_hat)

            d_loss_real = bce_logits(d_real, torch.ones_like(d_real))
            d_loss_fake = bce_logits(d_fake, torch.zeros_like(d_fake))
            d_loss = d_loss_real + d_loss_fake

            optD.zero_grad()
            d_loss.backward()
            optD.step()

            # --- Train G, S, E, R ---
            embedder.train()
            generator.train()
            supervisor.train()
            recovery.train()
            discriminator.eval()

            z_noise = torch.randn(B, T, Z_DIM, device=DEVICE)
            e_hat = generator(z_noise)
            h_hat = supervisor(e_hat)
            d_fake_for_g = discriminator(h_hat)
            g_adv_loss = bce_logits(d_fake_for_g, torch.ones_like(d_fake_for_g))

            s_loss = supervised_loss(supervisor, embedder, x)
            ae_loss = autoencoder_loss(embedder, recovery, x)

            g_loss = g_adv_loss + 10 * s_loss + ae_loss

            optG.zero_grad()
            optS.zero_grad()
            optE.zero_grad()
            optR.zero_grad()
            g_loss.backward()
            optG.step()
            optS.step()
            optE.step()
            optR.step()

        print(f"[{asset_name}] Adv Epoch {epoch+1}/{EPOCHS_ADV} - D_loss: {d_loss.item():.4f} - G_loss: {g_loss.item():.4f}")

    print(f"TimeGAN training complete for {asset_name}.")
    return embedder, recovery, generator, supervisor, dataset


@torch.no_grad()
def generate_synthetic(generator, supervisor, recovery, num_samples, seq_len):
    generator.eval()
    supervisor.eval()
    recovery.eval()
    z_noise = torch.randn(num_samples, seq_len, Z_DIM, device=DEVICE)
    e_hat = generator(z_noise)
    h_hat = supervisor(e_hat)
    x_hat = recovery(h_hat)
    return x_hat.cpu().numpy()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        asset_name = sys.argv[1]
    else:
        asset_name = ASSET_NAME_DEFAULT

    embedder, recovery, generator, supervisor, dataset = train_timegan_for_asset(asset_name)

    NUM_GEN_SAMPLES = 8
    print(f"Generating synthetic sequences for {asset_name} with TimeGAN...")
    seq_len = dataset.x.shape[1]
    gen = generate_synthetic(generator, supervisor, recovery, NUM_GEN_SAMPLES, seq_len)
    real = dataset.x[:NUM_GEN_SAMPLES].numpy()

    real_out = f"{asset_name}_timegan_real_sequences.npy"
    gen_out  = f"{asset_name}_timegan_generated_sequences.npy"
    np.save(real_out, real)
    np.save(gen_out, gen)
    print("Saved:", real_out, "and", gen_out)

    # Plot comparison
    fig, axes = plt.subplots(NUM_GEN_SAMPLES, 2, figsize=(10, 2*NUM_GEN_SAMPLES))
    import numpy as _np
    if NUM_GEN_SAMPLES == 1:
        axes = _np.expand_dims(axes, 0)

    for i in range(NUM_GEN_SAMPLES):
        axes[i, 0].plot(real[i, :, 0], color="blue")
        axes[i, 0].set_title(f"{asset_name.upper()} Real seq {i+1}")
        axes[i, 1].plot(gen[i, :, 0], color="green")
        axes[i, 1].set_title(f"{asset_name.upper()} TimeGAN seq {i+1}")

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"{asset_name}_timegan_real_vs_generated_{ts}.png"
    plt.savefig(out_file, dpi=150)
    print("Saved plot:", out_file)
    plt.show()
