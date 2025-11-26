import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

"""
Usage:
    python3 plot_real_vs_diffusion.py sp500
    python3 plot_real_vs_diffusion.py goog
    python3 plot_real_vs_diffusion.py corn

This will:
- load {asset}_diffusion_real_sequences.npy
- load {asset}_diffusion_generated_sequences.npy
- plot a few real vs diffusion sequences on the same graph (aligned)
- save to {asset}_real_vs_diffusion_overlay_YYYYmmdd_HHMMSS.png
"""

DEFAULT_ASSET = "sp500"
NUM_PLOTS = 4  # how many sequences to overlay


def load_sequences(asset_name):
    real_file = f"{asset_name}_diffusion_real_sequences.npy"
    gen_file  = f"{asset_name}_diffusion_generated_sequences.npy"

    print(f"Loading real sequences from: {real_file}")
    print(f"Loading diffusion sequences from: {gen_file}")

    real = np.load(real_file)   # shape [N, T] or [N, T, 1]
    gen  = np.load(gen_file)

    # Squeeze last dim if present
    if real.ndim == 3:
        real = real[:, :, 0]
    if gen.ndim == 3:
        gen = gen[:, :, 0]

    print("Real shape:", real.shape)
    print("Gen  shape:", gen.shape)

    # Ensure we can index safely
    n = min(real.shape[0], gen.shape[0])
    real = real[:n]
    gen  = gen[:n]

    return real, gen


def plot_overlays(asset_name, real, gen, num_plots=NUM_PLOTS):
    n, T = real.shape
    num_plots = min(num_plots, n)

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2.5 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]

    x = np.arange(T)

    for i in range(num_plots):
        ax = axes[i]
        ax.plot(x, real[i], label="Real", linewidth=1.5)
        ax.plot(x, gen[i], label="Diffusion", linewidth=1.0, alpha=0.8)
        ax.set_ylabel("Value")
        ax.set_title(f"{asset_name.upper()} - Sequence {i+1}")
        ax.grid(True)
        if i == 0:
            ax.legend()

    axes[-1].set_xlabel("Time step")

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"{asset_name}_real_vs_diffusion_overlay_{ts}.png"
    plt.savefig(out_file, dpi=150)
    print("Saved overlay plot to:", out_file)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        asset_name = sys.argv[1]
    else:
        asset_name = DEFAULT_ASSET

    real, gen = load_sequences(asset_name)
    plot_overlays(asset_name, real, gen)
