import os
import numpy as np
import matplotlib.pyplot as plt

# -------- GAF helper --------

def gaf_from_sequence(seq, method="summation"):
    """
    seq: 1D numpy array of shape [T]
    returns: GAF image [T, T] in [-1,1]
    """
    x = np.asarray(seq, dtype=float)
    # Min-max scale to [-1,1]
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        x_scaled = np.zeros_like(x)
    else:
        x_scaled = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    x_scaled = np.clip(x_scaled, -1.0, 1.0)

    # Polar encoding
    phi = np.arccos(x_scaled)       # [T]
    r = np.linspace(0.0, 1.0, len(x_scaled))

    # GAF (summation or difference)
    if method == "summation":
        gaf = np.cos(phi[:, None] + phi[None, :])
    else:
        gaf = np.sin(phi[:, None] - phi[None, :])
    return gaf

def save_gaf_images_from_npy(npy_file, out_dir, max_images=20, method="summation", prefix=""):
    print(f"Loading {npy_file} ...")
    arr = np.load(npy_file)  # shapes: [N,T] or [N,T,1]
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    N, T = arr.shape
    print(f"Array shape: {arr.shape}")

    os.makedirs(out_dir, exist_ok=True)
    num = min(N, max_images)
    for i in range(num):
        seq = arr[i]
        gaf = gaf_from_sequence(seq, method=method)
        plt.figure(figsize=(3,3))
        plt.imshow(gaf, cmap="viridis", origin="lower")
        plt.axis("off")
        fname = os.path.join(out_dir, f"{prefix}{i:03d}.png")
        plt.savefig(fname, bbox_inches="tight", pad_inches=0)
        plt.close()
    print(f"Saved {num} GAF images to {out_dir}")


if __name__ == "__main__":
    # Diffusion: real vs generated
    save_gaf_images_from_npy(
        "diffusion_real_sequences.npy",
        "gaf_diffusion_real",
        max_images=50,
        prefix="real_"
    )
    save_gaf_images_from_npy(
        "diffusion_generated_sequences.npy",
        "gaf_diffusion_generated",
        max_images=50,
        prefix="gen_"
    )

    # TimeGAN: real vs generated
    save_gaf_images_from_npy(
        "timegan_real_sequences.npy",
        "gaf_timegan_real",
        max_images=50,
        prefix="real_"
    )
    save_gaf_images_from_npy(
        "timegan_generated_sequences.npy",
        "gaf_timegan_generated",
        max_images=50,
        prefix="gen_"
    )
