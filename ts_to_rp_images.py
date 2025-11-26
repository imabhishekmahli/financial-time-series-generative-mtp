import os
import numpy as np
import matplotlib.pyplot as plt

# =============== CONFIG =================

# How many sequences from each source to convert to images
NUM_IMAGES_PER_SET = 20

# Output directories
OUT_DIRS = {
    "diffusion_real": "rp_diffusion_real",
    "diffusion_gen":  "rp_diffusion_generated",
    "timegan_real":   "rp_timegan_real",
    "timegan_gen":    "rp_timegan_generated",
}

# =============== UTILS ==================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def recurrence_plot(seq):
    """
    seq: 1D numpy array [T]
    Returns: 2D numpy array [T, T] as recurrence plot
    """
    seq = np.asarray(seq, dtype=float)
    # Normalize to zero mean, unit variance
    seq = (seq - seq.mean()) / (seq.std() + 1e-8)

    # Build distance matrix |x_i - x_j|
    x = seq.reshape(-1, 1)            # [T,1]
    diff = x - x.T                    # [T,T]
    R = np.abs(diff)                  # [T,T]

    # Normalize to [0,1]
    R = R / (R.max() + 1e-8)

    # Invert so similar points are bright
    R_img = 1.0 - R
    return R_img

def save_rp_images_from_array(arr, out_dir, prefix):
    """
    arr: numpy array of sequences
        - shape [N, T] or [N, T, 1]
    out_dir: folder to save images
    prefix: filename prefix
    """
    ensure_dir(out_dir)

    if arr.ndim == 3:      # [N,T,1]
        arr = arr[:, :, 0] # → [N,T]

    N, T = arr.shape
    print(f"Saving RP images for {prefix} from array shape: {arr.shape}")

    num = min(NUM_IMAGES_PER_SET, N)

    for i in range(num):
        seq = arr[i]                  # [T]
        R_img = recurrence_plot(seq)  # [T,T]

        plt.figure(figsize=(3,3))
        plt.imshow(R_img, cmap="gray", origin="lower")
        plt.axis("off")
        fname = os.path.join(out_dir, f"{prefix}_{i:03d}.png")
        plt.savefig(fname, bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close()

    print(f"Saved {num} RP images to {out_dir}")

# =============== MAIN ==================

if __name__ == "__main__":
    print("Starting time-series -> Recurrence Plot image conversion...")

    sets = [
        ("diffusion_real_sequences.npy",     "diffusion_real"),
        ("diffusion_generated_sequences.npy","diffusion_gen"),
        ("timegan_real_sequences.npy",       "timegan_real"),
        ("timegan_generated_sequences.npy",  "timegan_gen"),
    ]

    for npy_file, key in sets:
        if os.path.exists(npy_file):
            print(f"Found {npy_file}, loading...")
            arr = np.load(npy_file)
            out_dir = OUT_DIRS[key]
            prefix = key
            save_rp_images_from_array(arr, out_dir, prefix)
        else:
            print(f"{npy_file} not found, skipping.")

    print("Done. Check the rp_* folders for image outputs.")
