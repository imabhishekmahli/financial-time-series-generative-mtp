import os
import numpy as np
from PIL import Image
from scipy.linalg import sqrtm

def load_images_as_features(folder, max_images=100):
    paths = sorted([os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if f.endswith(".png")])
    if len(paths) == 0:
        raise RuntimeError(f"No images found in {folder}")
    paths = paths[:max_images]
    feats = []
    for p in paths:
        img = Image.open(p).convert("L")
        arr = np.array(img, dtype=float) / 255.0
        feats.append(arr.flatten())  # pixel-space features
    feats = np.stack(feats, axis=0)  # [N, D]
    return feats

def compute_fid(feats_real, feats_gen):
    """
    feats_real, feats_gen: [N, D] feature arrays
    """
    mu1 = feats_real.mean(axis=0)
    mu2 = feats_gen.mean(axis=0)
    sigma1 = np.cov(feats_real, rowvar=False)
    sigma2 = np.cov(feats_gen, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    # numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return float(fid)

if __name__ == "__main__":
    # ----- Diffusion: RP images -----
    diff_real_feats = load_images_as_features("rp_diffusion_real", max_images=50)
    diff_gen_feats  = load_images_as_features("rp_diffusion_generated", max_images=50)
    fid_diff = compute_fid(diff_real_feats, diff_gen_feats)
    print(f"FID (RP, Diffusion real vs gen): {fid_diff:.4f}")

    # ----- TimeGAN: RP images -----
    time_real_feats = load_images_as_features("rp_timegan_real", max_images=50)
    time_gen_feats  = load_images_as_features("rp_timegan_generated", max_images=50)
    fid_time = compute_fid(time_real_feats, time_gen_feats)
    print(f"FID (RP, TimeGAN real vs gen): {fid_time:.4f}")
