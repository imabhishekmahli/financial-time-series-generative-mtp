import numpy as np

# Simple DTW implementation (no external dependency)
def dtw_distance(a, b):
    """
    a, b: 1D numpy arrays of length T
    returns: scalar DTW distance
    """
    T1, T2 = len(a), len(b)
    dp = np.full((T1+1, T2+1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, T1+1):
        for j in range(1, T2+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i, j] = cost + min(dp[i-1, j],    # deletion
                                  dp[i, j-1],    # insertion
                                  dp[i-1, j-1])  # match
    return dp[T1, T2]

def avg_pairwise_dtw(real, gen, num_pairs=50):
    """
    Compute average DTW between random real and generated sequences.
    real, gen: [N,T] arrays
    """
    N_real = real.shape[0]
    N_gen  = gen.shape[0]
    num = min(num_pairs, N_real, N_gen)
    idx_real = np.random.choice(N_real, size=num, replace=False)
    idx_gen  = np.random.choice(N_gen,  size=num, replace=False)
    dists = []
    for i_r, i_g in zip(idx_real, idx_gen):
        d = dtw_distance(real[i_r], gen[i_g])
        dists.append(d)
    return float(np.mean(dists))

if __name__ == "__main__":
    np.random.seed(42)

    # ----- Diffusion -----
    d_real = np.load("diffusion_real_sequences.npy")
    d_gen  = np.load("diffusion_generated_sequences.npy")
    if d_real.ndim == 3: d_real = d_real[:, :, 0]
    if d_gen.ndim  == 3: d_gen  = d_gen[:, :, 0]

    dtw_diffusion = avg_pairwise_dtw(d_real, d_gen, num_pairs=30)
    print(f"Average DTW (Diffusion real vs gen): {dtw_diffusion:.4f}")

    # ----- TimeGAN -----
    t_real = np.load("timegan_real_sequences.npy")
    t_gen  = np.load("timegan_generated_sequences.npy")
    if t_real.ndim == 3: t_real = t_real[:, :, 0]
    if t_gen.ndim  == 3: t_gen  = t_gen[:, :, 0]

    dtw_timegan = avg_pairwise_dtw(t_real, t_gen, num_pairs=30)
    print(f"Average DTW (TimeGAN real vs gen): {dtw_timegan:.4f}")
