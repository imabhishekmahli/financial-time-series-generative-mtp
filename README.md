# Financial Time-Series Generative Modeling (MTP)

This project compares **Diffusion Models (DDPM)** and **TimeGAN** for generating synthetic financial time-series data.

## Assets

- S&P 500 index (^GSPC)
- Google stock (GOOG)
- Corn futures (ZC=F)
- Plus small experiments on AAPL and BTC (in other folders)

## Pipeline

1. **Data prep** – `data_prep_multi.py`  
   - Download prices with `yfinance`
   - Convert to log-returns
   - Create 60-day sliding windows
2. **Diffusion model (DDPM)** – `ddpm_single_asset.py`  
   - Trains a 1D DDPM per asset
   - Generates synthetic return sequences
3. **TimeGAN** – `timegan_single_asset.py`  
   - Trains TimeGAN per asset
   - Generates synthetic sequences with temporal dependencies
4. **Evaluation**
   - RP / GAF image generation
   - CNN classifier (real vs fake)
   - DTW distances
   - Predictive Score (Train-on-Synthetic, Test-on-Real)

## Visualizations

Real vs diffusion overlay plots are stored in:
`bigDataset/real_vs_diffusion_images/`

