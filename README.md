# A Self-Guided Hybrid Diffusion GAN Framework for Financial Time Series Modeling with Forecast Refinement

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)

This repository contains the official implementation of the research project **"A Self-Guided Hybrid Diffusion GAN Framework for Financial Time Series Modeling with Forecast Refinement"**, submitted in partial fulfillment of the requirements for the degree of **Master of Technology** in Computer Science and Engineering at the **Indian Institute of Technology Tirupati**.

* **Author:** [Abhishek Kumar](https://github.com/yourusername) (Roll No: CS24M120)
* **Supervisor:** [Dr. Chalavadi Vishnu](https://iittp.ac.in/) (Assistant Professor, Dept. of CSE, IIT Tirupati)

---

## 📌 Overview

Financial time series modeling is highly challenging due to non-stationarity, volatility clustering, and heavy-tailed distributions. Traditional statistical models and deep learning sequences often fail to capture both short-term temporal coherence and long-term structural dependencies simultaneously. 

This framework introduces a **novel hybrid generative architecture** that combines the complementary strengths of Diffusion Models and Generative Adversarial Networks (GANs):
* **Diffusion Models:** Capture the global data distribution and long-term temporal dynamics stable.
* **GANs:** Enhance local temporal coherence and generate sharp, realistic patterns.
* **Self-Guided Mechanism:** Enables conditional generation during inference without task-specific retraining.
* **Forecast Refinement:** Leverages the learned diffusion prior to iteratively improve baseline forecasts.

---

## 🚀 Key Features

* **Unified Framework:** Integrates synthetic data generation, forecasting, and prediction refinement within a single architecture.
* **Self-Guided Inference:** Allows flexible conditioning and forecasting at inference time without modifying the training process.
* **Forecast Refinement Module:** Implements an energy minimization function to clean and sharpen raw predictions from baseline models like ARIMA or LSTM.
* **Stylized Fact Preservation:** Explicitly designed to preserve critical financial properties such as autocorrelation structures and volatility clustering.
* **Diverse Evaluation Suite:** Validated using robust forecasting metrics (RMSE, MAE), similarity metrics (DTW, TSTR), and financial statistical analysis.

---

### Mathematical Formulation Summary

The forward diffusion process corrupts data $x_0$ with Gaussian noise via a schedule $\beta_t$:
$$q(x_{t}|x_{t-1})=\mathcal{N}(x_{t};\sqrt{1-\beta_{t}}x_{t-1},\beta_{t}I)$$

The forecast refinement module optimizes an initial forecast $\tilde{y}$ by minimizing a diffusion-prior energy function regulated by parameter $\lambda$:
$$E(y)=-log\,p_{\theta}(y)+\lambda R(y,\tilde{y})$$

---

## 📊 Performance & Key Findings

### 1. Forecasting Performance Comparison
The proposed hybrid model achieves the lowest error rates across traditional and deep learning baselines:

| Model | RMSE ↓ | MAE ↓ |
| :--- | :---: | :---: |
| ARIMA | 0.085 | 0.062 |
| LSTM | 0.061 | 0.045 |
| GAN | 0.058 | 0.042 |
| Diffusion (DDPM) | 0.072 | 0.051 |
| TSDiff | 0.054 | 0.039 |
| **Hybrid (Proposed)** | **0.047** | **0.034** |

### 2. Effect of Forecast Refinement
Applying the diffusion refinement prior yields a consistent reduction in baseline model prediction error without requiring retraining:

| Baseline Model | Before Refinement (RMSE) | After Refinement (RMSE) |
| :--- | :---: | :---: |
| LSTM | 0.061 | **0.052** |
| ARIMA | 0.085 | **0.070** |

### 3. Synthetic Data Quality Evaluation
Evaluated using Dynamic Time Warping (DTW) and Train-on-Synthetic, Test-on-Real (TSTR) accuracy:

| Model | DTW ↓ | TSTR Accuracy ↑ |
| :--- | :---: | :---: |
| GAN | 0.42 | 0.78 |
| Diffusion | 0.35 | 0.72 |
| **Hybrid (Ours)** | **0.28** | **0.83** |

### 4. Ablation Study
Validating the performance contribution of individual model components:

| Model Variant | RMSE ↓ |
| :--- | :---: |
| GAN Only | 0.058 |
| Diffusion Only | 0.072 |
| Hybrid (No Refinement) | 0.051 |
| **Hybrid (Proposed full framework)** | **0.047** |

---
@mastersthesis{kumar2026selfguided,
  author       = {Abhishek Kumar},
  title        = {A Self-Guided Hybrid Diffusion GAN Framework for Financial Time Series Modeling with Forecast Refinement},
  school       = {Indian Institute of Technology Tirupati},
  year         = {2026},
  month        = {April},
  type         = {Master of Technology Thesis},
  address      = {Tirupati, Andhra Pradesh, India}
}


  
