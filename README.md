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

## 🛠️ System Architecture Pipeline
