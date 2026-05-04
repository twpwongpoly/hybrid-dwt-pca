# Hybrid DWT-PCA Python Module

A Python implementation of hybrid multi-level Discrete Wavelet Transform (DWT) and Principal Component Analysis (PCA) for data fusion of multi-channel images.

## Features
- Multi-level DWT decomposition (`db4` wavelet by default)
- PCA-based fusion on the coarsest approximation coefficients and all detail subbands (horizontal, vertical, diagonal)
- Returns a single fused image with enhanced features

## Installation
```bash
git clone https://github.com/twpwongpoly/hybrid-dwt-pca.git
cd hybrid-dwt-pca
pip install -r requirements.txt
