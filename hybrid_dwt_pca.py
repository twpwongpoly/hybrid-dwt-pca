import numpy as np
import pywt # For Discrete Wavelet Transform (DWT)
from scipy import linalg # For eigenvalue decomposition in PCA

def dwt_pca_fuse(images, wavelet='db4', level=3):
    """
    Fuse a list of preprocessed B-scans (images) using hybrid multi-level DWT-PCA.
 
    This function performs:
    1. Multi-level DWT decomposition on each input B-scan to separate multi-scale approximations and details.
    2. PCA on the coarsest approximation coefficients to compute weighted fusion.
    3. PCA on detail coefficients at each level (H, V, D subbands) for fusing.
    4. Inverse DWT (multi-level reconstruction) to obtain the fused B-scan.
 
    Parameters:
    - images: list of 2D numpy arrays (preprocessed B-scans) – all must have the same shape (time x positions).
      For your setup, this would include VV and HH at different frequencies (e.g., 4 images if 2 freqs each).
      Dimensions must be divisible by 2^level in both axes to avoid padding issues; otherwise, pad manually.
    - wavelet: str, the wavelet family to use for DWT (default 'db4' – Daubechies 4).
      Adjust this to change the decomposition filter. Common options: 'haar' (simple, fast), 'db2' (shorter support),
      'sym4' (symmetric), or 'bior4.4' (biorthogonal for better reconstruction). Experiment based on your data's frequency content.
      See pywt.wavelist() for all available wavelets.
    - level: int, the number of decomposition levels (default 3, e.g., set to 2 or 3 for multi-level).
      Higher levels capture coarser features and can improve fusion for data with multi-scale structures (e.g., GPR reflections at different depths).
      Start with level=1 for simplicity; increase if finer details need better separation. Note: Higher levels reduce approximation size,
      so ensure image dimensions support it (e.g., for level=3, min dim >= 8). If not, pywt will handle but may truncate.
 
    Returns:
    - fused_image: 2D numpy array, the fused B-scan with enhanced features from all inputs.
 
    Usage Tips:
    - Ensure all input images have identical dimensions; resize/pad if necessary (e.g., using np.pad for power-of-2 sizes).
    - For better fusion, experiment with wavelet types: 'db4' is balanced, but 'haar' is faster for testing.
    - To visualize intermediate results, add plots for the coarsest fused_approx or detail levels.
    - Computational cost scales with number of images, size, and levels; for large batches, consider downsampling.
    - If level=1, this behaves similarly to the previous single-level version.
    """
    # Step 1: Decompose each image using multi-level 2D DWT.
    # Returns a list: [cA_level, (cH_level, cV_level, cD_level), ..., (cH_1, cV_1, cD_1)]
    coeffs_list = [pywt.wavedec2(img, wavelet, level=level) for img in images]
 
    # Step 2: Fuse the coarsest approximation coefficients using PCA.
    approx_coeffs = [coeffs[0] for coeffs in coeffs_list] # List of cA_level (coarsest LL)
    ah, aw = approx_coeffs[0].shape
    K = len(approx_coeffs)
    approx_stack = np.array(approx_coeffs)  # (K, ah, aw)
    approx_data = approx_stack.reshape(K, ah * aw).T  # (ah*aw, K)
    mean_a = np.mean(approx_data, axis=0)
    centered_a = approx_data - mean_a
    cov_a = np.cov(centered_a.T)
    # Handle small covariance case
    if np.allclose(cov_a, 0):
        fused_approx = np.mean(approx_stack, axis=0)
    else:
        evals_a, evecs_a = linalg.eigh(cov_a)
        idx_a = np.argsort(evals_a)[::-1]
        v_a = evecs_a[:, idx_a[0]]
        # Normalize weights to sum to 1
        v_a = v_a / np.sum(v_a)
        # Ensure weights are positive (flip if necessary)
        if np.sum(v_a) < 0:
            v_a = -v_a
        fused_approx = np.dot(centered_a, v_a).reshape(ah, aw)
 
    fused_details = [] # Will hold fused (cH_k, cV_k, cD_k) for each level (coarsest to finest)
 
    for lev in range(level): # For each decomposition level
        # Horizontal details
        dets_h = np.array([c[lev + 1][0] for c in coeffs_list])  # (K, dh, dw)
        dh, dw = dets_h.shape[1:]
        h_data = dets_h.reshape(K, dh * dw).T  # (dh*dw, K)
        mean_h = np.mean(h_data, axis=0)
        centered_h = h_data - mean_h
        cov_h = np.cov(centered_h.T)
        # Handle small covariance case
        if np.allclose(cov_h, 0):
            fused_h = np.mean(dets_h, axis=0)
        else:
            evals_h, evecs_h = linalg.eigh(cov_h)
            idx_h = np.argsort(evals_h)[::-1]
            v_h = evecs_h[:, idx_h[0]]
            # Normalize weights to sum to 1
            v_h = v_h / np.sum(v_h)
            # Ensure weights are positive (flip if necessary)
            if np.sum(v_h) < 0:
                v_h = -v_h
            fused_h = np.dot(centered_h, v_h).reshape(dh, dw)
        
        # Vertical details
        dets_v = np.array([c[lev + 1][1] for c in coeffs_list])
        v_data = dets_v.reshape(K, dh * dw).T
        mean_v = np.mean(v_data, axis=0)
        centered_v = v_data - mean_v
        cov_v = np.cov(centered_v.T)
        # Handle small covariance case
        if np.allclose(cov_v, 0):
            fused_v = np.mean(dets_v, axis=0)
        else:
            evals_v, evecs_v = linalg.eigh(cov_v)
            idx_v = np.argsort(evals_v)[::-1]
            v_v = evecs_v[:, idx_v[0]]
            # Normalize weights to sum to 1
            v_v = v_v / np.sum(v_v)
            # Ensure weights are positive (flip if necessary)
            if np.sum(v_v) < 0:
                v_v = -v_v
            fused_v = np.dot(centered_v, v_v).reshape(dh, dw)
        
        # Diagonal details
        dets_d = np.array([c[lev + 1][2] for c in coeffs_list])
        d_data = dets_d.reshape(K, dh * dw).T
        mean_d = np.mean(d_data, axis=0)
        centered_d = d_data - mean_d
        cov_d = np.cov(centered_d.T)
        # Handle small covariance case
        if np.allclose(cov_d, 0):
            fused_d = np.mean(dets_d, axis=0)
        else:
            evals_d, evecs_d = linalg.eigh(cov_d)
            idx_d = np.argsort(evals_d)[::-1]
            v_d = evecs_d[:, idx_d[0]]
            # Normalize weights to sum to 1
            v_d = v_d / np.sum(v_d)
            # Ensure weights are positive (flip if necessary)
            if np.sum(v_d) < 0:
                v_d = -v_d
            fused_d = np.dot(centered_d, v_d).reshape(dh, dw)
        
        fused_details.append((fused_h, fused_v, fused_d))
 
    # Step 4: Combine fused coefficients into the required list format for reconstruction.
    # Format: [fused_approx, fused_details_level (coarsest), ..., fused_details_1 (finest)]
    fused_coeffs = [fused_approx] + fused_details
 
    # Reconstruct the fused image using multi-level inverse DWT.
    fused_image = pywt.waverec2(fused_coeffs, wavelet)
 
    # Post-fusion normalization to unit std (matching input energy levels)
    std_fused = np.std(fused_image)
    if std_fused != 0:
        fused_image /= std_fused
  
    return fused_image