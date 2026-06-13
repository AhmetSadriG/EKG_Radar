import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

def main():
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "dataset_s1.npz"
    try:
        data = np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        print("Dataset not found.")
        return

    X = data['X'] # [Time, Channels=5]
    y = data['y'] # [Time]
    timestamp_objs = data['timestamps']
    
    # Calculate duration
    fs = 200 # Assumed
    duration = len(y) / fs
    print(f"Dataset Duration: {duration:.2f} seconds ({len(y)} samples)")
    
    # Check for NaN/Inf
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("WARNING: NaNs or Infs found in X!")
        X = np.nan_to_num(X)
        
    # --- Correlation Analysis ---
    # We correlate the Envelope of Radar (sum of absolute changes) vs ECG
    # Because radar measures mechanical motion (velocity/displacement), ECG measures electrical.
    # They should be related but maybe not linearly identical.
    
    # Use first channel or mean of channels
    radar_signal = X[:, 0] 
    # Or PCA? Let's just use Mean of absolute values to see "activity"
    radar_activity = np.mean(np.abs(X), axis=1)
    
    # Normalize for correlation
    radar_norm = (radar_activity - np.mean(radar_activity)) / (np.std(radar_activity) + 1e-6)
    ecg_norm = (y - np.mean(y)) / (np.std(y) + 1e-6)
    
    # Calculate Correlation
    print("Calculating cross-correlation...")
    # Use a chunk to save time/memory if huge, but 50k is small enough
    corr = correlate(ecg_norm, radar_norm, mode='full')
    lags = correlation_lags(len(ecg_norm), len(radar_norm), mode='full')
    
    # Find max
    max_idx = np.argmax(np.abs(corr))
    max_lag = lags[max_idx]
    max_corr = corr[max_idx] / len(ecg_norm) # Normalized correlation coefficient
    
    lag_seconds = max_lag / fs
    
    print(f"Max Correlation: {max_corr:.4f} at Lag: {max_lag} samples ({lag_seconds:.3f} seconds)")
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    # Plot a 10-second segment
    segment = 2000 # 10s
    plt.plot(ecg_norm[:segment], label='ECG (Norm)')
    plt.plot(radar_norm[:segment], label='Radar Activity (Norm)', alpha=0.7)
    plt.title("Sample Segment (First 10s)")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(lags / fs, corr / len(ecg_norm))
    plt.title("Cross-Correlation vs Lag (Seconds)")
    plt.xlabel("Lag (Seconds) - Negative means Radar is AHEAD of ECG")
    plt.ylabel("Correlation Coeff")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    # Shifted comparison
    if np.abs(max_lag) < 1000: # If lag is reasonable (< 5s) show alignment
        shifted_radar = np.roll(radar_norm, max_lag)
        plt.plot(ecg_norm[:segment], label='ECG')
        plt.plot(shifted_radar[:segment], label=f'Radar Shifted ({lag_seconds:.2f}s)')
        plt.title("Aligned Segment (Based on Max Corr)")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Lag too large to visualize/meaningful?", ha='center')
        
    plt.tight_layout()
    plt.savefig("synchronization_check.png")
    print("Saved synchronization_check.png")

if __name__ == "__main__":
    main()
