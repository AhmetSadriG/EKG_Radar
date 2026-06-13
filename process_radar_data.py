import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1" # Fix KMeans memory leak warning
import glob
from scipy.signal import butter, filtfilt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ==========================================
# Configuration Class
# ==========================================
class RadarConfig:
    def __init__(self):
        # Hardware Parameters (IWR1843BOOST + DCA1000)
        self.start_freq = 77e9
        self.slope = 65e12 # 65 MHz/us
        self.idle_time = 10e-6
        self.ramp_end_time = 60e-6
        self.adc_samples = 256
        self.sample_rate = 5e6
        
        # Frame Config
        self.num_tx = 3
        self.num_rx = 4
        self.num_chirps_per_loop = 3 # TX0, TX1, TX2 TDM
        self.num_loops = 23
        self.chirps_per_frame = self.num_loops * self.num_tx # 69
        self.frame_periodicity = 5e-3 # 5ms
        
        # Constants
        self.c = 3e8
        self.lambda_val = self.c / self.start_freq
        
        # Antenna Layout (IWR1843BOOST)
        # Virtual Array has 3*4 = 12 elements.
        # TX0 (Azimuth), TX1 (Azimuth), TX2 (Elevation)
        # Simplified virtual linear array assumed for Azimuth focusing first
        self.num_virtual_antennas = self.num_tx * self.num_rx

        # Range Gating
        self.min_range_m = 0.15
        self.max_range_m = 0.40

# ==========================================
# 1. Data Ingestion
# ==========================================
def read_radar_bin(file_path):
    """
    Reads DCA1000 binary file and converts to complex format.
    Data format: 16-bit signed, interleaved [I, Q, I, Q...]
    """
    try:
        raw_data = np.fromfile(file_path, dtype=np.int16)
        raw_data = raw_data.reshape(-1, 2)
        complex_data = raw_data[:, 0] + 1j * raw_data[:, 1]
        return complex_data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def reshape_to_datacube(data, config):
    """
    Reshapes 1D array to [Frames, Chirps, RX, Samples]
    """
    samples_per_frame = config.chirps_per_frame * config.num_rx * config.adc_samples
    num_frames = len(data) // samples_per_frame
    
    if num_frames == 0:
        return None
        
    data = data[:num_frames * samples_per_frame]
    
    # Check data organization logic:
    # DCA1000 Non-interleaved: [Samples, RX, Chirps] typically if reformatted,
    # but raw output is often just stream of samples.
    # Standard IWR1843 capture ordering:
    # Frame -> Chirp -> Antenna -> Sample (or Sample -> Antenna depending on lane config)
    # We will assume: Frame -> Chirp -> RX -> Sample for this implementation drift 
    
    # 1. Reshape to Frames
    data = data.reshape(num_frames, -1)
    
    # 2. Inside Frame: Chirps * RX * Samples
    # Assuming standard non-interleaved output where we get all samples for a chirp/rx
    # Typical: [Chirps, RX, Samples]
    try:
        datacube = data.reshape(num_frames, config.chirps_per_frame, config.num_rx, config.adc_samples)
        return datacube
    except:
        return None

# ==========================================
# 2. Algorithm I: 3D Beamforming
# ==========================================
def perform_3d_beamforming(datacube, config):
    """
    Implements Eq. 2: Coherent combination to scan 3D space.
    
    Since strictly iterating (x,y,z) is computationally heavy, we use the 
    equivalent FFT-based Angle estimation to separate reflections into voxels.
    
    Steps:
    1. Range FFT
    2. Virtual Array Construction
    3. Angle FFT (Azimuth/Elevation)
    
    Returns:
        beamformed_data: [Frames, VirtualAntennas, RangeBins] -> [Frames, Space(Angle), Range]
        Actually, we want a time-series for each "voxel".
        So we keep Time (Frames) and project spatial dimensions.
    """
    num_frames, num_chirps, num_rx, num_samples = datacube.shape
    
    # --- Range FFT ---
    # Windowing
    window = np.hamming(num_samples)
    range_profile = np.fft.fft(datacube * window, axis=3)
    
    # --- Virtual Array (MIMO) ---
    # Reshape to separate loops and TX: [Frames, Loops, TX, RX, Range]
    range_profile = range_profile.reshape(num_frames, config.num_loops, config.num_tx, num_rx, num_samples)
    
    # Flatten TX and RX to Virtual Array: [Frames, Loops, VirtualAntennas, Range]
    # For IWR1843: 
    # Virtual elements 0-7 are Azimuth (same elevation) from TX0, TX1
    # Virtual elements 8-11 are Elevation from TX2
    # We construct the 12-element virtual array.
    virtual_cube = range_profile.reshape(num_frames, config.num_loops, config.num_virtual_antennas, num_samples)
    
    # To get a time-series for each voxel, we process the "Loops" dimension?
    # No, typically "Frames" is the slow time (heartbeat scale). "Loops" is fast time (Doppler scale).
    # Since we want cardiac motion (low freq), we can extract phase from the slow-time axis (Frames).
    # We can average across Loops to improve SNR before Phase extraction (or Zero-Doppler bin).
    # Let's effectively Coherent Integration across loops.
    
    base_signal = np.mean(virtual_cube, axis=1) # Shape: [Frames, 12, Range]
    
    # --- Angle FFT (Beamforming) ---
    # We perform FFT across the virtual antenna dimension to spatially separate signals.
    # Note: Proper calibration is needed for real IWR1843 data to align phase centers.
    # We assume ideal array here for algorithm demonstration.
    # Focusing mainly on Azimuth (first 8 elements).
    
    azimuth_data = base_signal[:, :8, :]
    # Angle FFT
    angle_bins = 64
    spatial_spectrum = np.fft.fftshift(np.fft.fft(azimuth_data, n=angle_bins, axis=1), axes=1)
    
    # Result: [Frames, Angle, Range] -> effectively [Time, Space]
    # Each (Angle, Range) pair is a "Voxel" candidate.
    return spatial_spectrum

# ==========================================
# 3. Algorithm II: Micro-Motions Amplification
# ==========================================
def extract_phase_and_amplify(voxel_signal, config):
    """
    Extracts phase (displacement) and applies Eq. 3 to amplify micro-motions.
    
    voxel_signal: Complex time series [Frames] for a specific voxel.
    """
    # 1. Phase Extraction
    phase = np.angle(voxel_signal)
    
    # Phase unwrapping is critical for tracking motion > lambda/2
    unwrapped_phase = np.unwrap(phase)
    
    # 2. Micro-Motion Amplification (Eq. 3)
    # s''_0 = [(s_{-3} + s_3) + 2(s_{-2} + s_2) - (s_{-1} + s_1) - 4s_0] / 16h^2
    # This is a specialized second-derivative filter (acceleration).
    
    h = config.frame_periodicity
    s = unwrapped_phase
    
    n = len(s)
    amplified_signal = np.zeros(n)
    
    # Implement the finite difference kernel
    # Kernel: [1, 2, -1, -4, -1, 2, 1] / 16h^2 centered at 0?
    # Formula in article: (s_3 + s_{-3}) + ...
    # This looks like a symmetric filter.
    
    # Indices relative to 0: -3, -2, -1, 0, 1, 2, 3
    # Coeffs: 1, 2, -1, -4, -1, 2, 1
    
    for i in range(3, n-3):
        term1 = s[i-3] + s[i+3]
        term2 = 2 * (s[i-2] + s[i+2])
        term3 = -1 * (s[i-1] + s[i+1])
        term4 = -4 * s[i]
        
        amplified_signal[i] = (term1 + term2 + term3 + term4) / (16 * h**2)
        
    return amplified_signal

# ==========================================
# 4. Algorithm III: Cardiac Signals Focusing
# ==========================================
def dynamic_time_warping(s1, s2):
    """
    Simple DTW implementation for pattern matching.
    """
    n, m = len(s1), len(s2)
    dtw_matrix = np.inf * np.ones((n+1, m+1))
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s1[i-1] - s2[j-1])
            # Take minimum of neighbors
            last_min = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            dtw_matrix[i, j] = cost + last_min
            
    return dtw_matrix[n, m]

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Applies Butterworth Bandpass Filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

def cardiac_focusing(amplified_signals, config):
    """
    Identifies cardiac-related voxels using Periodicity Pattern Matching.
    
    amplified_signals: [NumVoxels, TimePoints]
    """
    num_voxels, time_points = amplified_signals.shape
    scores = np.zeros(num_voxels)
    
    # --- Template-Based Periodicity Search (Article Method) ---
    # Instead of autocorrelation (which picks harmonic noise), we search for a period T
    # that minimizes the variance between segments.
    
    # 1. Define Heart Rate Range to search
    min_bpm = 50
    max_bpm = 130
    fs = 1.0 / config.frame_periodicity
    
    # Convert to period in samples
    # T = Fs * 60 / BPM
    min_period = int(fs * 60 / max_bpm)
    max_period = int(fs * 60 / min_bpm)
    
    candidates = np.arange(min_period, max_period + 1)
    
    for i in range(num_voxels):
        sig = amplified_signals[i]
        
        # Skip low energy or flat signals
        if np.std(sig) < 1e-6:
            scores[i] = 0
            continue
            
        best_score = -np.inf
        
        # Optimization Loop: Find best T
        # We can implement a simplified version: Fold signal and check variance
        
        # For efficiency in Python loop over large voxels, we might want to vectorise.
        # But for strictly following logic:
        
        for T in candidates:
            # Number of full segments
            num_segments = len(sig) // T
            if num_segments < 2: 
                continue
                
            # Extract segments
            segments = []
            for k in range(num_segments):
                segments.append(sig[k*T : (k+1)*T])
            
            segments = np.array(segments)
            
            # Calculate Variance across segments at each time index
            # Var(t) = Average euclidean distance from Mean Template
            mean_template = np.mean(segments, axis=0)
            
            # Variance metric: Sum of squared differences from template
            # Normalized by signal energy
            variance = np.sum((segments - mean_template)**2) / (num_segments * T)
            energy = np.var(sig)
            
            if energy == 0:
                continue
                
            # Score: Inverse of Normalized Variance
            # Lower variance = Higher periodicity
            score = energy / (variance + 1e-9)
            
            if score > best_score:
                best_score = score
                
        scores[i] = best_score
            
    return scores


# ==========================================
# 5. Algorithm IV: Spatial Filtering
# ==========================================
def spatial_filtering(spatial_data, scores, features_grid):
    """
    K-Means clustering to merge nearby signals (Eq. 12).
    
    spatial_data: [NumVoxels, Time] (amplified signals)
    scores: [NumVoxels] (periodicity scores)
    features_grid: [NumVoxels, 3] (r, theta, phi or similar position coords)
    """
    N_CLUSTERS = 26  # Mevcut model girisiyle uyumlu sabit kanal sayisi

    # En iyi skorlu top-N voxel'i sec (esik yerine sabit sayi)
    top_n = min(N_CLUSTERS * 4, len(scores))  # K-means icin en az 4x voxel
    top_indices = np.argsort(scores)[-top_n:]
    valid_indices = top_indices

    if len(valid_indices) == 0:
        return None

    selected_signals = spatial_data[valid_indices]
    selected_pos = features_grid[valid_indices]

    X = selected_pos

    n_clusters = min(N_CLUSTERS, len(valid_indices))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    
    # 3. Merging (Power Weighted)
    final_signals = []
    for k in range(n_clusters):
        cluster_mask = (kmeans.labels_ == k)
        cluster_sigs = selected_signals[cluster_mask]
        
        # Simple mean for implementation (article uses power weighted)
        merged_sig = np.mean(cluster_sigs, axis=0)
        final_signals.append(merged_sig)
        
    return np.array(final_signals)

# ==========================================
# Main Processing Pipeline
# ==========================================
def process_file(file_path):
    print(f"Processing {os.path.basename(file_path)}...")
    config = RadarConfig()
    
    # 1. Read
    raw_complex = read_radar_bin(file_path)
    if raw_complex is None: return
    
    datacube = reshape_to_datacube(raw_complex, config)
    if datacube is None: 
        print("Data reshape failed.")
        return
    print(f"  Datacube Shape: {datacube.shape}")
    
    # 2. 3D Beamforming (Range-Angle Map extraction)
    # Output: [Frames, Angle, Range]
    spatial_time_series = perform_3d_beamforming(datacube, config)
    print(f"  Spatial Series Shape: {spatial_time_series.shape}")
    
    # Flatten Spatial Dimensions to Voxels for processing
    # Voxel = (Angle, Range) pair
    num_frames, num_angles, num_ranges = spatial_time_series.shape
    num_voxels = num_angles * num_ranges
    
    # Reshape to [Voxels, Frames]
    voxel_signals_complex = spatial_time_series.reshape(num_frames, num_voxels).T
    
    # Create Feature Grid (Position of each voxel) for Spatial Filtering
    # (Angle Index, Range Index)
    angle_idx, range_idx = np.meshgrid(np.arange(num_angles), np.arange(num_ranges), indexing='ij')
    voxel_positions = np.stack([angle_idx.flatten(), range_idx.flatten()], axis=1)
    
    # 3. Micro-Motion Amplification
    print("  Amplifying micro-motions...")
    amplified_signals = []
    # Optimization: Only process relevant range bins? 
    # For speed, process all, but could slice magnitude mask.
    
    # --- Range Gating & Energy Thresholding ---
    
    # Calculate Range and Angle for each Voxel
    # Range Resolution = c / (2 * Bandwidth)
    bandwidth = config.slope * config.ramp_end_time 
    range_res = config.c / (2 * bandwidth)
    
    # Map Voxel Index -> Range (m)
    # voxel_positions has [AngleIdx, RangeIdx]
    voxel_ranges_m = voxel_positions[:, 1] * range_res
    
    # 1. Range Mask: Keep only voxels within [min_range, max_range]
    range_mask = (voxel_ranges_m >= config.min_range_m) & (voxel_ranges_m <= config.max_range_m)
    
    # 2. Energy Mask: Keep top 50% energetic voxels WITHIN that range
    energies = np.sum(np.abs(voxel_signals_complex), axis=1)
    
    # Combine masks: Range AND Energy
    # We first apply range mask, then threshold energy only on those
    valid_range_indices = np.where(range_mask)[0]
    
    if len(valid_range_indices) == 0:
        print("No voxels found in target range!")
        return None
        
    valid_energies = energies[valid_range_indices]
    energy_threshold = np.percentile(valid_energies, 50) # Keep top 50% of the relevant range
    
    # Final mask
    final_mask = range_mask & (energies >= energy_threshold)
    
    active_indices = np.where(final_mask)[0]
    processed_voxels = voxel_signals_complex[active_indices]
    active_positions = voxel_positions[active_indices]
    
    for sig in processed_voxels:
        amp_sig = extract_phase_and_amplify(sig, config)
        amplified_signals.append(amp_sig)
    
    amplified_signals = np.array(amplified_signals)
    
    # --- Bandpass Filter (Explicit) ---
    print("  Applying Bandpass Filter (0.75 - 2.5 Hz)...")
    fs_radar = 1.0 / config.frame_periodicity # 200 Hz
    filtered_signals = []
    for sig in amplified_signals:
        if len(sig) > 100:
            filt = bandpass_filter(sig, 0.75, 2.5, fs_radar)
            filtered_signals.append(filt)
        else:
             filtered_signals.append(sig)
    amplified_signals = np.array(filtered_signals)

    # 4. Cardiac Focusing
    # Bandpass filter is now applied EXPLICITLY before cardiac_focusing
    print("  Focusing cardiac signals...")
    scores = cardiac_focusing(amplified_signals, config)
    
    # 5. Spatial Filtering
    print("  Spatial filtering...")
    # This returns the candidates for Heartbeat signal
    final_cardiac_motions = spatial_filtering(amplified_signals, scores, active_positions)
    
    # --- Visualization ---
    output_dir = "results_article_implementation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Range-Angle Map (Energy)
    avg_map = np.mean(np.abs(spatial_time_series), axis=0).T
    plt.figure(figsize=(10, 5))
    plt.imshow(20*np.log10(avg_map + 1), aspect='auto', origin='lower', cmap='jet')
    plt.title(f"Range-Angle Map: {os.path.basename(file_path)}")
    plt.savefig(os.path.join(output_dir, f"{os.path.basename(file_path)}_heatmap.png"))
    plt.close()
    
    # Plot 2: Final Extracted Signals
    if final_cardiac_motions is not None:
        plt.figure(figsize=(12, 6))
        for i, sig in enumerate(final_cardiac_motions):
            plt.plot(sig, label=f"Cluster {i}")
        plt.title(f"Extracted Cardiac Motions (Amplified): {os.path.basename(file_path)}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{os.path.basename(file_path)}_signals.png"))
        plt.close()
        print(f"  Saved results to {output_dir}")
        return final_cardiac_motions
    else:
        print("  No valid cardiac signals found.")
        return None

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "Radar_verileri", "Beamforming_verisi")
    files = glob.glob(os.path.join(data_dir, "*.bin"))
    
    if not files:
        print("No files found.")
    else:
        for f in files:
            process_file(f)
