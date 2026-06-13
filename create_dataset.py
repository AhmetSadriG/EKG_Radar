"""
Radar .bin + Fabenode CSV  →  dataset_radar_ecg.npz

Kullanım:
    python create_dataset.py --radar kayit.bin --ecg fabenode_raw.csv --out yeni_dataset.npz

EKG kaydı: bluetooth_hearth.ipynb son hücresi → fabenode_raw.csv (500 Hz, int16)
Not: Radar ve EKG kaydı AYNI ANDA başlatılmalı (ya da --offset_s ile düzeltilir).
"""
import numpy as np
import csv
import argparse
import os
from scipy.signal import resample
from process_radar_data import (
    RadarConfig, read_radar_bin, reshape_to_datacube,
    perform_3d_beamforming, extract_phase_and_amplify,
    bandpass_filter, cardiac_focusing, spatial_filtering,
)


# ── EKG yükleme ────────────────────────────────────────────────
def load_ecg_csv(csv_path: str):
    """
    Fabenode CSV formatı: Timestamp (HH:MM:SS.ffffff), Raw_Value
    bluetooth_hearth.ipynb'nin fabenode_raw.csv çıkışını okur.
    Döndürür: (ecg_array, fs_hz)
    """
    from datetime import datetime

    timestamps, values = [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                if len(row) == 2:
                    # Fabenode: "HH:MM:SS.ffffff", int16_value
                    try:
                        t = datetime.strptime(row[0].strip(), "%H:%M:%S.%f")
                        ts_ms = (t.hour * 3600 + t.minute * 60 + t.second) * 1000 + t.microsecond / 1000
                        timestamps.append(ts_ms)
                    except ValueError:
                        timestamps.append(float(row[0]))
                    values.append(float(row[1]))
                elif len(row) == 1:
                    values.append(float(row[0]))
            except ValueError:
                continue  # başlık satırını atla

    ecg = np.array(values, dtype=np.float64)

    if timestamps:
        duration_s = (timestamps[-1] - timestamps[0]) / 1000.0
        fs = len(ecg) / duration_s
    else:
        fs = 200.0  # varsayılan
        print(f"  [UYARI] Timestamp yok, fs={fs:.0f} Hz varsayıldı")

    print(f"  EKG: {len(ecg)} örnek, {fs:.1f} Hz, {len(ecg)/fs:.1f} s")
    return ecg, fs


# ── Radar işleme: .bin (tek veya çok dosya) → 5 kanal sinyal ───
def process_radar_to_channels(bin_input, config: RadarConfig):
    """
    bin_input: tek dosya yolu VEYA glob pattern
    Örn: 'D:\\Veriler\\adc_data1_Raw_*.bin'
    """
    import glob as glob_mod, re

    if isinstance(bin_input, list):
        files = bin_input
    elif '*' in bin_input or '?' in bin_input:
        files = glob_mod.glob(bin_input)
        def _key(f):
            m = re.search(r'_(\d+)\.bin$', f)
            return int(m.group(1)) if m else 0
        files = sorted(files, key=_key)
    else:
        files = [bin_input]

    if not files:
        raise RuntimeError(f"Dosya bulunamadı: {bin_input}")

    fs_radar = 1.0 / config.frame_periodicity
    bw = config.slope * config.ramp_end_time
    range_res = config.c / (2 * bw)

    print(f"Radar işleniyor: {len(files)} dosya")

    # İlk dosyadan voxel grid ve aktif voxel maskesini belirle
    first_raw = read_radar_bin(files[0])
    if first_raw is None:
        raise RuntimeError(f"İlk dosya okunamadı: {files[0]}")
    first_dc = reshape_to_datacube(first_raw, config)
    if first_dc is None:
        raise RuntimeError("İlk dosya reshape edilemedi — num_loops ayarını kontrol et")
    first_spatial = perform_3d_beamforming(first_dc, config)
    num_angles = first_spatial.shape[1]
    num_ranges = first_spatial.shape[2]

    ai, ri = np.meshgrid(np.arange(num_angles), np.arange(num_ranges), indexing='ij')
    positions = np.stack([ai.flatten(), ri.flatten()], axis=1)

    ranges_m = positions[:, 1] * range_res
    range_mask = (ranges_m >= config.min_range_m) & (ranges_m <= config.max_range_m)
    valid_idx = np.where(range_mask)[0]
    if len(valid_idx) == 0:
        raise RuntimeError("Hedef mesafe aralığında voxel bulunamadı")

    first_flat = first_spatial.reshape(first_spatial.shape[0], -1)
    energies = np.sum(np.abs(first_flat), axis=0)
    thr = np.percentile(energies[valid_idx], 50)
    active_mask = range_mask & (energies >= thr)
    active_idx = np.where(active_mask)[0]
    active_positions = positions[active_idx]
    print(f"  Aktif voxel: {len(active_idx)}")

    # Tüm dosyalardan aktif voxel sinyallerini birleştir (bellek verimli)
    all_chunks = []
    for i, f in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {os.path.basename(f)}")
        raw = read_radar_bin(f)
        if raw is None:
            print(f"    ATLANDI: okunamadı")
            continue
        dc = reshape_to_datacube(raw, config)
        if dc is None:
            print(f"    ATLANDI: reshape başarısız")
            continue
        spatial = perform_3d_beamforming(dc, config)
        flat = spatial.reshape(spatial.shape[0], -1)
        all_chunks.append(flat[:, active_idx])  # sadece aktif voxeller

    if not all_chunks:
        raise RuntimeError("Hiçbir dosya işlenemedi")

    combined = np.concatenate(all_chunks, axis=0)  # (total_frames, n_active)
    voxel_complex = combined.T                      # (n_active, total_frames)
    total_frames = combined.shape[0]
    print(f"  Toplam frame: {total_frames} ({total_frames * config.frame_periodicity:.1f} sn)")

    # Faz çıkarımı + amplifikasyon
    amp_sigs = np.array([
        extract_phase_and_amplify(voxel_complex[i], config) for i in range(len(active_idx))
    ])

    # Bandpass
    filtered = []
    for s in amp_sigs:
        if len(s) > 100:
            filtered.append(bandpass_filter(s, 0.75, 2.5, fs_radar))
        else:
            filtered.append(s)
    amp_sigs = np.array(filtered)

    # Kalp odaklama + mekansal filtreleme → 5 küme
    scores = cardiac_focusing(amp_sigs, config)
    clusters = spatial_filtering(amp_sigs, scores, active_positions)

    if clusters is None:
        raise RuntimeError("Kalp sinyali bulunamadı — kayıt kalitesini kontrol edin")

    print(f"  Radar kanalları: {clusters.shape}  ({clusters.shape[1]} frame)")
    return clusters.T, fs_radar  # → (frames, n_clusters)


# ── Ana birleştirme ─────────────────────────────────────────────
def create_dataset(bin_path: str, csv_path: str, out_path: str, offset_s: float = 0.0):
    config = RadarConfig()

    # 1. Radar → X: (frames, 5)
    X_raw, fs_radar = process_radar_to_channels(bin_path, config)

    # 2. EKG → yeniden örnekle → Y: (frames,)
    ecg_raw, fs_ecg = load_ecg_csv(csv_path)
    target_len = X_raw.shape[0]
    ecg_resampled = resample(ecg_raw, int(len(ecg_raw) * fs_radar / fs_ecg))

    # 3. Offset uygula
    offset_samples = int(round(offset_s * fs_radar))
    if offset_samples > 0:
        X_raw = X_raw[offset_samples:]
    elif offset_samples < 0:
        ecg_resampled = ecg_resampled[-offset_samples:]

    # 4. Ortak uzunluğa kırp
    min_len = min(len(X_raw), len(ecg_resampled))
    X = X_raw[:min_len].astype(np.float64)
    y = ecg_resampled[:min_len].astype(np.float64)

    # 5. Zaman damgaları (saniye cinsinden)
    ts = np.arange(min_len) / fs_radar

    print(f"\nDataset:")
    print(f"  X shape    : {X.shape}")
    print(f"  y shape    : {y.shape}")
    print(f"  Süre       : {min_len/fs_radar:.1f} s")

    np.savez(out_path, X=X, y=y, timestamps=ts)
    print(f"  Kaydedildi : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar",    required=True, help="Radar .bin dosyası")
    parser.add_argument("--ecg",      required=True, help="EKG .csv dosyası")
    parser.add_argument("--out",      default="dataset_yeni.npz", help="Çıktı .npz dosyası")
    parser.add_argument("--offset_s", type=float, default=0.0,
                        help="EKG başlangıç offseti (sn). Pozitif: EKG geç başladı")
    args = parser.parse_args()

    create_dataset(args.radar, args.ecg, args.out, args.offset_s)
