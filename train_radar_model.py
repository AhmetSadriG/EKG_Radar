import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.signal import butter, filtfilt

def bandpass_ecg(ecg, fs=200.0, low=0.75, high=4.0):
    # Radar kanalları 0.75-2.5 Hz bandpass'tan geçiyor; ECG'yi de aynı banda çek
    # QRS keskin pikleri radar'da yok — sadece kardiyak ritim öğretilebilir
    b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, ecg).astype(np.float32)

def apply_vmd_cardiac(X, fs=200.0, K=4, alpha=2000):
    """
    Her radar kanalina VMD uygular, kardiyak banda (0.75-4 Hz) en yakin
    modu alir. Nefes ve gurultu modlarini atar.
    X: [frames, channels]
    """
    try:
        from vmdpy import VMD
    except ImportError:
        return X

    result = np.zeros_like(X, dtype=np.float32)
    target  = 1.5 / fs          # ~1.5 Hz normalize hedef
    c_min   = 0.75 / fs
    c_max   = 4.0  / fs

    for ch in range(X.shape[1]):
        sig = X[:, ch].astype(np.float64)
        try:
            u, _, omega = VMD(sig, alpha, 0.1, K, 0, 1, 1e-7)
            freqs = omega[-1]   # son iterasyon merkez frekanslari (normalize)
            cardiac = [i for i, f in enumerate(freqs) if c_min <= f <= c_max]
            best = min(cardiac, key=lambda i: abs(freqs[i] - target)) if cardiac \
                   else int(np.argmin(np.abs(freqs - target)))
            result[:, ch] = u[best].astype(np.float32)
        except Exception:
            result[:, ch] = sig.astype(np.float32)

    return result

def pearson_loss(y_pred, y_true):
    vx = y_pred - y_pred.mean(dim=-1, keepdim=True)
    vy = y_true - y_true.mean(dim=-1, keepdim=True)
    corr = (vx * vy).sum(dim=-1) / (
        torch.sqrt((vx ** 2).sum(dim=-1) * (vy ** 2).sum(dim=-1)) + 1e-8
    )
    return (1 - corr.abs()).mean()

# ==========================================
# 1. Dataset Class (Per-Session Normalization + Per-Session Split)
# ==========================================
class RadarDataset(Dataset):
    def __init__(self, npz_files, window_size=640, stride=50, augment=False, use_vmd=False, label=''):
        if isinstance(npz_files, str):
            npz_files = [npz_files]

        X_windows, y_windows = [], []

        for path in npz_files:
            if not os.path.exists(path):
                print(f"Atlandy (bulunamady): {path}")
                continue
            d = np.load(path, allow_pickle=True)
            X = d['X'].astype(np.float32)
            y = bandpass_ecg(d['y'].astype(np.float32))

            if use_vmd:
                print(f"  VMD uygulanyor: {os.path.basename(path)} ...")
                X = apply_vmd_cardiac(X)

            # Per-session Z-score
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
            y = (y - y.mean()) / (y.std() + 1e-6)

            # Tum pencereleri kullan (split yok — session-level split main()'de yapilir)
            session_len = len(X)
            for start in range(300, session_len - window_size, stride):
                X_windows.append(X[start:start + window_size].T)
                y_windows.append(y[start:start + window_size][np.newaxis, :])

        self.X_windows = X_windows
        self.y_windows = y_windows
        self.window_size = window_size
        self.augment = augment
        print(f"Toplam pencere ({label}): {len(self.X_windows)}")

    def __len__(self):
        return len(self.X_windows)

    def __getitem__(self, idx):
        X = self.X_windows[idx].copy()
        y = self.y_windows[idx].copy()
        if self.augment:
            # Gaussian gurultu ekle
            if np.random.rand() < 0.5:
                X += np.random.normal(0, 0.05, X.shape).astype(np.float32)
            # Rastgele genlik olcekleme
            if np.random.rand() < 0.5:
                X *= np.random.uniform(0.85, 1.15)
            # Kanal dropout — dusuk korelasyonlu kanallari rastgele sifirla
            if np.random.rand() < 0.4:
                n_drop = np.random.randint(1, 5)
                drop_ch = np.random.choice(X.shape[0], size=n_drop, replace=False)
                X[drop_ch] = 0.0
        return (torch.from_numpy(X), torch.from_numpy(y))

# ==========================================
# 2. Model Architecture (U-Net with Channel Attention)
# ==========================================
class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation: hangi kanalların önemli olduğunu öğrenir."""
    def __init__(self, num_channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(num_channels, max(num_channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(num_channels // reduction, 4), num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):          # x: [B, C, T]
        w = x.mean(dim=-1)         # [B, C] — global average pool
        w = self.se(w)             # [B, C] — kanal agirliklari
        return x * w.unsqueeze(-1) # [B, C, T]


class RadarECGModel(nn.Module):
    def __init__(self, num_channels=5, dropout=0.2):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout),
        )
        self.pool1 = nn.MaxPool1d(2)  # 640 -> 320

        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
        )
        self.pool2 = nn.MaxPool1d(2)  # 320 -> 160

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
        )

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)  # 160 -> 320
        self.dec1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),  # 128 = 64 (up) + 64 (skip)
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
        )

        self.up2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)  # 320 -> 640
        self.dec2 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),  # 64 = 32 (up) + 32 (skip)
            nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout),
        )

        self.out = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        # self.ch_att devre disi — overfitting artiriyor, veri artinca tekrar denenebilir
        e1 = self.enc1(x)                        # [B, 32, 640]
        e2 = self.enc2(self.pool1(e1))            # [B, 64, 320]
        b  = self.bottleneck(self.pool2(e2))      # [B, 128, 160]

        d1 = self.up1(b)                          # [B, 64, 320]
        d1 = self.dec1(torch.cat([d1, e2], dim=1))  # [B, 64, 320]

        d2 = self.up2(d1)                         # [B, 32, 640]
        d2 = self.dec2(torch.cat([d2, e1], dim=1))  # [B, 32, 640]

        return self.out(d2)                       # [B, 1, 640]

# ==========================================
# 3. Training Loop
# ==========================================
def train_model(model, train_loader, val_loader, epochs=50, lr=2e-4, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 25
    history = {'train_loss': [], 'val_loss': []}

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X_batch, y_batch in loop:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = 0.2 * F.mse_loss(y_pred, y_batch) + 0.8 * pearson_loss(y_pred, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += (0.2 * F.mse_loss(y_pred, y_batch) + 0.8 * pearson_loss(y_pred, y_batch)).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_radar_model.pth')
            print("  Model kaydedildi (en iyi).")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping: {early_stop_patience} epoch iyilesme olmadi.")
                break

    torch.save(model.state_dict(), 'final_radar_model.pth')
    print("Final model saved to final_radar_model.pth")

    return history

def visualize_prediction(model, dataset, device='cpu'):
    idx = np.random.randint(0, len(dataset))
    X, y = dataset[idx]

    model.eval()
    with torch.no_grad():
        X_in = X.unsqueeze(0).to(device)
        y_pred = model(X_in).cpu().squeeze().numpy()

    y_true = y.squeeze().numpy()

    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='Ground Truth (ECG)', alpha=0.7)
    plt.plot(y_pred, label='Predicted (Radar)', alpha=0.7)
    plt.title(f"Prediction Example (Window {idx})")
    plt.legend()
    plt.savefig("prediction_example.png")
    print("Saved prediction_example.png")

def main():
    BASE = os.path.dirname(__file__)

    ALL_FILES = [
        # dataset_radar_ecg_6min.npz cikarildi — eski 5-kanal format, yeni 26-kanal ile uyumsuz
        # s28/s29/s30: yanlis ayar (32 chirp, tek TX) — KULLANMA
        # s31-s34: kayit sorunlari — KULLANMA
        os.path.join(BASE, "dataset_s10.npz"),
        os.path.join(BASE, "dataset_s11.npz"),
        os.path.join(BASE, "dataset_s12.npz"),
        os.path.join(BASE, "dataset_s14.npz"),
        os.path.join(BASE, "dataset_s17.npz"),
        os.path.join(BASE, "dataset_s18.npz"),
        os.path.join(BASE, "dataset_s19.npz"),
        os.path.join(BASE, "dataset_s20.npz"),
        os.path.join(BASE, "dataset_s21.npz"),
        os.path.join(BASE, "dataset_s22.npz"),
        os.path.join(BASE, "dataset_s24.npz"),
        os.path.join(BASE, "dataset_s25.npz"),
        os.path.join(BASE, "dataset_s26.npz"),
        # Yeni kayitlar — s10-s26 ile ayni range (0.15-0.40m), tutarli
        os.path.join(BASE, "dataset_s35.npz"),
        os.path.join(BASE, "dataset_s36.npz"),
        os.path.join(BASE, "dataset_s37.npz"),
        os.path.join(BASE, "dataset_s38.npz"),
        os.path.join(BASE, "dataset_s39.npz"),
        os.path.join(BASE, "dataset_s40.npz"),
        os.path.join(BASE, "dataset_s41.npz"),
        os.path.join(BASE, "dataset_s42.npz"),
        # Dogrulanmis yeni kayitlar (sign-corrected windowed corr >= 0.12, +% >= 65%)
        # s44: avg=0.208, +88%  |  s45: avg=0.329, +100%
        os.path.join(BASE, "dataset_s44.npz"),
        os.path.join(BASE, "dataset_s45.npz"),
        # s51: avg=0.125, +67%  |  s54: avg=0.147, +67%
        os.path.join(BASE, "dataset_s51.npz"),
        os.path.join(BASE, "dataset_s54.npz"),
        # s59: avg=0.312, +100%  |  s61: avg=0.160, +83%
        os.path.join(BASE, "dataset_s59.npz"),
        os.path.join(BASE, "dataset_s61.npz"),
        # Atlananlar: s27,s43,s46-49,s52-53,s55-58,s60 (50/50 veya negatif dominant)
    ]

    BATCH_SIZE = 32
    EPOCHS = 80
    LR = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {DEVICE}")

    # Session-level split: temporal split yerine session bazli ayir
    # Boylece val seti tamamen farkli sessionlardan olusur (sign-flip sorunu ortadan kalkar)
    import random
    random.seed(42)
    shuffled = ALL_FILES.copy()
    random.shuffle(shuffled)
    n_val = max(2, len(shuffled) // 5)          # ~%20 session val'e ayir
    val_files   = shuffled[:n_val]
    train_files = shuffled[n_val:]
    print(f"Train session: {len(train_files)}  |  Val session: {len(val_files)}")
    print(f"Val: {[os.path.basename(f) for f in val_files]}")

    train_dataset = RadarDataset(train_files, stride=50,  augment=True,  use_vmd=False, label='train')
    val_dataset   = RadarDataset(val_files,   stride=320, augment=False, use_vmd=False, label='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = RadarECGModel(num_channels=26, dropout=0.3).to(DEVICE)
    print(model)

    history = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE)

    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_loss.png')

    visualize_prediction(model, val_dataset, device=DEVICE)

if __name__ == "__main__":
    main()
