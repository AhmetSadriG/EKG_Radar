"""
Per-session fine-tuning.
Base modeli (best_transformer_model.pth) yukle,
tek bir session uzerinde ince ayar yap, sonucu raporla.

Kullanim:
  python finetune_session.py            # s45 ve s59 uzerinde fine-tune
  python finetune_session.py --session 45  # sadece s45
"""

import os, sys, argparse, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.signal import butter, filtfilt, welch
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from train_transformer_model import CNNTransformerECG, RadarDataset, bandpass_ecg, pearson_loss

FS = 200.0


def load_session(npz_path, window_size=640, train_stride=50, val_stride=100,
                 split_ratio=0.8):
    """Tek session'i yukle, temporal 80/20 split uygula."""
    d = np.load(npz_path, allow_pickle=True)
    X = d['X'].astype(np.float32)
    y = bandpass_ecg(d['y'].astype(np.float32))

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    y = (y - y.mean()) / (y.std() + 1e-6)

    session_len = len(X)
    split_idx = int(split_ratio * session_len)

    X_train, y_train, X_val, y_val = [], [], [], []

    for start in range(300, split_idx - window_size, train_stride):
        X_train.append(X[start:start + window_size].T)
        y_train.append(y[start:start + window_size][np.newaxis, :])

    for start in range(split_idx, session_len - window_size, val_stride):
        X_val.append(X[start:start + window_size].T)
        y_val.append(y[start:start + window_size][np.newaxis, :])

    print(f"  Train: {len(X_train)} pencere  |  Val: {len(X_val)} pencere")
    return (RadarDataset(X_train, y_train, augment=True),
            RadarDataset(X_val,   y_val,   augment=False))


def finetune(session_num, epochs=30, lr=5e-5, batch_size=8):
    npz_path = os.path.join(BASE, f"dataset_s{session_num}.npz")
    if not os.path.exists(npz_path):
        print(f"HATA: {npz_path} bulunamadi!")
        return

    print(f"\n{'='*50}")
    print(f"Fine-tuning: S{session_num}  |  LR={lr}  |  Epochs={epochs}")
    print(f"{'='*50}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Base modeli yukle
    model = CNNTransformerECG(in_channels=26, d_model=128, nhead=4,
                              num_transformer_layers=3, dropout=0.1).to(device)
    base_path = os.path.join(BASE, 'best_transformer_model.pth')
    model.load_state_dict(torch.load(base_path, map_location=device))
    print(f"Base model yuklendi: best_transformer_model.pth")

    train_ds, val_ds = load_session(npz_path)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Base modelin bu session'daki performansi (fine-tune oncesi)
    base_corrs = evaluate(model, val_loader, device)
    print(f"Base model (fine-tune oncesi): ort={base_corrs.mean()*100:.1f}%  "
          f"medyan={np.median(base_corrs)*100:.1f}%  >85%={( base_corrs>0.85).mean()*100:.1f}%")

    # Fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_corrs = base_corrs.copy()
    history = {'train_loss': [], 'val_loss': [], 'val_corr': []}

    print(f"\n{'Ep':>4}  {'Train':>8}  {'Val':>8}  {'Corr%':>7}  {'>85%':>6}")
    print("-" * 40)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_b, y_b in tqdm(train_loader, desc=f"E{epoch+1}", leave=False):
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = 0.2 * F.mse_loss(pred, y_b) + 0.8 * pearson_loss(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        corrs = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                val_loss += (0.2 * F.mse_loss(pred, y_b)
                             + 0.8 * pearson_loss(pred, y_b)).item()
                for i in range(pred.shape[0]):
                    c = np.corrcoef(pred[i,0].cpu().numpy(), y_b[i,0].numpy())[0,1]
                    if not np.isnan(c):
                        corrs.append(c)
        val_loss /= len(val_loader)
        val_corr = float(np.mean(corrs)) if corrs else 0.0
        pct_85 = 100.0 * np.mean([c > 0.85 for c in corrs]) if corrs else 0.0
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_corr'].append(val_corr)

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_corrs = np.array(corrs)
            torch.save(model.state_dict(),
                       os.path.join(BASE, f'finetuned_s{session_num}.pth'))
            saved = " *"

        print(f"{epoch+1:4d}  {train_loss:8.4f}  {val_loss:8.4f}  "
              f"{val_corr*100:7.2f}  {pct_85:6.1f}%{saved}")

    # Sonuc raporu
    print(f"\n--- S{session_num} Fine-tune Sonucu ---")
    print(f"Base model   : ort={base_corrs.mean()*100:.1f}%  "
          f"medyan={np.median(base_corrs)*100:.1f}%  "
          f">85%={(base_corrs>0.85).mean()*100:.1f}%")
    print(f"Fine-tuned   : ort={best_corrs.mean()*100:.1f}%  "
          f"medyan={np.median(best_corrs)*100:.1f}%  "
          f">85%={(best_corrs>0.85).mean()*100:.1f}%")

    # Grafik
    plot_finetune(session_num, history, best_corrs, base_corrs, model, val_loader, device)
    return best_corrs


def evaluate(model, val_loader, device):
    model.eval()
    corrs = []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            pred = model(X_b)
            for i in range(pred.shape[0]):
                c = np.corrcoef(pred[i,0].cpu().numpy(), y_b[i,0].numpy())[0,1]
                if not np.isnan(c):
                    corrs.append(c)
    return np.array(corrs)


def plot_finetune(session_num, history, best_corrs, base_corrs, model, val_loader, device):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'S{session_num} Fine-tuning Sonucu', fontsize=12)

    # 1. Training history
    ax = axes[0]
    ax.plot(history['val_corr'], color='green', label='Val Corr')
    ax.axhline(0.85, color='red', linestyle='--', label='Hedef 0.85')
    ax.axhline(base_corrs.mean(), color='gray', linestyle=':', label='Base ort.')
    ax.set_title('Val Korelasyon')
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Korelasyon dagilimi karsilastirma
    ax = axes[1]
    ax.hist(base_corrs,  bins=20, alpha=0.5, color='gray',      label=f'Base (ort={base_corrs.mean():.2f})')
    ax.hist(best_corrs,  bins=20, alpha=0.7, color='steelblue', label=f'Fine-tuned (ort={best_corrs.mean():.2f})')
    ax.axvline(0.85, color='red', linestyle='--', label='Hedef')
    ax.set_xlabel('Pearson Korelasyon')
    ax.set_ylabel('Pencere Sayisi')
    ax.set_title('Korelasyon Dagilimi')
    ax.legend(fontsize=8)

    # 3. En iyi pencere
    ax = axes[2]
    device_ = next(model.parameters()).device
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b = X_b.to(device_)
            out = model(X_b)
            for i in range(out.shape[0]):
                preds.append(out[i,0].cpu().numpy())
                trues.append(y_b[i,0].numpy())
    best_idx = int(np.argmax([np.corrcoef(p,t)[0,1] for p,t in zip(preds,trues)]))
    c_best = np.corrcoef(preds[best_idx], trues[best_idx])[0,1]
    t_arr = np.arange(len(preds[best_idx])) / FS
    ax.plot(t_arr, trues[best_idx], label='Gercek EKG', linewidth=0.9, alpha=0.85)
    ax.plot(t_arr, preds[best_idx], label=f'Tahmin (corr={c_best:.2f})', linewidth=0.9, alpha=0.85)
    ax.set_xlabel('Zaman (s)')
    ax.set_title('En Iyi Val Penceresi')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(BASE, f'finetune_s{session_num}.png')
    plt.savefig(out_path, dpi=100)
    print(f"finetune_s{session_num}.png kaydedildi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, default=None,
                        help='Session numarasi (varsayilan: 45 ve 59)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-5)
    args = parser.parse_args()

    sessions = [args.session] if args.session else [45, 59, 44]

    results = {}
    for s in sessions:
        corrs = finetune(s, epochs=args.epochs, lr=args.lr)
        if corrs is not None:
            results[s] = corrs

    if len(results) > 1:
        print(f"\n{'='*50}")
        print("GENEL OZET:")
        for s, c in results.items():
            print(f"  S{s}: ort={c.mean()*100:.1f}%  medyan={np.median(c)*100:.1f}%  "
                  f">85%={(c>0.85).mean()*100:.1f}%")
