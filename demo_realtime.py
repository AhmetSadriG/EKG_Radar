"""
Real-Time mmWave Radar ECG Demo
================================
Plays back a recorded radar session as a live stream.
Window updates every STEP_SIZE frames.

Usage:
  python demo_realtime.py              # random session, 2x speed
  python demo_realtime.py --session 59
  python demo_realtime.py --speed 3
"""

import os, sys, argparse, warnings, random, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import torch
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from train_transformer_model import CNNTransformerECG, bandpass_ecg
from scipy.signal import welch

FS        = 200.0      # Hz — radar frame rate
WIN_SIZE  = 640        # samples per window (~3.2 s)
STEP_SIZE = 100        # frames per animation step (~0.5 s)
DISPLAY_S = 10.0       # seconds shown on screen
DISP_LEN  = int(DISPLAY_S * FS)


def pick_random_session():
    """Pick a random session that has both a dataset and a fine-tuned model."""
    ft_files = glob.glob(os.path.join(BASE, 'finetuned_s*.pth'))
    candidates = []
    for ft in ft_files:
        num = int(os.path.basename(ft).replace('finetuned_s', '').replace('.pth', ''))
        if os.path.exists(os.path.join(BASE, f'dataset_s{num}.npz')):
            candidates.append(num)
    if not candidates:
        return 45
    return random.choice(candidates)


# ─── Model ve Veri Yukle ──────────────────────────────────────────────────────

def load_model(session_num):
    model = CNNTransformerECG(in_channels=26, d_model=128, nhead=4,
                              num_transformer_layers=3, dropout=0.0)
    ft_path = os.path.join(BASE, f'finetuned_s{session_num}.pth')
    base_path = os.path.join(BASE, 'best_transformer_model.pth')
    if os.path.exists(ft_path):
        model.load_state_dict(torch.load(ft_path, map_location='cpu'))
        model_tag = f'Fine-tuned S{session_num}'
    else:
        model.load_state_dict(torch.load(base_path, map_location='cpu'))
        model_tag = 'Base Model'
    model.eval()
    return model, model_tag


def load_session_data(session_num):
    path = os.path.join(BASE, f'dataset_s{session_num}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'dataset_s{session_num}.npz bulunamadi!')
    d = np.load(path, allow_pickle=True)
    X = d['X'].astype(np.float32)
    y = bandpass_ecg(d['y'].astype(np.float32))

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    y = (y - y.mean()) / (y.std() + 1e-6)
    return X, y


def estimate_bpm(signal, fs=FS):
    if len(signal) < 128:
        return 0.0
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 256))
    mask = (freqs >= 0.75) & (freqs <= 2.0)
    if not mask.any():
        return 0.0
    return float(freqs[mask][np.argmax(psd[mask])] * 60.0)


# ─── Ana Demo ─────────────────────────────────────────────────────────────────

def run_demo(session_num=45, speed=1, show_ecg=False):
    print(f'Loading session S{session_num}...')
    X_all, y_all = load_session_data(session_num)
    model, model_tag = load_model(session_num)
    total_frames = len(X_all)
    print(f'Total frames: {total_frames}  ({total_frames/FS:.1f} s)')
    print(f'Model: {model_tag}')

    # Bufferlar — ekran genisligi kadar veri tut
    pred_buf = np.zeros(DISP_LEN) * np.nan
    ecg_buf  = np.zeros(DISP_LEN) * np.nan
    bpm_hist = []
    corr_hist = []

    # ─── Grafik Kurulumu ──────────────────────────────────────────────────────
    plt.rcParams.update({'font.size': 11})
    fig = plt.figure(figsize=(14, 8), facecolor='#0D1117')
    fig.canvas.manager.set_window_title('mmWave Radar - Gercek Zamanli EKG Demo')

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.88, bottom=0.08, left=0.07, right=0.97)

    # BPM display (top right)
    ax_bpm = fig.add_subplot(gs[0, 2])
    ax_bpm.set_facecolor('#0D1117')
    ax_bpm.set_xlim(0, 1); ax_bpm.set_ylim(0, 1)
    ax_bpm.axis('off')
    bpm_text   = ax_bpm.text(0.5, 0.65, '-- BPM', ha='center', va='center',
                              fontsize=38, fontweight='bold', color='#00FF88')
    model_text = ax_bpm.text(0.5, 0.18, model_tag, ha='center', va='center',
                              fontsize=8, color='#AAAAAA')
    ax_bpm.text(0.5, 0.92, 'Heart Rate', ha='center', va='top',
                fontsize=10, color='#888888')

    # Correlation / quality indicator
    ax_corr = fig.add_subplot(gs[1, 2])
    ax_corr.set_facecolor('#0D1117')
    ax_corr.set_xlim(0, 1); ax_corr.set_ylim(0, 1)
    ax_corr.axis('off')
    corr_text = ax_corr.text(0.5, 0.6, 'r = --', ha='center', va='center',
                              fontsize=24, fontweight='bold', color='#4FC3F7')
    ax_corr.text(0.5, 0.92, 'ECG Similarity (Pearson r)', ha='center', va='top',
                 fontsize=9, color='#888888')
    quality_text = ax_corr.text(0.5, 0.2, '', ha='center', va='center',
                                 fontsize=11, color='#FFD700')

    # BPM history plot
    ax_bpm_hist = fig.add_subplot(gs[2, 2])
    ax_bpm_hist.set_facecolor('#111827')
    ax_bpm_hist.set_xlabel('Time (s)', color='#AAAAAA')
    ax_bpm_hist.set_ylabel('BPM', color='#AAAAAA')
    ax_bpm_hist.set_title('BPM History', color='white', fontsize=9)
    ax_bpm_hist.tick_params(colors='#AAAAAA')
    for spine in ax_bpm_hist.spines.values():
        spine.set_edgecolor('#333333')
    ax_bpm_hist.set_ylim(40, 130)
    bpm_line, = ax_bpm_hist.plot([], [], color='#00FF88', linewidth=1.5)
    ax_bpm_hist.axhline(60, color='#FF6B6B', linestyle='--', alpha=0.4, linewidth=0.7)
    ax_bpm_hist.axhline(100, color='#FF6B6B', linestyle='--', alpha=0.4, linewidth=0.7)
    ax_bpm_hist.grid(True, alpha=0.12, color='#444444')

    # Main ECG reconstruction panel
    ax_pred = fig.add_subplot(gs[0:2, 0:2])
    ax_pred.set_facecolor('#111827')
    ax_pred.set_title('ECG Reconstruction from Radar Signal', color='white', fontsize=11)
    ax_pred.set_xlabel('Time (s)', color='#AAAAAA')
    ax_pred.set_ylabel('Normalized Amplitude', color='#AAAAAA')
    ax_pred.tick_params(colors='#AAAAAA')
    for spine in ax_pred.spines.values():
        spine.set_edgecolor('#333333')
    ax_pred.set_xlim(0, DISPLAY_S)
    ax_pred.set_ylim(-4, 4)
    ax_pred.grid(True, alpha=0.12, color='#444444')
    t_disp = np.linspace(0, DISPLAY_S, DISP_LEN)
    pred_line, = ax_pred.plot(t_disp, pred_buf, color='#FF6B6B',
                               linewidth=1.4, label='Radar Prediction', alpha=0.9)
    ecg_line, = ax_pred.plot(t_disp, ecg_buf, color='#4FC3F7',
                              linewidth=1.0, label='Ground Truth ECG', alpha=0.7,
                              visible=show_ecg)
    legend = ax_pred.legend(loc='upper right', facecolor='#1a1a2e',
                             edgecolor='#333333', labelcolor='white', fontsize=9)

    time_text = ax_pred.text(0.02, 0.95, 't = 0.0 s', transform=ax_pred.transAxes,
                              fontsize=9, color='#888888', va='top')

    hint_text = ax_pred.text(0.98, 0.95,
                              '[E] show/hide ECG   [Space] pause   [Q] quit',
                              transform=ax_pred.transAxes,
                              fontsize=8, color='#555555', va='top', ha='right')

    # Bottom: ground truth ECG reference
    ax_ecg = fig.add_subplot(gs[2, 0:2])
    ax_ecg.set_facecolor('#111827')
    ax_ecg.set_title('Reference ECG (Fabenode)', color='white', fontsize=9)
    ax_ecg.set_xlabel('Time (s)', color='#AAAAAA')
    ax_ecg.tick_params(colors='#AAAAAA')
    for spine in ax_ecg.spines.values():
        spine.set_edgecolor('#333333')
    ax_ecg.set_xlim(0, DISPLAY_S)
    ax_ecg.set_ylim(-4, 4)
    ax_ecg.grid(True, alpha=0.12, color='#444444')
    ecg_ref_line, = ax_ecg.plot(t_disp, ecg_buf.copy(), color='#4FC3F7',
                                  linewidth=1.0, alpha=0.9)
    ax_ecg.set_visible(show_ecg)

    # Title
    fig.text(0.5, 0.95,
             f'Real-Time Cardiac Monitoring with mmWave Radar  |  Session S{session_num}',
             ha='center', va='top', fontsize=13, fontweight='bold', color='white')

    # ── "Press Space to start" overlay ───────────────────────────────────────
    overlay_text = fig.text(
        0.5, 0.50,
        'Press  SPACE  to start',
        ha='center', va='center',
        fontsize=28, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#1a1a2e',
                  edgecolor='#4FC3F7', linewidth=2, alpha=0.92)
    )
    subtitle_text = fig.text(
        0.5, 0.42,
        f'Session S{session_num}  |  {model_tag}',
        ha='center', va='center',
        fontsize=11, color='#AAAAAA'
    )

    # ─── Animasyon State ──────────────────────────────────────────────────────
    # Ilk 2 saniyeyi atla (stabilizasyon icin)
    START_FRAME = WIN_SIZE + int(2.0 * FS)

    state = {
        'frame_idx':  START_FRAME,
        'pred_buf':   pred_buf,
        'ecg_buf':    ecg_buf,
        'bpm_hist':   bpm_hist,
        'corr_hist':  corr_hist,
        'times':      [],
        'current_bpm':  0.0,
        'current_corr': 0.0,
        'show_ecg':   show_ecg,
        'paused':     False,
        'waiting':    True,   # waiting for first Space press
    }

    # ─── Keyboard Controls ────────────────────────────────────────────────────
    def on_key(event):
        if event.key in ('e', 'E'):
            state['show_ecg'] = not state['show_ecg']
            visible = state['show_ecg']
            ecg_line.set_visible(visible)
            ax_ecg.set_visible(visible)
            fig.canvas.draw_idle()

        elif event.key == ' ':
            if state['waiting']:
                # First press: remove overlay and start
                state['waiting'] = False
                overlay_text.set_visible(False)
                subtitle_text.set_visible(False)
                ani.resume()
                fig.canvas.draw_idle()
            else:
                # Subsequent presses: pause / resume
                state['paused'] = not state['paused']
                if state['paused']:
                    ani.pause()
                    hint_text.set_text('[E] show/hide ECG   [Space] RESUME   [Q] quit')
                else:
                    ani.resume()
                    hint_text.set_text('[E] show/hide ECG   [Space] pause   [Q] quit')
                fig.canvas.draw_idle()

        elif event.key in ('q', 'Q', 'escape'):
            plt.close('all')

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(frame_num):
        s = state
        if s['waiting']:
            return []   # do nothing until Space is pressed

        fi = s['frame_idx']

        if fi + STEP_SIZE > total_frames:
            fi = WIN_SIZE + int(2.0 * FS)  # loop (stabilizasyon sonrasina don)
            s['pred_buf'][:] = np.nan
            s['ecg_buf'][:]  = np.nan
            s['bpm_hist'].clear()
            s['corr_hist'].clear()
            s['times'].clear()

        # Mevcut pencere
        win_X = X_all[fi - WIN_SIZE:fi]   # (640, 26)
        win_y = y_all[fi - WIN_SIZE:fi]   # (640,)

        # Model inference
        with torch.no_grad():
            inp = torch.tensor(win_X.T[np.newaxis]).float()   # (1, 26, 640)
            pred = model(inp)[0, 0].numpy()                    # (640,)

        # BPM hesapla
        bpm = estimate_bpm(pred)
        corr = float(np.corrcoef(pred, win_y)[0, 1]) if not np.isnan(pred).any() else 0.0
        s['current_bpm']  = bpm
        s['current_corr'] = corr

        # Buffer'a ekle (en son STEP_SIZE ornegi)
        s['pred_buf'] = np.roll(s['pred_buf'], -STEP_SIZE)
        s['ecg_buf']  = np.roll(s['ecg_buf'],  -STEP_SIZE)
        s['pred_buf'][-STEP_SIZE:] = pred[-STEP_SIZE:]
        s['ecg_buf'][-STEP_SIZE:]  = win_y[-STEP_SIZE:]

        t_sec = fi / FS
        s['bpm_hist'].append(bpm)
        s['times'].append(t_sec)

        # Ekrani guncelle
        pred_line.set_ydata(s['pred_buf'])
        ecg_line.set_ydata(s['ecg_buf'])          # her zaman guncelle, visibility on_key ile
        ecg_ref_line.set_ydata(s['ecg_buf'])

        # Dinamik y ekseni
        valid = s['pred_buf'][~np.isnan(s['pred_buf'])]
        if len(valid) > 0:
            yc = np.percentile(np.abs(valid), 95)
            ax_pred.set_ylim(-max(yc * 1.3, 1.5), max(yc * 1.3, 1.5))
            ax_ecg.set_ylim(-max(yc * 1.3, 1.5), max(yc * 1.3, 1.5))

        # BPM metni + renk
        bpm_color = '#00FF88' if 50 < bpm < 110 else '#FF4444'
        bpm_text.set_text(f'{bpm:.0f} BPM')
        bpm_text.set_color(bpm_color)

        # Korelasyon metni + kalite
        corr_color = '#00FF88' if corr >= 0.85 else ('#FFD700' if corr >= 0.70 else '#FF4444')
        corr_text.set_text(f'r = {corr:.3f}')
        corr_text.set_color(corr_color)
        if corr >= 0.85:
            quality_text.set_text('High Quality')
            quality_text.set_color('#00FF88')
        elif corr >= 0.70:
            quality_text.set_text('Medium Quality')
            quality_text.set_color('#FFD700')
        else:
            quality_text.set_text('Low Quality')
            quality_text.set_color('#FF4444')

        # BPM gecmis grafigi
        t_arr = np.array(s['times'])
        b_arr = np.array(s['bpm_hist'])
        if len(t_arr) > 1:
            t_rel = t_arr - t_arr[0]
            bpm_line.set_data(t_rel, b_arr)
            ax_bpm_hist.set_xlim(0, max(t_rel[-1] + 1, 10))

        time_text.set_text(f't = {t_sec:.1f} s')
        s['frame_idx'] = fi + STEP_SIZE

        return [pred_line, ecg_line, ecg_ref_line,
                bpm_text, corr_text, quality_text, bpm_line, time_text]

    interval_ms = int(STEP_SIZE / FS * 1000 / speed)
    print(f'Ready. Press SPACE in the window to start. ({speed}x speed)')

    ani = animation.FuncAnimation(
        fig, update,
        interval=interval_ms,
        blit=False,
        cache_frame_data=False
    )

    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mmWave Radar ECG Demo')
    parser.add_argument('--session', type=int, default=None,
                        help='Session number (default: random)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed (1=real-time, 2=2x, default: 1)')
    parser.add_argument('--show-ecg', action='store_true',
                        help='Show ground truth ECG on startup (default: hidden)')
    args = parser.parse_args()

    session = args.session if args.session is not None else 59
    print(f'Selected session: S{session}')

    run_demo(session_num=session,
             speed=args.speed,
             show_ecg=args.show_ecg)
