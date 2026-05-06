"""
LexKo: Spectrum Analyzer — Streamlit Version
pip install -r requirements.txt
streamlit run app.py
"""

import io
import os
import colorsys
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import librosa
import librosa.display
import cv2
from PIL import Image

# ═══════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════
N_MELS     = 256
HOP_LENGTH = 512
COLORMAP   = "magma"

BANDS = [
    ("Sub-Bass",  (0,   200, 255), 20,    60,    "sub_bass"),
    ("Bass",      (0,   255, 100), 60,    250,   "bass"),
    ("Low-Mid",   (255, 255,   0), 250,   2000,  "low_mid"),
    ("High-Mid",  (255, 128,   0), 2000,  6000,  "high_mid"),
    ("Air/High",  (200,   0, 255), 6000,  20000, "high"),
]

BAND_COLORS_HEX = {
    "sub_bass": "#00C8FF",
    "bass":     "#00FF64",
    "low_mid":  "#FFFF00",
    "high_mid": "#FF8000",
    "high":     "#C800FF",
}

# ═══════════════════════════════════════════════════════════
# Image Helpers
# ═══════════════════════════════════════════════════════════
def bgr_to_pil(bgr_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

def bgr_to_bytes(bgr_img: np.ndarray) -> io.BytesIO:
    buf = io.BytesIO()
    bgr_to_pil(bgr_img).save(buf, format="PNG")
    buf.seek(0)
    return buf

# ═══════════════════════════════════════════════════════════
# Core Analysis
# ═══════════════════════════════════════════════════════════
def generate_span_image(y: np.ndarray, sr: int) -> np.ndarray:
    """SPAN-style averaged spectrum — returns BGR numpy array."""
    D = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024))
    avg_db   = librosa.amplitude_to_db(np.mean(D, axis=1), ref=np.max)
    peak_db  = librosa.amplitude_to_db(np.max(D,  axis=1), ref=np.max)
    freqs    = librosa.fft_frequencies(sr=sr, n_fft=4096)

    W, H = 1200, 500
    canvas = np.full((H, W, 3), (18, 18, 18), dtype=np.uint8)
    grid   = (40, 40, 40)

    def hz_to_x(f):
        f = np.clip(f, 20, 20000)
        return int((np.log10(f) - np.log10(20)) / (np.log10(20000) - np.log10(20)) * (W - 1))

    for f in [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        x = hz_to_x(f)
        cv2.line(canvas, (x, 0), (x, H), grid, 1)
        label = f"{f//1000}k" if f >= 1000 else str(f)
        cv2.putText(canvas, label, (x + 5, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)

    for db in range(0, -100, -20):
        yp = int((db / -80) * (H - 50))
        cv2.line(canvas, (0, yp), (W, yp), grid, 1)
        cv2.putText(canvas, f"{db}dB", (W - 45, yp - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)

    mask   = (freqs >= 20) & (freqs <= 20000)
    f_plot = freqs[mask]
    step   = max(1, len(f_plot) // (W // 2))

    def build_pts(db_arr):
        pts = []
        for i in range(0, len(f_plot), step):
            x   = hz_to_x(f_plot[i])
            yv  = int(np.clip(db_arr[mask][i] / -80, 0, 1) * (H - 50))
            pts.append([x, yv])
        return np.array(pts, np.int32)

    avg_pts  = build_pts(avg_db)
    peak_pts = build_pts(peak_db)

    overlay   = canvas.copy()
    fill_pts  = np.vstack([avg_pts, [[avg_pts[-1][0], H], [avg_pts[0][0], H]]])
    cv2.fillPoly(overlay, [fill_pts], (40, 180, 80))
    cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
    cv2.polylines(canvas, [avg_pts],  False, (80, 255, 120), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [peak_pts], False, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Voxengo SPAN Style Analysis", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    return canvas


def generate_mel_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    """Mel spectrogram — returns BGR numpy array."""
    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, fmin=20, fmax=20000)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH,
                             x_axis=None, y_axis=None, fmin=20, fmax=20000,
                             cmap=COLORMAP, ax=ax)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)
    buf.seek(0)

    arr = np.array(Image.open(buf).convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def analyze_mel_spectrogram(spec_bgr: np.ndarray, sr: int):
    """Band intensity analysis on a mel spectrogram using frequency mapping."""
    img_gray  = cv2.cvtColor(spec_bgr, cv2.COLOR_BGR2GRAY)
    H, W      = img_gray.shape
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=20, fmax=sr // 2)

    intensities, band_rows = {}, {}
    for _, _, f_low, f_high, key in BANDS:
        il = int(np.clip(np.searchsorted(mel_freqs, f_low),  0, N_MELS - 1))
        ih = int(np.clip(np.searchsorted(mel_freqs, f_high), 0, N_MELS - 1))
        rt = int(H - (ih / N_MELS) * H)
        rb = int(H - (il / N_MELS) * H)
        rt, rb = max(0, min(rt, H - 1)), max(0, min(rb, H - 1))
        if rt >= rb:
            rb = min(rt + 1, H - 1)
        region = img_gray[rt:rb, :]
        intensities[key] = float(np.mean(region)) if region.size > 0 else 0.0
        band_rows[key]   = (rt, rb)
    return intensities, band_rows


def analyze_spectrogram_image(img_bgr: np.ndarray):
    """Band analysis on an arbitrary spectrogram image (uniform row split)."""
    H_std, W_std = 400, 1200
    resized      = cv2.resize(img_bgr, (W_std, H_std))
    img_gray     = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    H            = H_std
    h_step       = H / len(BANDS)

    intensities, band_rows = {}, {}
    for i, (_, _, _, _, key) in enumerate(BANDS):
        idx_inv = len(BANDS) - 1 - i
        rt = int(idx_inv * h_step)
        rb = int((idx_inv + 1) * h_step)
        region = img_gray[rt:rb, :]
        intensities[key] = float(np.mean(region)) if region.size > 0 else 0.0
        band_rows[key]   = (rt, rb)
    return resized, intensities, band_rows


def make_rules(intensities: dict, audio_type: str) -> list:
    sb, bs = intensities["sub_bass"], intensities["bass"]
    lm, hm = intensities["low_mid"],  intensities["high_mid"]
    hi      = intensities["high"]
    avg     = np.mean(list(intensities.values()))
    alerts  = []

    if audio_type == "vocal":
        if lm > avg * 1.3:
            alerts.append(("low_mid",  "⚠ Low-Mid 過強 → 聲音太混濁 (Muddy)\n建議在 200–500 Hz 進行 EQ 衰減"))
        if hm < avg * 0.7:
            alerts.append(("high_mid", "⚠ High-Mid 偏弱 → 人聲不夠突出\n建議提升 3 kHz 附近 (Presence)"))
        if sb > avg * 1.4:
            alerts.append(("sub_bass", "⚠ Sub-Bass 過重 → 低頻隆隆聲\n建議高通濾波器 (HPF) 從 80 Hz 切除"))
        if hi < avg * 0.5:
            alerts.append(("high",     "ℹ Air/High 較暗 → 空氣感不足\n可嘗試提升 10–16 kHz 增加光澤"))
        if not alerts:
            alerts.append(("low_mid",  "✅ 頻譜分佈良好！人聲音調平衡 (Tonal Balance) 佳"))

    elif audio_type == "master":
        if sb > avg * 1.5:
            alerts.append(("sub_bass", "⚠ Sub-Bass 過載 → 動態餘裕 (Headroom) 不足\n建議低頻限幅或衰減"))
        if bs > avg * 1.4:
            alerts.append(("bass",     "⚠ Bass 能量過重\n建議 60–150 Hz 適當衰減"))
        if hm < avg * 0.6:
            alerts.append(("high_mid", "⚠ 高中頻不足 → 整體聽起來悶\n建議提升 2–5 kHz 清晰度"))
        if hi < avg * 0.5:
            alerts.append(("high",     "ℹ 高頻偏暗\n建議小量提升 8–12 kHz 通透感"))
        if not alerts:
            alerts.append(("low_mid",  "✅ 母帶頻譜均勻，Tonal Balance 良好"))

    elif audio_type == "instrument":
        if lm > avg * 1.35:
            alerts.append(("low_mid",  "⚠ Low-Mid 堆積 (Frequency Masking)\n建議挖除 250–500 Hz 共鳴峰"))
        if hm > avg * 1.4:
            alerts.append(("high_mid", "⚠ High-Mid 過亮/刺耳\n建議在 4–6 kHz 稍作衰減"))
        if sb < avg * 0.4 and bs < avg * 0.6:
            alerts.append(("bass",     "ℹ 低頻根基薄弱\n節奏樂器可小幅提升 80–120 Hz"))
        if not alerts:
            alerts.append(("high_mid", "✅ 樂器頻段分佈合理，動態表現良好"))

    return alerts


def generate_minimal_album_art(intensities: dict, size: int = 512) -> np.ndarray:
    """Gradient album art derived from band intensities — returns BGR numpy array."""
    sb, bs = intensities.get("sub_bass", 0), intensities.get("bass", 0)
    lm, hm = intensities.get("low_mid", 0),  intensities.get("high_mid", 0)
    hi      = intensities.get("high", 0)

    seed_a = (sb * 17.3 + hi * 31.7) * (sb + hi + 1)
    hue_a  = (seed_a * 0.61803398875) % 1.0
    seed_b = (bs * 13.1 + lm * 19.3 + hm * 23.9) * 7.5
    hue_b  = (seed_b * 0.61803398875) % 1.0

    total  = (sb + bs + lm + hm + hi) / 5
    sat    = 0.6 + (total / 255.0) * 0.3
    lum    = 0.35 + (total / 255.0) * 0.3

    def hsl_to_bgr(h, s, l):
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return np.array([b * 255, g * 255, r * 255], dtype=np.uint8)

    ca = hsl_to_bgr(hue_a, sat, lum)
    cb = hsl_to_bgr(hue_b, sat, lum)

    ws = 64
    corners = np.zeros((2, 2, 3), dtype=np.uint8)
    corners[0, 0] = ca
    corners[1, 1] = cb
    mid = ((ca.astype(np.int16) + cb.astype(np.int16)) // 2).astype(np.uint8)
    corners[0, 1] = corners[1, 0] = mid

    temp   = cv2.resize(corners, (ws, ws), interpolation=cv2.INTER_CUBIC)
    bk     = (ws - 1) | 1
    temp   = cv2.GaussianBlur(temp, (bk, bk), 0)
    canvas = cv2.resize(temp, (size, size), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.GaussianBlur(canvas, (31, 31), 0)

    noise  = np.random.randint(-8, 8, (size, size, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    mask_s = np.zeros((ws, ws), dtype=np.uint8)
    cv2.circle(mask_s, (ws // 2, ws // 2), int(ws * 0.8), 255, -1)
    mask_b = cv2.GaussianBlur(mask_s, (ws | 1, ws | 1), 0) / 255.0
    mask_f = cv2.resize(mask_b, (size, size))
    for i in range(3):
        canvas[:, :, i] = (canvas[:, :, i] * (0.85 + 0.15 * mask_f)).astype(np.uint8)

    return canvas


def make_result_image(spec_bgr: np.ndarray, intensities: dict, band_rows: dict,
                      alerts: list, audio_type: str, sr: int, duration: float) -> np.ndarray:
    """Full annotated result image: spectrogram + band bar chart — returns BGR numpy array."""
    H_spec, W = spec_bgr.shape[:2]
    H_bar     = 450
    H_total   = H_spec + H_bar + 10
    canvas    = np.zeros((H_total, W, 3), dtype=np.uint8)
    canvas[:H_spec, :] = spec_bgr

    alerted = {k for k, _ in alerts}
    for name, color_rgb, _, _, key in BANDS:
        rt, rb = band_rows[key]
        bgr    = (color_rgb[2], color_rgb[1], color_rgb[0])
        thick  = 4 if key in alerted else 2
        cv2.rectangle(canvas, (0, rt), (W - 1, rb), bgr, thick)
        cv2.putText(canvas, name, (10, rt + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2, cv2.LINE_AA)

    canvas[H_spec:H_spec + 10, :] = (60, 60, 60)
    bar_y0 = H_spec + 10
    canvas[bar_y0:, :] = (18, 18, 18)

    for pct in [25, 50, 75, 100]:
        cv2.line(canvas, (int(W * pct / 100), bar_y0), (int(W * pct / 100), H_total - 1), (40, 40, 40), 1)
    for rp in [0.25, 0.5, 0.75]:
        gy = int(bar_y0 + H_bar * rp)
        cv2.line(canvas, (0, gy), (W - 1, gy), (35, 35, 35), 1)

    n      = len(BANDS)
    bar_w  = W // (n * 2 + 1)
    gap    = bar_w // 2

    for i, (name, color_rgb, _, _, key) in enumerate(BANDS):
        val    = intensities[key]
        bh     = int((H_bar - 100) * (val / 255.0))
        bx     = gap + i * (bar_w + gap)
        by_bot = H_total - 60
        by_top = by_bot - bh
        bgr    = (color_rgb[2], color_rgb[1], color_rgb[0])

        if key in alerted:
            cv2.rectangle(canvas, (bx - 4, by_top - 4), (bx + bar_w + 4, by_bot + 4), bgr, 2)

        for row in range(by_top, by_bot):
            alpha = 1.0 - (row - by_top) / max(bh, 1) * 0.6
            c = tuple(int(ch * alpha) for ch in bgr)
            cv2.line(canvas, (bx, row), (bx + bar_w, row), c, 1)

        cv2.line(canvas, (bx, by_top), (bx + bar_w, by_top), bgr, 3)
        cv2.putText(canvas, f"{val:.1f}", (bx, by_top - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2, cv2.LINE_AA)
        cv2.putText(canvas, name, (bx, by_bot + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, bgr, 2, cv2.LINE_AA)

    cv2.putText(canvas,
                f"LexKo: Spectrum Analyzer | {audio_type.upper()} | {duration:.1f}s | {sr}Hz",
                (20, H_total - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
    return canvas


def make_contrast_image(intensities_my: dict, intensities_ref: dict) -> np.ndarray:
    """Side-by-side bar chart comparing two tracks — returns BGR numpy array."""
    W, H   = 1000, 500
    canvas = np.full((H, W, 3), (18, 18, 18), dtype=np.uint8)
    cv2.putText(canvas, "Mixing Contrast Analysis (My Track vs Reference)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)

    for y in range(100, H - 50, 50):
        cv2.line(canvas, (50, y), (W - 50, y), (40, 40, 40), 1)

    n       = len(BANDS)
    group_w = (W - 100) // n
    bar_w   = group_w // 3
    by_bot  = H - 60

    for i, (name, color_rgb, _, _, key) in enumerate(BANDS):
        bgr     = (color_rgb[2], color_rgb[1], color_rgb[0])
        bx_my   = 70 + i * group_w
        bx_ref  = bx_my + bar_w + 10

        bh_my  = int((H - 150) * (intensities_my[key]  / 255))
        bh_ref = int((H - 150) * (intensities_ref[key] / 255))

        cv2.rectangle(canvas, (bx_my,  by_bot - bh_my),  (bx_my  + bar_w, by_bot), bgr, -1)
        cv2.rectangle(canvas, (bx_ref, by_bot - bh_ref), (bx_ref + bar_w, by_bot), (150, 150, 150), -1)

        cv2.putText(canvas, "My",  (bx_my,  by_bot + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr, 1)
        cv2.putText(canvas, "Ref", (bx_ref, by_bot + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        diff       = intensities_my[key] - intensities_ref[key]
        diff_color = (100, 100, 255) if diff > 15 else (100, 255, 100) if diff < -15 else (200, 200, 200)
        cv2.putText(canvas, f"{diff:+.1f}",
                    (bx_my, by_bot - max(bh_my, bh_ref) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, diff_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, name, (bx_my, H - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

    return canvas


def get_reference_advice(intensities_my: dict, intensities_ref: dict) -> list:
    advice = []
    for name, _, _, _, key in BANDS:
        diff = intensities_my[key] - intensities_ref[key]
        if diff > 30:
            advice.append(f"⚠ **{name}**：你的能量過多，建議衰減。")
        elif diff < -30:
            advice.append(f"ℹ **{name}**：參考音軌在此頻段更厚實，建議適度提升。")
    if not advice:
        advice.append("✅ 兩者頻譜結構非常接近，維持現狀即可！")
    return advice


# ═══════════════════════════════════════════════════════════
# Streamlit App
# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════
# Streamlit App
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LexKo: Spectrum Analyzer",
    page_icon="🎵",
    layout="wide",
)

# --- 核心視覺修正：精品極簡風 ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    
    /* 背景與基礎字體 */
    .stApp {
        background-color: #1A1A1A;
        font-family: 'Inter', sans-serif;
    }
    
    /* 強力注入標題樣式 (用 !important 確保覆蓋 Streamlit 預設) */
    h1 {
        font-family: 'Arial Black', sans-serif !important;
        color: #1ABC9C !important;
        font-size: 4.5rem !important;
        font-weight: 900 !important;
        letter-spacing: -3px !important;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
        line-height: 1 !important;
    }
    
    /* 副標題：法律審判感的灰 */
    .brand-subtitle {
        color: #AAAAAA;
        font-family: 'Segoe UI', sans-serif;
        font-size: 1.1rem;
        font-weight: 400;
        letter-spacing: 2px;
        margin-top: -10px;
        margin-bottom: 40px;
        text-transform: uppercase;
    }

    /* 調整元件圓角與邊框 */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
    
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #333333;
        background-color: #262626;
        color: white;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border-color: #1ABC9C;
        color: #1ABC9C;
        background-color: #1A1A1A;
        box-shadow: 0 0 15px rgba(26, 188, 156, 0.1);
    }
    
    /* 側邊欄樣式 */
    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #222222;
    }

    /* 標籤頁樣式優化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 8px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ── Session state init ──────────────────────────────────────
_defaults = {
    "mel_bgr":       None,
    "span_bgr":      None,
    "spec_bgr":      None,   # spec used for analysis (mel or uploaded img)
    "intensities":   None,
    "band_rows":     None,
    "sr":            44100,
    "duration":      0.0,
    "result_bgr":    None,
    "art_bgr":       None,
    "contrast_bgr":  None,
    "ref_intensities": None,
    "alerts":        None,
    "audio_file_id": None,
    "img_file_id":   None,
    "ref_file_id":   None,
    "audio_type":    "master",
    "base_name":     "lexko",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════
st.title("LexKo")
st.markdown('<p class="brand-subtitle">Spectrum Analyzer</p>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# Main Upload Area
# ═══════════════════════════════════════════════════════════
col_u1, col_u2 = st.columns(2)

with col_u1:
    audio_file = st.file_uploader(
        "🎵 Audio File",
        type=["wav", "mp3", "flac", "m4a"],
        help="WAV · MP3 · FLAC · M4A (Drag & Drop supported)"
    )

with col_u2:
    img_file = st.file_uploader(
        "🖼 Spectrogram Image",
        type=["png", "jpg", "jpeg"],
        help="Upload an existing spectrogram PNG/JPG"
    )

# ── Process audio upload ────────────────────────────────
if audio_file is not None and audio_file.file_id != st.session_state.audio_file_id:
    st.session_state.audio_file_id = audio_file.file_id
    st.session_state.base_name = os.path.splitext(audio_file.name)[0]
    # reset previous results
    for k in ("mel_bgr", "span_bgr", "spec_bgr", "intensities", "band_rows",
                "result_bgr", "art_bgr", "contrast_bgr", "ref_intensities",
                "alerts", "img_file_id"):
        st.session_state[k] = None

    with st.status("Analyzing audio...", expanded=True) as status:
        audio_bytes = audio_file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        duration = len(y) / sr

        mel_bgr  = generate_mel_spectrogram(y, sr)
        span_bgr = generate_span_image(y, sr)
        intensities, band_rows = analyze_mel_spectrogram(mel_bgr, sr)

        st.session_state.update({
            "mel_bgr":     mel_bgr,
            "span_bgr":    span_bgr,
            "spec_bgr":    mel_bgr,
            "intensities": intensities,
            "band_rows":   band_rows,
            "sr":          sr,
            "duration":    duration,
        })
        status.update(label=f"✅ {sr} Hz · {duration:.1f} s Analysis Complete", state="complete", expanded=False)

# ── Process image upload ────────────────────────────────
if img_file is not None and img_file.file_id != st.session_state.img_file_id:
    st.session_state.img_file_id = img_file.file_id
    st.session_state.base_name = os.path.splitext(img_file.name)[0]
    for k in ("mel_bgr", "span_bgr", "spec_bgr", "intensities", "band_rows",
                "result_bgr", "art_bgr", "contrast_bgr", "ref_intensities",
                "alerts", "audio_file_id"):
        st.session_state[k] = None

    with st.status("Parsing spectrogram...", expanded=True) as status:
        pil = Image.open(io.BytesIO(img_file.read())).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        resized, intensities, band_rows = analyze_spectrogram_image(img_bgr)

        st.session_state.update({
            "mel_bgr":     resized,
            "span_bgr":    None,
            "spec_bgr":    resized,
            "intensities": intensities,
            "band_rows":   band_rows,
            "sr":          44100,
            "duration":    0.0,
        })
        status.update(label="✅ Spectrogram Parsed", state="complete", expanded=False)

# ═══════════════════════════════════════════════════════════
# Sidebar — Controls
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚖️ Controls")
    
    st.session_state.audio_type = st.selectbox(
        "Analysis Mode",
        ["master", "instrument", "vocal"],
        index=["master", "instrument", "vocal"].index(st.session_state.audio_type),
        format_func=str.upper,
    )

    st.divider()

    # ── Actions (shown only after file loaded) ──────────────
    if st.session_state.intensities is not None:
        st.markdown("### Actions")

        if st.button("📋 Mixing Report", use_container_width=True, type="primary"):
            with st.spinner("Generating report…"):
                intensities = st.session_state.intensities
                band_rows   = st.session_state.band_rows
                spec_bgr    = st.session_state.spec_bgr
                audio_type  = st.session_state.audio_type
                sr          = st.session_state.sr
                duration    = st.session_state.duration

                alerts = make_rules(intensities, audio_type)
                result = make_result_image(spec_bgr, intensities, band_rows,
                                            alerts, audio_type, sr, duration)
                st.session_state.alerts     = alerts
                st.session_state.result_bgr = result
            st.rerun()

        if st.button("🎨 Generate Album Art", use_container_width=True):
            with st.spinner("Creating album art…"):
                art = generate_minimal_album_art(st.session_state.intensities)
                st.session_state.art_bgr = art
            st.rerun()

        st.divider()
        st.markdown("### Reference Contrast")
        ref_file = st.file_uploader(
            "Reference Track",
            type=["wav", "mp3", "flac", "png", "jpg", "jpeg"],
            key="ref_uploader",
        )
        if ref_file is not None and ref_file.file_id != st.session_state.ref_file_id:
            st.session_state.ref_file_id = ref_file.file_id
            with st.spinner("Analyzing reference…"):
                ext = os.path.splitext(ref_file.name)[1].lower()
                ref_bytes = ref_file.read()
                if ext in (".png", ".jpg", ".jpeg"):
                    pil_ref = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
                    ref_bgr = cv2.cvtColor(np.array(pil_ref), cv2.COLOR_RGB2BGR)
                    _, ref_intensities, _ = analyze_spectrogram_image(ref_bgr)
                else:
                    y_ref, sr_ref = librosa.load(io.BytesIO(ref_bytes), sr=None, mono=True)
                    mel_ref = generate_mel_spectrogram(y_ref, sr_ref)
                    ref_intensities, _ = analyze_mel_spectrogram(mel_ref, sr_ref)

                contrast = make_contrast_image(st.session_state.intensities, ref_intensities)
                st.session_state.ref_intensities = ref_intensities
                st.session_state.contrast_bgr    = contrast
            st.rerun()

        st.divider()
        if st.button("🗑️ Clear All", use_container_width=True):
            for k in _defaults:
                st.session_state[k] = _defaults[k]
            st.rerun()
    else:
        st.info("Waiting for files to unlock controls...")

# ═══════════════════════════════════════════════════════════
# Main Display
# ═══════════════════════════════════════════════════════════
if st.session_state.mel_bgr is not None:
    # ── Spectrum tabs ───────────────────────────────────────────
    tab_defs = [("🌈 Spectrum", "mel")]
    if st.session_state.span_bgr is not None:
        tab_defs.append(("📊 SPAN Style", "span"))
    if st.session_state.result_bgr is not None:
        tab_defs.append(("📋 Mixing Report", "result"))
    if st.session_state.art_bgr is not None:
        tab_defs.append(("💿 Album Art", "art"))
    if st.session_state.contrast_bgr is not None:
        tab_defs.append(("⚖️ Contrast", "contrast"))

    tabs = st.tabs([label for label, _ in tab_defs])

    for tab, (_, key) in zip(tabs, tab_defs):
        with tab:
            img = st.session_state[f"{key}_bgr"]
            if img is not None:
                st.image(bgr_to_pil(img), use_container_width=True)
                
                # Dynamic download logic
                if key in ("mel", "span", "result", "art", "contrast"):
                    suffix_map = {
                        "mel": "Spectrum",
                        "span": "Span_Style",
                        "result": "Mixing_Report",
                        "art": "Album_Art",
                        "contrast": "Contrast"
                    }
                    suffix = suffix_map.get(key, key.title())
                    base = st.session_state.get("base_name", "lexko")
                    
                    st.download_button(
                        f"⬇ Download {suffix.replace('_', ' ')}",
                        data=bgr_to_bytes(img),
                        file_name=f"{base}_{suffix}.png",
                        mime="image/png",
                        key=f"dl_{key}"
                    )

    # ── Mixing Advice ───────────────────────────────────────────
    if st.session_state.alerts:
        st.divider()

        # ── Band Energy Cards ───────────────────────────────────────
        st.markdown("#### Band Energy")
        cols = st.columns(5)
        intensities = st.session_state.intensities

        for col, (name, _, _, _, key) in zip(cols, BANDS):
            val = intensities[key]
            pct = val / 255 * 100
            hx  = BAND_COLORS_HEX[key]
            with col:
                st.markdown(f"""
                <div style="background:#212121;border-radius:10px;padding:12px;text-align:center;
                            border-top:3px solid {hx};">
                    <div style="color:{hx};font-weight:700;font-size:0.8rem;">{name}</div>
                    <div style="font-size:1.6rem;font-weight:900;color:#fff;">{val:.1f}</div>
                    <div style="background:#333;border-radius:4px;height:6px;margin-top:8px;">
                        <div style="background:{hx};width:{pct:.0f}%;height:6px;border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        sr       = st.session_state.sr
        duration = st.session_state.duration
        st.markdown(f"#### 🎚 Mixing Analysis — `{st.session_state.audio_type.upper()}`")
        st.caption(f"SR: {sr} Hz · Duration: {duration:.1f} s")

        for _, (key, msg) in enumerate(st.session_state.alerts, 1):
            if msg.startswith("✅"):
                st.success(msg)
            elif msg.startswith("⚠"):
                st.warning(msg)
            else:
                st.info(msg)

    # ── Reference Advice ────────────────────────────────────────
    if st.session_state.ref_intensities is not None:
        st.divider()
        st.markdown("#### ⚖️ Reference Comparison")
        for line in get_reference_advice(st.session_state.intensities, st.session_state.ref_intensities):
            if line.startswith("✅"):
                st.success(line)
            elif line.startswith("⚠"):
                st.warning(line)
            else:
                st.info(line)
else:
    st.markdown(
        """
        <div style='margin-top:80px; text-align:center; color:#444;'>
            <div style='font-size:3rem;'>🎵</div>
            <div style='font-size:1.2rem; margin-top:12px;'>Waiting for audio or spectrogram image...</div>
            <div style='font-size:0.9rem; color:#666;'>Drag and drop files to start analysis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Footer ──────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#444;font-size:0.8rem;'>"
    "LexKo Suite © 2026 | Professional Spectrum Analysis v2.0"
    "</div>",
    unsafe_allow_html=True,
)
