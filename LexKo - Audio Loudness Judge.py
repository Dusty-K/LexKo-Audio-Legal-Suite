import os
import sys
import streamlit as st
import numpy as np
import pydub
import pyloudnorm as pyln
import pandas as pd
import io
import subprocess
import re
import tempfile

# === 1. FFmpeg Setup (Portable Version Support) ===
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# Adjusting to the structure: Gear/ffmpeg-8.0.1-essentials_build/bin
ffmpeg_bin_path = os.path.join(base_path, "Gear", "ffmpeg-8.0.1-essentials_build", "bin")

if os.path.exists(ffmpeg_bin_path):
    os.environ["PATH"] = ffmpeg_bin_path + os.pathsep + os.environ.get("PATH", "")
    pydub.AudioSegment.converter = os.path.join(ffmpeg_bin_path, "ffmpeg.exe")
    pydub.AudioSegment.ffprobe = os.path.join(ffmpeg_bin_path, "ffprobe.exe")
    ffmpeg_exe = os.path.join(ffmpeg_bin_path, "ffmpeg.exe")
else:
    ffmpeg_exe = "ffmpeg" # Fallback to system ffmpeg

# === 2. Streamlit Page Config ===
st.set_page_config(
    page_title="LexKo: Audio Loudness Judge",
    page_icon="⚖️",
    layout="centered"
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
    </style>
    """, unsafe_allow_html=True)

# === 3. Analysis Logic ===
@st.cache_data
def analyze_audio(file_bytes, file_name, target_lufs, target_peak):
    try:
        # Load audio using pydub
        audio = pydub.AudioSegment.from_file(io.BytesIO(file_bytes))
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples /= (2**15 if audio.sample_width == 2 else 2**31)
        
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        
        # Calculate Integrated LUFS
        meter = pyln.Meter(audio.frame_rate)
        lufs = meter.integrated_loudness(samples)
        
        # Calculate True Peak using FFmpeg (more accurate)
        tp = audio.max_dBFS # Default fallback
        
        # We need a temporary file for FFmpeg to read
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            cmd = [ffmpeg_exe, "-i", tmp_path, "-af", "ebur128=peak=true", "-f", "null", "-"]
            # Hide console window on Windows
            si = None
            if os.name == 'nt':
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            res = subprocess.run(cmd, capture_output=True, text=True, errors='ignore', startupinfo=si)
            tp_match = re.search(r"True peak:\s+(-?\d+\.\d+)\s+dBFS", res.stderr)
            if tp_match:
                tp = float(tp_match.group(1))
        except Exception as e:
            st.warning(f"FFmpeg analysis failed for {file_name}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        return {
            "name": file_name,
            "lufs": lufs,
            "tp": tp,
            "lufs_pass": lufs <= (target_lufs + 0.3),
            "tp_pass": tp <= target_peak,
            "verdict": "MATCHED" if (lufs <= target_lufs + 0.3 and tp <= target_peak) else "NOT MATCHED"
        }
    except Exception as e:
        return {"name": file_name, "error": str(e)}

# === 4. Header Section ===
st.title("LexKo")
st.markdown('<p class="brand-subtitle">Audio Loudness Judge</p>', unsafe_allow_html=True)

# === 5. Sidebar Configuration ===
with st.sidebar:
    st.header("⚖️ Standards")
    target_lufs = st.selectbox("Target LUFS", [-10.0, -12.0, -14.0, -16.0, -18.0, -23.0, -24.0], index=2)
    target_peak = st.selectbox("True Peak (dBTP)", [0.0, -0.5, -1.0, -2.0], index=2)
    
    st.divider()
    
    with st.expander("📚 Knowledge Base"):
        st.markdown("""
        **Integrated LUFS**
        Total perceived loudness over the track duration.
        - Spotify/YouTube: -14 LUFS
        - Apple Music: -16 LUFS
        - Broadcast: -23/-24 LUFS
        
        **True Peak (TP)**
        Absolute analog peak. Ceiling of -1.0 dBTP is recommended to prevent clipping.
        """)
    
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.rerun()

# === 6. Main UI ===
uploaded_files = st.file_uploader("Drop audio files here", type=["wav", "mp3"], accept_multiple_files=True)

results_list = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.status(f"Analyzing {uploaded_file.name}...", expanded=True) as status:
            file_bytes = uploaded_file.read()
            res = analyze_audio(file_bytes, uploaded_file.name, target_lufs, target_peak)
            
            if "error" in res:
                st.error(f"Error analyzing {uploaded_file.name}: {res['error']}")
                continue
            
            results_list.append(res)
            
            # --- Results Display ---
            col1, col2 = st.columns(2)
            
            l_diff = res['lufs'] - target_lufs
            p_diff = res['tp'] - target_peak
            
            col1.metric(
                label="Integrated LUFS", 
                value=f"{res['lufs']:.2f}", 
                delta=f"{l_diff:.2f} LU", 
                delta_color="inverse" if res['lufs'] > target_lufs + 0.3 else "normal"
            )
            
            col2.metric(
                label="True Peak", 
                value=f"{res['tp']:.2f} dBTP", 
                delta=f"{p_diff:.2f} dB", 
                delta_color="inverse" if res['tp'] > target_peak else "normal"
            )
            
            st.audio(file_bytes)
            
            if res['verdict'] == "MATCHED":
                st.success(f"✅ Verdict: **MATCHED** - Ready for Distribution")
            else:
                st.error(f"❌ Verdict: **NOT MATCHED** - Adjustment Recommended")
            
            status.update(label=f"Analysis Complete: {uploaded_file.name}", state="complete", expanded=False)

    # === 7. Export Section ===
    if results_list:
        st.divider()
        df = pd.DataFrame(results_list)
        # Rename for export
        export_df = df[['name', 'lufs', 'tp', 'verdict']].rename(columns={
            'name': 'Filename',
            'lufs': 'Integrated LUFS',
            'tp': 'True Peak (dBTP)',
            'verdict': 'Verdict'
        })
        
        csv = export_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Export Report (CSV)",
            data=csv,
            file_name=f"LexKo_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
else:
    st.info("Waiting for audio files... (WAV/MP3 supported)")

# Footer
st.markdown("---")
st.caption("LexKo Suite © 2026 | Professional Audio Tools")