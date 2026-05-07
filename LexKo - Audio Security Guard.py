import os
import io
import sys
import struct
import hashlib
import numpy as np
import streamlit as st
from scipy.io import wavfile
from mutagen.id3 import ID3, TIT2, TPE1, ID3NoHeaderError
from mutagen.mp3 import MP3

# === 1. 環境變數與路徑偵測 (開發與打包相容版) ===
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# ─── 核心引擎：LSB 隱寫術 ─────────────────────────────────────────────
# (完全保留原始運算邏輯，未做任何修改)

def derive_key_int(key_str: str) -> int:
    """把字串金鑰轉成整數偏位值（0 ~ 2^20 - 1）"""
    sha = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    crc = int(sha[:8], 16)
    return crc & 0xFFFFF

def text_to_bits(text: str) -> list:
    """UTF-8 文字 → bit 列表"""
    raw = text.encode("utf-8")
    bits = []
    for byte in raw:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def bits_to_text(bits: list) -> str:
    """bit 列表 → UTF-8 文字"""
    bytes_list = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            break
        byte_val = 0
        for b in byte_bits:
            byte_val = (byte_val << 1) | b
        bytes_list.append(byte_val)
    return bytes(bytes_list).decode("utf-8", errors="replace")

def embed_message(samples: np.ndarray, key_int: int, message: str) -> np.ndarray:
    """使用跨距 (Stride) LSB 邏輯嵌入訊息"""
    output = samples.copy()
    if output.ndim == 1:
        output = output.reshape(-1, 1)

    msg_bits = text_to_bits(message)
    total_bits = len(msg_bits)
    length_bits = [(total_bits >> (31 - i)) & 1 for i in range(32)]
    all_bits = length_bits + msg_bits

    stride = (key_int % 32) + 64
    start_pos = (key_int % 5000) + 2000

    curr_idx = start_pos
    for bit in all_bits:
        if curr_idx >= len(output):
            raise ValueError("音訊長度不足，無法完成碎片化嵌入！")
        val = int(output[curr_idx, 0])
        output[curr_idx, 0] = np.int16((val & ~1) | bit)
        curr_idx += stride

    return output

def extract_message(samples: np.ndarray, key_int: int) -> str:
    """從 samples 以跳躍式 LSB 方式提取訊息"""
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    stride = (key_int % 32) + 64
    start_pos = (key_int % 5000) + 2000

    length_bits = []
    for i in range(32):
        curr_idx = start_pos + (i * stride)
        if curr_idx >= len(samples):
            raise ValueError("讀取超出範圍，金鑰可能錯誤。")
        bit = int(samples[curr_idx, 0]) & 1
        length_bits.append(bit)

    total_msg_bits = 0
    for b in length_bits:
        total_msg_bits = (total_msg_bits << 1) | b

    if total_msg_bits <= 0 or total_msg_bits > 1000000:
        raise ValueError("讀取到的長度異常，金鑰錯誤或檔案未加密。")

    msg_bits = []
    content_start = start_pos + (32 * stride)
    for i in range(total_msg_bits):
        curr_idx = content_start + (i * stride)
        if curr_idx >= len(samples):
            break
        msg_bits.append(int(samples[curr_idx, 0]) & 1)

    return bits_to_text(msg_bits)

def auto_generate_key(filename: str) -> tuple:
    """以檔名（去掉副檔名）自動產生金鑰字串，回傳 (key, title)"""
    title = os.path.splitext(filename)[0]
    sha = hashlib.sha256(title.encode("utf-8")).hexdigest()[:12]
    key = f"LEXKO-{sha.upper()}"
    return key, title


# === 2. Streamlit 頁面設定 ===

st.set_page_config(
    page_title="LexKo: Audio Security Guard",
    page_icon="🛡️",
    layout="centered"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');

    .stApp {
        background-color: #1A1A1A;
        font-family: 'Inter', sans-serif;
    }

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

    /* 模式標籤頁底線顏色 */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #1ABC9C !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #AAAAAA !important;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .stTabs [aria-selected="true"] {
        color: #1ABC9C !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1A1A1A !important;
        border-bottom: 1px solid #2A2A2A;
    }

    /* 按鈕 */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #333333;
        background-color: #262626;
        color: white;
        font-weight: bold;
        transition: all 0.3s;
        width: 100%;
        height: 3em;
    }
    .stButton > button:hover {
        border-color: #1ABC9C;
        color: #1ABC9C;
        background-color: #1A1A1A;
        box-shadow: 0 0 15px rgba(26, 188, 156, 0.1);
    }

    /* Seal 執行按鈕 */
    .seal-btn > button {
        background-color: #1ABC9C !important;
        border-color: #1ABC9C !important;
        color: #111111 !important;
        font-size: 1.05rem;
        height: 3.4em !important;
    }
    .seal-btn > button:hover {
        background-color: #16A085 !important;
        box-shadow: 0 0 20px rgba(26, 188, 156, 0.25) !important;
        color: #ffffff !important;
    }

    /* Reveal 執行按鈕 */
    .reveal-btn > button {
        background-color: #3498DB !important;
        border-color: #3498DB !important;
        color: #111111 !important;
        font-size: 1.05rem;
        height: 3.4em !important;
    }
    .reveal-btn > button:hover {
        background-color: #2980B9 !important;
        box-shadow: 0 0 20px rgba(52, 152, 219, 0.25) !important;
        color: #ffffff !important;
    }

    /* 輸入框 */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #212121 !important;
        border: 1px solid #333333 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-family: 'Consolas', monospace !important;
    }

    /* 結果文字框 */
    .result-box {
        background-color: #212121;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 16px;
        color: #1ABC9C;
        font-family: 'Consolas', monospace;
        font-size: 0.95rem;
        min-height: 130px;
        white-space: pre-wrap;
        word-break: break-all;
    }

    /* 資訊面板 */
    .info-panel {
        background-color: #212121;
        border: 1px solid #2A2A2A;
        border-left: 3px solid #1ABC9C;
        border-radius: 8px;
        padding: 14px 18px;
        color: #AAAAAA;
        font-size: 0.88rem;
        line-height: 1.7;
    }

    /* 側邊欄 */
    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #222222;
    }
    </style>
    """, unsafe_allow_html=True)


# === 3. Header ===

st.title("LexKo")
st.markdown('<p class="brand-subtitle">Audio Security Guard</p>', unsafe_allow_html=True)


# === 4. 側邊欄：使用說明 ===

with st.sidebar:
    st.header("🛡️ How It Works")
    st.markdown("""
    <div class="info-panel">
    <b style="color:#1ABC9C">LSB Steganography</b><br>
    訊息藏於音訊的最低有效位元（LSB），人耳完全無法察覺。<br><br>
    <b style="color:#1ABC9C">Stride Embedding</b><br>
    採用跨距跳躍式嵌入，雜訊分散於整個音訊，大幅降低可偵測性。<br><br>
    <b style="color:#1ABC9C">Auto Generate Key</b><br>
    以音訊檔名自動生成唯一金鑰，快速方便。
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div class="info-panel">
    <b style="color:#FFFFFF">⚠️  注意事項</b><br>
    • 僅支援 <b style="color:#1ABC9C">WAV PCM 16-bit</b> 格式<br>
    • 請妥善保存金鑰，遺失後無法提取訊息<br>
    • Seal 後請下載 <code>_sealed.wav</code>，原檔不受影響
    </div>
    """, unsafe_allow_html=True)


# === 5. 檔案上傳 ===

uploaded_file = st.file_uploader("Drop WAV file here", type=["wav"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    st.audio(file_bytes, format="audio/wav")

    st.divider()

    # === 6. 金鑰區 ===
    st.markdown("### 🔑 Security Key", help="可以自訂金鑰，亦可使用「Auto Generate」透過檔案名稱來生成。")

    # 直接用 session_state key 綁定 text_input，避免互蓋問題
    if "security_key" not in st.session_state:
        st.session_state["security_key"] = ""

    col_key, col_auto = st.columns([3, 1])

    with col_auto:
        if st.button("🚀 Auto Generate"):
            key, title = auto_generate_key(uploaded_file.name)
            st.session_state["security_key"] = key

    with col_key:
        st.text_input(
            label="key",
            label_visibility="collapsed",
            placeholder="手動輸入金鑰，或點擊右方 Auto Generate Key 自動生成…",
            key="security_key"
        )

    if "auto_key_info" in st.session_state and st.session_state["security_key"]:
        st.caption(f"🔍 Auto Key 來源 — {st.session_state['auto_key_info']}")

    key_value = st.session_state["security_key"]

    st.divider()

    # === 7. 模式切換 Tabs ===
    tab_seal, tab_reveal = st.tabs(["🛡️  SEAL  嵌入", "🔍  REVEAL  提取"])

    # ── Seal Tab ──────────────────────────────────────────────────────
    with tab_seal:
        st.markdown("### 💬 Secret Message")
        message = st.text_area(
            label="secret_msg",
            label_visibility="collapsed",
            placeholder="在此輸入要隱藏的訊息…\n支援中文、英文、符號，任何 UTF-8 文字皆可。",
            height=150,
            key="seal_message"
        )

        st.markdown('<div class="seal-btn">', unsafe_allow_html=True)
        do_seal = st.button("🛡️  Start Sealing  開始嵌入", key="btn_seal")
        st.markdown('</div>', unsafe_allow_html=True)

        if do_seal:
            if not key_value.strip():
                st.error("⚠️  請輸入金鑰或點擊 Auto Generate Key 自動生成！")
            elif not message.strip():
                st.error("⚠️  請輸入要隱藏的訊息！")
            else:
                with st.spinner("⏳  嵌入中，請稍候…"):
                    try:
                        # 讀取 WAV
                        wav_io = io.BytesIO(file_bytes)
                        sample_rate, samples = wavfile.read(wav_io)

                        if samples.dtype != np.int16:
                            st.warning("⚠️  偵測到非 16-bit 格式，強制轉換可能產生輕微雜訊。")
                            samples = (samples / np.max(np.abs(samples)) * 32767).astype(np.int16)

                        key_int = derive_key_int(key_value.strip())
                        new_samples = embed_message(samples, key_int, message.strip())

                        # 輸出到記憶體
                        out_io = io.BytesIO()
                        wavfile.write(out_io, sample_rate, new_samples.astype(np.int16))
                        out_bytes = out_io.getvalue()

                        base_name = os.path.splitext(uploaded_file.name)[0]
                        out_name  = f"{base_name}_sealed.wav"

                        st.success("✅  訊息嵌入成功！請下載已封存的檔案。")

                        col_info, col_dl = st.columns([2, 1])
                        with col_info:
                            st.markdown(f"""
                            <div class="info-panel">
                            📄 輸出檔案：<b style="color:#1ABC9C">{out_name}</b><br>
                            🔑 使用金鑰：<code style="color:#1ABC9C">{key_value.strip()}</code><br>
                            <span style="color:#E74C3C">⚠️ 請妥善保存金鑰，提取時必須輸入相同金鑰。</span>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_dl:
                            st.download_button(
                                label="📥 Download Sealed WAV",
                                data=out_bytes,
                                file_name=out_name,
                                mime="audio/wav",
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"❌  嵌入失敗：{e}")

    # ── Reveal Tab ────────────────────────────────────────────────────
    with tab_reveal:
        st.markdown("### 📋 Extract Hidden Message")
        st.markdown('<div class="reveal-btn">', unsafe_allow_html=True)
        do_reveal = st.button("🔍  Start Revealing  開始提取", key="btn_reveal")
        st.markdown('</div>', unsafe_allow_html=True)

        if do_reveal:
            if not key_value.strip():
                st.error("⚠️  請輸入金鑰或點擊 Auto Generate Key 自動生成！")
            else:
                with st.spinner("⏳  提取中，請稍候…"):
                    try:
                        wav_io = io.BytesIO(file_bytes)
                        sample_rate, samples = wavfile.read(wav_io)

                        if samples.dtype != np.int16:
                            samples = samples.astype(np.int16)

                        key_int = derive_key_int(key_value.strip())
                        extracted = extract_message(samples, key_int)

                        st.success("✅  提取成功！")
                        st.markdown("**Extracted Message 提取結果：**")
                        st.markdown(
                            f'<div class="result-box">{extracted}</div>',
                            unsafe_allow_html=True
                        )

                        # 提供文字下載
                        st.download_button(
                            label="📥 Download as .txt",
                            data=extracted.encode("utf-8"),
                            file_name=f"LexKo_extracted_{os.path.splitext(uploaded_file.name)[0]}.txt",
                            mime="text/plain"
                        )

                    except Exception as e:
                        st.error(f"❌  提取失敗：{e}")

else:
    st.info("Waiting for WAV audio file... Drop it above to unlock Security Guard tools.")


# === Footer ===
st.markdown("---")
st.caption("LexKo Suite © 2026 | Professional Audio Tools")
