import os
import io
import struct
import sys
import streamlit as st
import tempfile
from PIL import Image
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC, TCON, APIC

# === 1. 核心 WAV 處理引擎 (採用 Original 版的高相容性邏輯) ===

def _read_wav_chunks(data: bytes):
    riff_pos = data.find(b'RIFF')
    if riff_pos == -1 or data[riff_pos+8:riff_pos+12] != b'WAVE':
        raise ValueError("Not a valid WAV file")
    
    chunks = []
    pos = riff_pos + 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos+4]
        try:
            chunk_size = struct.unpack_from('<I', data, pos+4)[0]
        except: break
        
        actual_size = min(chunk_size, len(data) - (pos + 8))
        chunk_data = data[pos+8: pos+8+actual_size]
        
        # 過濾舊的 id3 與 LIST 區塊，避免重複堆疊
        clean_id = chunk_id.lower().rstrip(b'\x00').strip()
        if clean_id not in [b'id3', b'list']:
            chunks.append((chunk_id, chunk_data))
        pos += 8 + actual_size + (actual_size % 2)
    return chunks

def _build_wav_full(chunks, id3_bytes, list_info_bytes) -> bytes:
    """完美融合：LIST INFO (給 Windows) + ID3 (給專業軟體)

    重要：結構順序必須與原版一致 —— LIST INFO 必須最先出現，
    Windows Explorer 才能正確解析 Property Detail。
    fmt / data 等原始區塊依序附後即可，Windows 會自行掃描找到它們。
    """
    body = b''

    # 1. 優先放 LIST INFO，這是 Windows 詳細資料的唯一來源
    if list_info_bytes:
        body += list_info_bytes

    # 2. 接著放 ID3 (包含封面，供專業音樂軟體讀取)
    if id3_bytes:
        body += b'id3 '
        body += struct.pack('<I', len(id3_bytes))
        body += id3_bytes + (b'\x00' if len(id3_bytes) % 2 else b'')

    # 3. 最後放入所有原始區塊 (fmt、data 等，保留原始順序)
    for cid, cdata in chunks:
        body += cid + struct.pack('<I', len(cdata)) + cdata
        if len(cdata) % 2: body += b'\x00'

    riff_size = 4 + len(body)
    return b'RIFF' + struct.pack('<I', riff_size) + b'WAVE' + body

def _build_list_info(title, artist, album, year, genre) -> bytes:
    """產生 Windows 專用的 LIST INFO 區塊，採用 cp950 編碼"""
    info_map = {
        b'INAM': title,
        b'IART': artist,
        b'IPRD': album,
        b'ICRD': year,
        b'IGNR': genre
    }
    sub = b''
    for cid, val in info_map.items():
        if not val or not val.strip(): continue
        # 確保使用 Windows 繁體中文預設編碼
        enc = val.encode('cp950', errors='replace') + b'\x00'
        sub += cid + struct.pack('<I', len(enc)) + enc
        if len(enc) % 2: sub += b'\x00'
    
    if not sub: return b''
    
    body = b'INFO' + sub
    return b'LIST' + struct.pack('<I', len(body)) + body

# === 2. Streamlit 介面與品牌設定 ===

st.set_page_config(page_title="LexKo: RMI Administrator", page_icon="📁", layout="centered")

# --- 核心視覺修正：精品極簡風 ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    
    /* 背景與基礎字體 */
    .stApp {
        background-color: #1A1A1A;
        font-family: 'Inter', sans-serif;
    }
    
    /* 強力注入標題樣式 */
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
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #333333;
        background-color: #262626;
        color: white;
        font-weight: bold;
        transition: all 0.3s;
        width: 100%;
        height: 3em;
    }
    .stButton>button:hover {
        border-color: #1ABC9C;
        color: #1ABC9C;
        background-color: #1A1A1A;
        box-shadow: 0 0 15px rgba(26, 188, 156, 0.1);
    }
    
    /* 側邊欄樣式 (如有需要) */
    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #222222;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("LexKo")
st.markdown('<p class="brand-subtitle">Audio RMI Administrator</p>', unsafe_allow_html=True)

# === 3. 主程式 UI ===

uploaded_file = st.file_uploader("Drop MP3 / WAV here", type=["mp3", "wav"])

if uploaded_file:
    # 自動抓取檔名作為預設標題
    default_title = os.path.splitext(uploaded_file.name)[0]
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 📝 Metadata Editor")
        title = st.text_input("Title (單曲名稱)", value=default_title)
        artist = st.text_input("Artist (演出者)")
        album = st.text_input("Album (專輯名稱)")
        year = st.text_input("Year (年份)")
        genre = st.text_input("Genre (類型)")
        
    with col2:
        st.markdown("### 🖼️ Cover Art")
        cover_file = st.file_uploader("Upload Image", type=["jpg", "png"])
        if cover_file:
            st.image(cover_file, caption="Selected Cover", use_container_width=True)
        else:
            st.info("No cover selected.")

    st.divider()

    if st.button("🛡️ Apply Security Guard & Save"):
        try:
            # 建立 ID3 標籤
            tags = ID3()
            tags.add(TIT2(encoding=3, text=[title]))
            tags.add(TPE1(encoding=3, text=[artist]))
            tags.add(TALB(encoding=3, text=[album]))
            tags.add(TDRC(encoding=3, text=[year]))
            tags.add(TCON(encoding=3, text=[genre]))
            
            if cover_file:
                mime = 'image/png' if cover_file.name.lower().endswith('.png') else 'image/jpeg'
                tags.add(APIC(encoding=3, mime=mime, type=3, desc='Cover', data=cover_file.getvalue()))

            # 處理檔案
            file_bytes = uploaded_file.getvalue()
            output_io = io.BytesIO()

            if uploaded_file.name.lower().endswith('.mp3'):
                # MP3 邏輯 (暫存處理)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                
                tags.save(tmp_path, v2_version=3)
                with open(tmp_path, 'rb') as f:
                    output_io.write(f.read())
                os.remove(tmp_path)
            else:
                # WAV 邏輯 (直接操作 RIFF)
                chunks = _read_wav_chunks(file_bytes)
                buf = io.BytesIO()
                tags.save(buf, v2_version=3)
                id3_bytes = buf.getvalue()
                list_bytes = _build_list_info(title, artist, album, year, genre)
                new_wav = _build_wav_full(chunks, id3_bytes, list_bytes)
                output_io.write(new_wav)

            st.success("✅ Universal Compatibility Metadata Applied!")
            st.download_button(
                label="📥 Download Processed File",
                data=output_io.getvalue(),
                file_name=f"LexKo_{uploaded_file.name}",
                mime="audio/mpeg" if uploaded_file.name.lower().endswith('.mp3') else "audio/wav"
            )
        except Exception as e:
            st.error(f"Processing Failed: {e}")

else:
    st.info("Waiting for audio file to unlock administrator tools...")

# Footer
st.markdown("---")
st.caption("LexKo Suite © 2026 | Professional Audio Tools")

