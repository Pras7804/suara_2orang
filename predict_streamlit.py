import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import joblib
import librosa
import tsfel
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

# ===============================
# Load Model dan Konfigurasi
# ===============================
model_command = joblib.load("model_command.pkl")
model_speaker = joblib.load("model_speaker.pkl")
scaler = joblib.load("scaler.pkl")
selector_cmd = joblib.load("selector_cmd.pkl")
selector_spk = joblib.load("selector_spk.pkl")
feature_cols = joblib.load("feature_columns.pkl")
cfg = tsfel.get_features_by_domain()

st.set_page_config(page_title="Voice Command & Speaker Recognition", page_icon="🎤")
st.title("🎤 Voice Command & Speaker Recognition")

# ===============================
# Inisialisasi Session State
# ===============================
if "recorded_file" not in st.session_state:
    st.session_state.recorded_file = None
if "pred_ready" not in st.session_state:
    st.session_state.pred_ready = False

# ===============================
# Fungsi Ekstraksi Fitur
# ===============================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = tsfel.time_series_features_extractor(cfg, y, fs=sr).fillna(0)
    features = features.reindex(columns=feature_cols, fill_value=0)

    # Normalisasi (pakai scaler yang sama saat training)
    X_scaled = scaler.transform(features)

    return X_scaled

# ===============================
# Rekam Suara Langsung
# ===============================
st.subheader("🎙️ Rekam suara langsung")
duration = st.slider("Durasi rekaman (detik)", 1, 5, 3)

if st.button("▶️ Mulai Rekam"):
    fs = 16000
    st.info("🎧 Merekam...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    wav.write("input.wav", fs, recording)
    st.session_state.recorded_file = "input.wav"
    st.session_state.pred_ready = True
    st.success("✅ Rekaman selesai!")

    with open("input.wav", "rb") as f:
        st.audio(f.read(), format="audio/wav")

# ===============================
# Upload File Suara
# ===============================
st.subheader("📂 Atau upload file suara (.wav)")
uploaded_file = st.file_uploader("Pilih file .wav", type=["wav"])

if uploaded_file is not None:
    with open("uploaded.wav", "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.recorded_file = "uploaded.wav"
    st.session_state.pred_ready = True
    st.success("✅ File berhasil diupload!")

    st.audio(uploaded_file, format="audio/wav")

# ===============================
# Prediksi Speaker & Command
# ===============================
if st.session_state.pred_ready:
    if st.button("🔍 Prediksi"):
        file_path = st.session_state.recorded_file
        st.info("🔎 Mengekstrak fitur dan memproses prediksi...")

        # Ekstraksi dan scaling fitur
        X_scaled = extract_features(file_path)

        # Seleksi fitur
        X_cmd = selector_cmd.transform(X_scaled)
        X_spk = selector_spk.transform(X_scaled)

        # Prediksi Speaker
        proba_spk = model_speaker.predict_proba(X_spk)[0]
        confidence_spk = np.max(proba_spk)
        speaker_pred = model_speaker.classes_[np.argmax(proba_spk)]

        # Threshold untuk unknown
        threshold = 0.6
        if confidence_spk < threshold:
            st.error(f"❌ Suara tidak dikenali (unknown speaker). [confidence={confidence_spk:.2f}]")
        else:
            # Prediksi Command
            command_pred = model_command.predict(X_cmd)[0]
            st.success(f"🗣️ Speaker: {speaker_pred} (confidence={confidence_spk:.2f})")
            st.info(f"🎧 Command: {command_pred}")

# ===============================
# Tombol Reset
# ===============================
if st.button("🔁 Rekam / Upload Ulang"):
    st.session_state.recorded_file = None
    st.session_state.pred_ready = False
    st.success("🔄 Siap untuk rekam atau upload ulang.")
