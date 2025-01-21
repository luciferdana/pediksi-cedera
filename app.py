import streamlit as st
import pickle
import pandas as pd
import os

# Definisi model
class DummyModel:
    def predict(self, X):
        return [1 if x[3] > 2 else 0 for x in X]

# Konfigurasi halaman dengan tema gelap
st.set_page_config(
    page_title="Prediksi Cedera Pemain",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk tema gelap dan desain modern
st.markdown("""
    <style>
    /* Tema Gelap & Styling Global */
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #0061ff 0%, #60efff 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
    }
    
    /* Form Container */
    .form-container {
        background-color: #2d2d2d;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Input Fields */
    .stNumberInput input {
        background-color: #3d3d3d;
        border: 1px solid #4d4d4d;
        color: white;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #0061ff 0%, #60efff 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Hasil Prediksi */
    .prediction-result {
        background: #2d2d2d;
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-result h2 {
        color: #60efff;
        margin-bottom: 1rem;
    }
    
    .result-berisiko {
        color: #ff4d4d;
        font-weight: bold;
    }
    
    .result-aman {
        color: #4dff4d;
        font-weight: bold;
    }
    
    /* Label Styling */
    .input-label {
        color: #60efff;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1.5rem;
        }
        .main-header h1 {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>Prediksi Cedera Pemain</h1>
        <p>Masukkan data pemain untuk memprediksi risiko cedera.</p>
    </div>
    """, unsafe_allow_html=True)

# Fungsi untuk memuat model
@st.cache_resource
def load_model_and_scaler():
    try:
        model_path = "model/model.pkl"
        scaler_path = "model/scaler.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error("File model atau scaler tidak ditemukan. Jalankan save_model.py terlebih dahulu.")
            st.stop()
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error saat memuat model: {str(e)}")
        st.stop()

# Form dalam container dengan styling
with st.container():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="input-label">Usia Pemain:</p>', unsafe_allow_html=True)
        player_age = st.number_input(
            "",
            min_value=15,
            max_value=45,
            value=34,
            key="age"
        )
        
        st.markdown('<p class="input-label">Berat Pemain (kg):</p>', unsafe_allow_html=True)
        player_weight = st.number_input(
            "",
            min_value=50,
            max_value=120,
            value=81,
            key="weight"
        )
        
        st.markdown('<p class="input-label">Tinggi Pemain (cm):</p>', unsafe_allow_html=True)
        player_height = st.number_input(
            "",
            min_value=150,
            max_value=220,
            value=192,
            key="height"
        )
    
    with col2:
        st.markdown('<p class="input-label">Jumlah Cedera Sebelumnya:</p>', unsafe_allow_html=True)
        previous_injuries = st.number_input(
            "",
            min_value=0,
            max_value=20,
            value=12,
            key="injuries"
        )
        
        st.markdown('<p class="input-label">Intensitas Latihan (1-10):</p>', unsafe_allow_html=True)
        training_intensity = st.number_input(
            "",
            min_value=1,
            max_value=10,
            value=10,
            key="intensity"
        )
        
        st.markdown('<p class="input-label">Waktu Pemulihan (jam):</p>', unsafe_allow_html=True)
        recovery_time = st.number_input(
            "",
            min_value=24,
            max_value=168,
            value=78,
            key="recovery"
        )
    
    # Tombol prediksi dengan styling
    st.markdown('<div style="text-align: center; margin-top: 2rem;">', unsafe_allow_html=True)
    predict_button = st.button("Prediksi Cedera")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Proses prediksi
if predict_button:
    try:
        model, scaler = load_model_and_scaler()
        
        input_data = {
            'Player_Age': [player_age],
            'Player_Weight': [player_weight],
            'Player_Height': [player_height],
            'Previous_Injuries': [previous_injuries],
            'Training_Intensity': [training_intensity],
            'Recovery_Time': [recovery_time]
        }
        input_df = pd.DataFrame(input_data)
        
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        
        result = "Berisiko Cedera" if prediction[0] == 1 else "Tidak Berisiko Cedera"
        result_class = "result-berisiko" if prediction[0] == 1 else "result-aman"
        
        st.markdown(f"""
            <div class="prediction-result">
                <h2>Hasil Prediksi</h2>
                <p>Status: <span class="{result_class}">{result}</span></p>
                <p style="margin-top: 1rem; color: #999;">
                    Prediksi ini berdasarkan analisis dari data yang dimasukkan
                </p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam proses prediksi: {str(e)}")