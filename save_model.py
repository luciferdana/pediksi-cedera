import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


# Definisikan DummyModel - sama seperti sebelumnya
class DummyModel:
    def predict(self, X):
        return [1 if x[3] > 2 else 0 for x in X]

# Buat instance model
model = DummyModel()

# Buat data contoh untuk fitting scaler
# Data dibuat berdasarkan range yang masuk akal untuk setiap fitur:
# [usia, berat, tinggi, cedera_sebelumnya, intensitas_latihan, waktu_pemulihan]
sample_data = np.array([
    [25, 70, 175, 1, 5, 48],  # Pemain muda, cedera sedikit
    [32, 75, 180, 3, 7, 72],  # Pemain senior, riwayat cedera lebih banyak
    [28, 68, 170, 0, 6, 36],  # Pemain tanpa riwayat cedera
    [35, 80, 185, 4, 8, 96]   # Pemain dengan banyak cedera
])

# Buat dan fit scaler
scaler = StandardScaler()
scaler.fit(sample_data)

# Simpan model dan scaler dalam direktori 'model'
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
    print("Model berhasil disimpan ke model/model.pkl")

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
    print("Scaler berhasil disimpan ke model/scaler.pkl")