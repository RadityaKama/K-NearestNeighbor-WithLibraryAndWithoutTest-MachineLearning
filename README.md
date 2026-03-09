# Klasifikasi K-Nearest Neighbors (K-NN): Library vs Non-Library

Proyek ini berisi implementasi dan perbandingan evaluasi algoritma pembelajaran mesin K-Nearest Neighbors (K-NN) untuk memprediksi performa siswa. Pengujian dilakukan dengan dua pendekatan komparatif: menggunakan library `scikit-learn` dan implementasi logika manual (non-library) dari awal menggunakan Python.

## Dataset
Dataset yang digunakan dalam pengujian ini adalah `StudentPerformanceFactors.csv`. Dataset ini merangkum berbagai elemen yang memengaruhi kinerja akademis siswa.
- [cite_start]**Fitur/Variabel Independen**: Variabel yang dianalisis meliputi waktu belajar, kehadiran, keterlibatan orang tua, jam tidur, skor ujian sebelumnya, hingga kualitas guru[cite: 1968].
- **Target Variabel**: Kolom `Exam_Score` dimodifikasi menjadi target klasifikasi biner. Nilai yang lebih besar atau sama dengan nilai median diklasifikasikan sebagai `1` (memenuhi standar/tinggi), sedangkan nilai di bawah median diklasifikasikan sebagai `0`.

## Persiapan Data (Data Preprocessing)
Sebelum data dilatih menggunakan K-NN, serangkaian pra-pemrosesan diterapkan pada dataset:
1. **Pembersihan Data**: Mengeliminasi baris yang tidak lengkap atau memiliki nilai kosong (`dropna`).
2. **Label Encoding**: Mengubah variabel kategorikal (bertipe string/objek) menjadi representasi numerik menggunakan `LabelEncoder`.
3. **Pembagian Data (Train-Test Split)**: Membagi dataset menjadi data latih (training) dan data uji (testing) dengan porsi pembagian 60:40.
4. **Standardisasi Fitur**: Menyamakan skala semua fitur menggunakan `StandardScaler` (untuk library) dan perhitungan manual `(X - mean) / std` (untuk non-library) guna mencegah jarak kalkulasi didominasi oleh variabel dengan rentang nilai besar.

## Implementasi K-NN
Pengujian akurasi dievaluasi menggunakan berbagai skenario nilai K: `[4, 7, 8, 11, 12, 21, 24, 25, 27, 29]`. Terdapat dua blok pemrosesan utama:

### 1. K-NN Manual (Non-Library)
Algoritma K-NN dibangun tanpa fungsi instan model pembelajaran mesin, memanfaatkan komputasi matriks `numpy`.
- **Fungsi `hitung_jarak_euclidean`**: Mengkalkulasi metrik jarak lurus (Euclidean) antara himpunan titik latih dan titik data tes.
- **Fungsi `prediksi_knn`**: Menyortir jarak yang ditemukan untuk mencari sebanyak *k* tetangga terdekat, mengekstrak label mereka, lalu menentukan kelas prediksi berdasarkan hasil voting mayoritas (*majority vote*).

### 2. K-NN menggunakan Scikit-Learn
Sebagai penanda tolak ukur (*baseline*), skrip mengimplementasikan algoritma yang setara menggunakan modul bawaan `KNeighborsClassifier` dari `sklearn.neighbors`.

## Cara Menjalankan Skrip
Pastikan Anda sudah menginstal beberapa modul dependencies berikut di environment Python Anda:
```bash
pip install pandas numpy scikit-learn joblib
