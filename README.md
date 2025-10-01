# Proyek Detektor Gestur Tangan (Gunting-Batu-Kertas)

Ini adalah proyek aplikasi web Machine Learning interaktif yang mampu mendeteksi dan memprediksi gestur tangan (Gunting, Batu, Kertas) secara real-time melalui kamera webcam. Proyek ini dibangun sebagai bagian dari tugas Machine Learning dan di-deploy agar dapat diakses secara online.

## Fitur Utama

- **Deteksi Real-Time:** Menggunakan MediaPipe untuk _hand tracking_ dan menampilkan kerangka tangan secara langsung di atas feed kamera.
- **Prediksi Otomatis:** Model akan secara otomatis memprediksi gestur setiap beberapa detik tanpa perlu klik manual.
- **Dasbor Modern:** Antarmuka pengguna yang modern dan responsif dibangun menggunakan Tailwind CSS, menampilkan hasil prediksi dan _chart_ probabilitas.
- **Model Machine Learning:** Menggunakan model Convolutional Neural Network (CNN) yang dilatih dengan TensorFlow/Keras pada dataset Gunting-Batu-Kertas.
- **Backend API:** Server Flask (Python) yang siap untuk produksi menggunakan Gunicorn.
- **Tombol On/Off Kamera:** Fungsionalitas untuk mengaktifkan dan menonaktifkan kamera.

---

## Teknologi yang Digunakan

- **Backend:**
  - Python
  - Flask (sebagai framework API)
  - TensorFlow / Keras (untuk membangun & melatih model)
  - Gunicorn (sebagai server produksi WSGI)
- **Frontend:**
  - HTML
  - Tailwind CSS (untuk styling)
  - JavaScript (untuk logika, interaktivitas, dan memanggil API)
  - MediaPipe.js (untuk hand tracking)
  - Chart.js (untuk visualisasi data probabilitas)
- **Deployment:**
  - Git & GitHub
  - Render.com

---

## Struktur File Proyek

```
Proyek_GuntingBatuKertas/
├── dataset/
│   └── train/
│       ├── rock/
│       ├── paper/
│       └── scissors/
├── models/
│   └── model_rps.h5
├── scripts/
│   └── latih_rps.py
├── static/
│   └── script.js
├── .gitignore
├── app.py
├── index.html
├── README.md
└── requirements.txt
```

---

## Cara Menjalankan Secara Lokal

1.  **Clone Repository**

    ```bash
    git clone
    cd Proyek_GuntingBatuKertas
    ```

2.  **Install Dependencies**
    Pastikan Anda memiliki Python. Lalu install semua library yang dibutuhkan.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan Server Backend**
    Buka terminal dan jalankan server Flask.

    ```bash
    gunicorn app:app
    ```

    Atau untuk development:

    ```bash
    python app.py
    ```

4.  **Buka Frontend**
    Buka file `index.html` langsung di browser Anda (misalnya Google Chrome).

---

## Catatan Deployment

Proyek ini di-deploy di Render.com dengan konfigurasi berikut:

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`
"# gunting_batu_kertas" 
