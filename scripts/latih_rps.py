# -*- coding: utf-8 -*-
"""
Deskripsi:
    Skrip ini bertujuan untuk melatih sebuah model klasifikasi gambar
    menggunakan Convolutional Neural Network (CNN) dengan metode Transfer Learning.
    Model dilatih untuk mengenali tiga kelas gestur tangan: batu, gunting, dan kertas.

Arsitektur:
    - Base Model: MobileNetV2 (pre-trained on ImageNet)
    - Classifier Head: GlobalAveragePooling2D -> Dense(512) -> Dropout(0.5) -> Dense(3, softmax)

Dependencies:
    - tensorflow
"""

# =====================================================================================
# 1. IMPORT LIBRARY
# =====================================================================================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# =====================================================================================
# 2. KONFIGURASI PARAMETER DAN PATH
# =====================================================================================
# Direktori root yang berisi dataset training.
train_dir = 'dataset/train'

# Ukuran gambar input yang akan digunakan oleh model (height, width).
IMG_SIZE = 150

# Jumlah sampel gambar yang diproses dalam satu iterasi (batch).
BATCH_SIZE = 32

# =====================================================================================
# 3. PREPROCESSING DAN AUGMENTASI DATA
# =====================================================================================
# Inisialisasi ImageDataGenerator untuk augmentasi data dan normalisasi.
# Augmentasi digunakan untuk memperkaya variasi data training secara artifisial,
# yang membantu model untuk generalisasi lebih baik dan mengurangi overfitting.
datagen = ImageDataGenerator(
    rescale=1./255,                 # Normalisasi nilai piksel ke rentang [0, 1]
    rotation_range=30,              # Rotasi acak gambar
    width_shift_range=0.2,          # Pergeseran horizontal acak
    height_shift_range=0.2,         # Pergeseran vertikal acak
    shear_range=0.2,                # Transformasi shear acak
    zoom_range=0.2,                 # Zoom acak
    horizontal_flip=True,           # Flip horizontal acak
    fill_mode='nearest',            # Metode pengisian piksel baru
    validation_split=0.2            # Porsi data yang akan digunakan untuk validasi (20%)
)

# =====================================================================================
# 4. PEMBUATAN DATA GENERATOR
# =====================================================================================
# Membuat data generator untuk set training (80% dari data).
# Generator akan memuat gambar dari direktori secara batch dan menerapkan augmentasi.
train_generator = datagen.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',              # Menandakan ini sebagai set training
    color_mode='rgb'
)

# Membuat data generator untuk set validasi (20% dari data).
validation_generator = datagen.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',            # Menandakan ini sebagai set validasi
    color_mode='rgb'
)

# =====================================================================================
# 5. PEMBANGUNAN ARSITEKTUR MODEL (TRANSFER LEARNING)
# =====================================================================================
# Memuat base model MobileNetV2 yang telah dilatih pada dataset ImageNet.
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,  # Tidak menyertakan lapisan klasifikasi (fully-connected) teratas
    weights='imagenet'
)

# Membekukan bobot dari base model agar tidak diperbarui selama proses training.
# Hal ini dilakukan untuk mempertahankan fitur-fitur yang telah dipelajari.
base_model.trainable = False

# Menambahkan lapisan kustom (classifier head) di atas base model.
# Lapisan ini akan dilatih secara spesifik untuk tugas klasifikasi gestur tangan.
x = base_model.output
x = GlobalAveragePooling2D()(x)     # Meratakan fitur spasial menjadi vektor tunggal
x = Dense(512, activation='relu')(x) # Lapisan fully-connected untuk pembelajaran fitur tingkat tinggi
x = Dropout(0.5)(x)                 # Lapisan Dropout untuk regularisasi dan mencegah overfitting
predictions = Dense(3, activation='softmax')(x) # Lapisan output dengan 3 neuron (jumlah kelas)

# Menggabungkan base model dengan classifier head menjadi model akhir.
model = Model(inputs=base_model.input, outputs=predictions)

# Meng-compile model, mendefinisikan optimizer, fungsi loss, dan metrik evaluasi.
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Menampilkan ringkasan arsitektur model.
model.summary()

# =====================================================================================
# 6. KONFIGURASI CALLBACKS UNTUK TRAINING
# =====================================================================================
# Callbacks adalah utilitas yang dipanggil pada berbagai titik selama proses training
# untuk melakukan tindakan tertentu seperti menyimpan model atau menghentikan training.

# ModelCheckpoint: Menyimpan bobot model terbaik berdasarkan `val_accuracy`.
model_checkpoint = ModelCheckpoint(
    filepath='models/best_model_rps.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# EarlyStopping: Menghentikan training jika tidak ada peningkatan pada `val_accuracy`
# setelah sejumlah epoch (patience).
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# ReduceLROnPlateau: Mengurangi learning rate jika `val_accuracy` tidak meningkat (stagnan).
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# =====================================================================================
# 7. PROSES TRAINING MODEL
# =====================================================================================
# Jumlah maksimum epoch yang akan dijalankan.
epochs = 25

# Memulai proses training model menggunakan data generator dan callbacks.
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[model_checkpoint, early_stopping, reduce_lr]
)

print("\nPelatihan selesai. Model terbaik telah disimpan di 'models/best_model_rps.keras'")