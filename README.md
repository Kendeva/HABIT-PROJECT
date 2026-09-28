# HABIT — Smart Home Living Assistant

HABIT adalah aplikasi berbasis AI untuk membantu pengguna menganalisis kondisi visual rumah, mendeteksi masalah maintenance pada permukaan bangunan, dan memberikan estimasi harga properti berdasarkan data perumahan Indonesia.

> Project Computer Science — BINUS University Semester 5

---

## Fitur

- **Home Scan** — menganalisis kondisi visual beberapa area rumah
- **Maintenance Detection** — mendeteksi masalah visual seperti crack, peeling, stain, algae, dan spalling
- **Property Estimate** — memberikan estimasi harga properti berdasarkan data perumahan Indonesia
- **Home Report** — merangkum hasil Home Scan, Maintenance, dan Property Estimate

---

## Pipeline

```text
Home Images
    ↓
Pretrained ViT
    ↓
Home Condition

Maintenance Image
    ↓
Custom CNN
    ↓
Maintenance Result

Property Information
    ↓
Regression Model
    ↓
Estimated Price + Estimated Range
```

---

## AI / ML Model

| Fitur | Model | Status |
|---|---|---|
| Home Scan | Pretrained ViT | Pretrained |
| Maintenance Detection | Custom CNN | Trained for HABIT |
| Property Estimate | HistGradientBoostingRegressor | Trained for HABIT |

HABIT menggabungkan **Deep Learning** dan **Machine Learning**. Home Scan menggunakan model vision yang sudah pretrained, sedangkan Custom CNN dan Regression Model dilatih khusus untuk project ini.

---

## Dataset

Dataset tidak disimpan langsung di repository. Dataset akan diambil melalui Hugging Face saat proses training.

| Dataset | Digunakan untuk | Sumber |
|---|---|---|
| `chandrabhuma/building_defect_vqa` | Maintenance Detection | Hugging Face |
| `web3hungry/indonesia-affordable-housing` | Property Estimate | Hugging Face |
| `DejanX13/vit-house-classifier` | Home Condition | Hugging Face pretrained model |

Untuk Property Estimate, proses training menggunakan maksimal **5.000 data yang sudah melalui cleaning** agar tetap realistis untuk project mahasiswa dan lebih ringan dijalankan secara lokal.

---

## Struktur Project

```text
HABIT/
├── app.py
├── model.py
├── options.py
├── train_maintenance.py
├── train_property.py
├── requirements.txt
├── README.md
├── assets/
│   └── style.css
└── models/
    ├── maintenance_cnn.pth
    └── property_model.joblib
```

---

## Cara Menjalankan

### 1. Buat virtual environment

```bash
python -m venv .venv
```

### 2. Aktifkan virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Training model

Maintenance:

```bash
python train_maintenance.py
```

Property Estimate:

```bash
python train_property.py
```

Home Scan tidak perlu ditrain karena menggunakan pretrained model.

### 5. Jalankan aplikasi

```bash
streamlit run app.py
```

Jika model sudah pernah ditrain dan file model masih ada di folder `models/`, cukup jalankan:

```bash
streamlit run app.py
```

---

## Author

**Keanu**

---

## Limitations

- **Home Condition belum spesifik untuk rumah Indonesia** — Home Scan masih menggunakan pretrained model umum, sehingga karakteristik rumah Indonesia belum sepenuhnya terwakili.
- **Maintenance masih bergantung pada kualitas gambar** — pencahayaan, sudut pengambilan gambar, jarak, dan kondisi permukaan dapat memengaruhi hasil prediksi.
- **Property Estimate masih menggunakan fitur yang terbatas** — model belum mempertimbangkan semua faktor yang memengaruhi harga properti seperti akses jalan, fasilitas sekitar, usia bangunan, dan kondisi pasar secara langsung.
- **Estimated Range masih memerlukan perbaikan** — rentang harga sudah menggunakan calibration dari validation residual, tetapi hasilnya masih belum selalu cukup dekat dengan harga aktual pada semua kasus.
- **Dataset property dibatasi hingga 5.000 data** — pembatasan ini membuat training lebih ringan, tetapi juga dapat mengurangi representasi variasi properti Indonesia.
- **Belum untuk penggunaan profesional** — hasil HABIT masih berupa AI-assisted reference dan tidak menggantikan inspeksi bangunan atau appraisal properti profesional.

---

## Future Work

- Meningkatkan **Property Estimate** dengan feature engineering dan model regression yang lebih akurat.
- Memperbaiki **Estimated Range** agar lebih stabil dan lebih dekat dengan harga aktual.
- Menambah dan mengevaluasi dataset visual rumah Indonesia untuk Home Scan.
- Meningkatkan generalisasi Maintenance Detection pada kondisi gambar dan jenis bangunan yang lebih beragam.
- Melakukan evaluasi model yang lebih lengkap menggunakan test data yang lebih representatif.
- Mengoptimalkan inference agar aplikasi lebih cepat pada perangkat tanpa GPU.

---

## Status

HABIT sudah dapat menjalankan seluruh alur utama project, tetapi beberapa bagian masih membutuhkan pengembangan lebih lanjut, terutama pada **akurasi Property Estimate, Estimated Range, dan generalisasi model visual**.
