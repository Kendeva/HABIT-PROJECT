# HABIT — Smart Home Living Assistant

HABIT is an AI-based application designed to help users understand a home's visual condition, identify visible maintenance issues, and estimate property prices using Indonesian housing data.

> Computer Science Project — BINUS University, Semester 5

LIVE APP : https://habit-project.streamlit.app/

---

## Overview

HABIT combines Deep Learning and Machine Learning in one simple Streamlit application.

The project focuses on three main tasks:

1. **Home Condition Analysis**
2. **Maintenance Detection**
3. **Indonesian Property Price Estimation**

The final results can also be viewed together through the **Home Report** page.

---

## Features

### Home Scan
Users can upload several home images and select the area manually.

HABIT analyzes the visual condition of each image and returns:

- overall home condition,
- condition for each uploaded area,
- prediction confidence,
- short result details.

### Maintenance Detection
Users can upload an image of a wall, ceiling, or building surface.

The model detects visible maintenance-related conditions such as:

- algae,
- major crack,
- minor crack,
- peeling,
- spalling,
- stain,
- no clear defect.

### Property Estimate
Users provide basic property information such as:

- province,
- housing category,
- land area,
- building area,
- bedrooms,
- bathrooms,
- floors,
- property type.

The model returns:

- estimated property price,
- estimated price range.

### Home Report
The Home Report summarizes the latest results from:

- Home Scan,
- Maintenance Detection,
- Property Estimate.

---

## System Flow

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

## AI / ML Models

| Feature | Model | Type |
|---|---|---|
| Home Scan | Pretrained ViT | Deep Learning |
| Maintenance Detection | Custom CNN | Deep Learning |
| Property Estimate | HistGradientBoostingRegressor | Machine Learning |

### Pretrained Model

The **Home Scan** uses:

```text
DejanX13/vit-house-classifier
```

This model is already pretrained and does not need to be trained locally.

### Models Trained for HABIT

The following models are trained specifically for this project:

```text
Custom CNN
→ Maintenance Detection
```

```text
HistGradientBoostingRegressor
→ Property Price Estimation
```

---

## Datasets

The datasets are not stored directly inside the repository. They are downloaded through Hugging Face during training.

| Dataset | Purpose | Source |
|---|---|---|
| `chandrabhuma/building_defect_vqa` | Maintenance Detection | Hugging Face |
| `web3hungry/indonesia-affordable-housing` | Property Estimate | Hugging Face |
| `DejanX13/vit-house-classifier` | Home Condition | Hugging Face pretrained model |

For the Property Estimate model, HABIT uses a maximum of **5,000 cleaned records** so the training process remains practical for a student project and can still be run locally.

---

## Project Structure

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

## Installation

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training

### Maintenance Model

```bash
python train_maintenance.py
```

The trained model will be saved as:

```text
models/maintenance_cnn.pth
```

### Property Estimate Model

```bash
python train_property.py
```

The trained model will be saved as:

```text
models/property_model.joblib
```

The Home Scan model does not require local training because it uses a pretrained model.

---

## Running the Application

After the required models have been trained, run:

```bash
streamlit run app.py
```

If the model files already exist inside the `models/` folder, the training scripts do not need to be run again.

---

## Limitations

- **Home Scan is not specifically trained on Indonesian houses** — the pretrained model is still based on a more general housing dataset, so Indonesian housing characteristics may not always be represented well.
- **Maintenance Detection depends on image quality** — lighting, camera angle, image distance, and surface visibility can affect the prediction.
- **Property Estimate still requires improvement** — the estimated price may not always be close to the actual market price because the available input features are still limited.
- **Estimated Range is not always consistent** — the range has been calibrated using validation residuals, but some predictions may still produce a range that is too wide or not close enough to the actual price.
- **Property training is limited to a maximum of 5,000 cleaned records** — this keeps training practical, but it may reduce the representation of the wider Indonesian property market.
- **Important property factors are still missing** — the current model does not directly include road access, nearby facilities, building age, detailed physical condition, or real-time market changes.
- **HABIT is not intended for professional inspection or appraisal** — the results should be treated as an AI-assisted reference.

---

## Future Work

- Improve the **Property Estimate** model through better feature engineering and comparison with other regression algorithms.
- Improve the **Estimated Range** so that it is more stable and closer to actual property prices.
- Add more Indonesian housing images to make **Home Scan** more relevant to local housing conditions.
- Improve **Maintenance Detection** using more varied building-condition and lighting data.
- Evaluate all models using a more representative test dataset.
- Add more detailed location information, such as city or regency, if a suitable dataset becomes available.
- Optimize inference performance for devices without a dedicated GPU.

---

## Project Status

The main HABIT workflow is already functional:

```text
Home Scan
Maintenance Detection
Property Estimate
Home Report
```

However, the project still requires further improvement, especially in:

- Property Estimate accuracy,
- Estimated Range consistency,
- Home Scan generalization,
- Maintenance Detection generalization.

HABIT is currently suitable as a Semester 5 Computer Science project and is still intended for further development and evaluation.

---

## Author

**Keanu Stadeva**

**Computer Science**
