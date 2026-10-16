# HABIT — Smart Living Assistant

HABIT is a student-built AI application that helps users explore home conditions, identify possible visible building defects, and estimate Indonesian property prices. It combines a pretrained image-classification model with models trained for this project in a Streamlit interface.

> Computer Science Project — BINUS University, Semester 5

LIVE APP : https://habit-project.streamlit.app/
=======
**Live demo:** https://habit-project.streamlit.app/  
**Author:** Keanu Stadeva · Computer Science, BINUS University
>>>>>>> 075fadb (Finalize HABIT smart living application)

> HABIT is an educational prototype. Its outputs are estimates and visual predictions, not professional building inspections, structural assessments, or official property valuations.

## Features

### 1. Home Scan
- Upload up to five JPG/PNG photos and select the area represented by each image.
- Classify the visual condition using the pretrained `DejanX13/vit-house-classifier` model.
- View per-image predictions, model scores, and an overall summary.
- The model downloads from Hugging Face on first use and is cached during the app session.

### 2. Maintenance Detection
- Upload a photo of a wall, ceiling, or building surface.
- A custom CNN predicts one of seven classes: algae, major crack, minor crack, peeling, plain/no clear defect, spalling, or stain.
- Provides a short suggested follow-up check.
- The current saved test metrics are approximately **56.0% accuracy** and **0.554 Macro F1**. Results should therefore be treated as preliminary and checked by a person.

### 3. Property Estimate
- Enter province, housing category, land/building area, bedrooms, bathrooms, floors, and property type.
- Predict a reference price and an estimated range using a regression pipeline trained on Indonesian housing data.
- The training script compares HistGradientBoosting, Random Forest, and Gradient Boosting models and selects the model with the lowest validation MAE.
- The included metrics file records 5,000 cleaned records and a held-out test R² of approximately **0.726** with MAE around **Rp151.9 million**. This is dataset-specific and does not guarantee accuracy for every region or property.

### 4. Home Report
Shows the latest available results from Home Scan, Maintenance Detection, and Property Estimate in one summary.

## Models and data sources

| Feature | Model | Source / training approach |
|---|---|---|
| Home Scan | Vision Transformer image classifier | Pretrained Hugging Face model: [`DejanX13/vit-house-classifier`](https://huggingface.co/DejanX13/vit-house-classifier) |
| Maintenance Detection | Custom CNN built with PyTorch | Fine-tuned from scratch using [`chandrabhuma/building_defect_vqa`](https://huggingface.co/datasets/chandrabhuma/building_defect_vqa) |
| Property Estimate | Scikit-learn regression pipeline | Trained using [`web3hungry/indonesia-affordable-housing`](https://huggingface.co/datasets/web3hungry/indonesia-affordable-housing), limited to at most 5,000 cleaned records |

The pretrained Home Scan model is not a model created from scratch by this project. The maintenance CNN architecture and property-estimation pipeline are implemented and trained by the project scripts. Dataset/model licenses and upstream changes should be reviewed before redistribution or commercial use.

## Project structure

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
    ├── maintenance_metrics.json
    ├── property_model.joblib
    └── property_metrics.json
```

The saved model files are included in this project archive. If they are missing from a fresh checkout, run the relevant training script below. Training metrics are saved locally and are excluded by `.gitignore` so generated artifacts are not committed accidentally.

## Requirements

- Python 3.10 or newer is recommended.
- Internet connection for downloading Hugging Face models/datasets.
- PyTorch can run on CPU; a compatible CUDA setup can speed up training when available.

## Install and run locally

Run these commands from the project folder.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks environment activation, use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit normally opens `http://localhost:8501` in your browser. Keep the terminal running while using the app.

## Train the models

Training is optional when the saved model files are already present.

```bash
python train_maintenance.py
python train_property.py
```

The scripts download their datasets from Hugging Face and save model artifacts and evaluation metrics under `models/`. Training may take time and memory, particularly on CPU. Home Scan uses a pretrained model and does not need the local training scripts.

## Evaluation and honest reporting

- **Maintenance:** report both accuracy and Macro F1 because class performance can differ. The currently recorded test result is modest, so do not describe this model as highly accurate.
- **Property Estimate:** use held-out test metrics (MAE, RMSE, R²) and the estimated-range coverage as measured by the training script. Errors are in Indonesian rupiah and can be large because property prices vary widely.
- **Home Scan:** the displayed confidence is the model's prediction score, not a verified accuracy percentage. A model score of 90% does not mean that 90% of all real-world predictions are correct.
- **Originality:** do not assign the whole application an arbitrary originality percentage. Describe the work accurately: a pretrained model is integrated for Home Scan, while the maintenance CNN and property pipeline are implemented/trained for this project.

## Limitations

- Predictions depend on the datasets, image quality, class balance, and input values used during development.
- Maintenance predictions may miss defects or misclassify a surface; they are not a substitute for a qualified inspection.
- Home Scan predicts broad visual categories and cannot establish structural safety.
- Property estimates are data-driven references, not appraisals, and may not reflect current local market prices or unique property features.
- The Home Report summarizes the latest results in the current session; it is not a permanent report database.
- Internet access is needed the first time pretrained weights or training datasets are downloaded.

## Troubleshooting

- **`ModuleNotFoundError`:** activate `.venv` and run `pip install -r requirements.txt`.
- **Model file missing:** run the corresponding training script, or restore the model artifact from the project backup.
- **Home Scan cannot download the model:** check internet access and the Hugging Face model page.
- **Port 8501 is busy:** run `streamlit run app.py --server.port 8502` and open `http://localhost:8502`.

## License and attribution

This repository integrates third-party datasets and a pretrained model. Check each upstream repository's license and terms before redistribution or commercial use. Add a project license only after confirming that the project code and included artifacts can legally be distributed under that license.
