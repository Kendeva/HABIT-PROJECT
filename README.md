# HABIT — Smart Living Assistant

HABIT is a Streamlit application that uses computer vision and machine learning to review visible home conditions, identify possible maintenance issues, and estimate Indonesian property prices.

**Live App:** https://habit-project.streamlit.app/  
**Developer:** Keanu Stadeva — Computer Science, BINUS University

## Features

- **Home Scan:** Analyze home images using a pretrained Vision Transformer (ViT).
- **Maintenance Detection:** Classify visible surface conditions using the current Custom CNN.
- **Property Estimate:** Estimate property prices from location and property details.
- **Home Report:** View results from the current app session in one summary.

## Current Models and Evaluation

| Feature | Model | Available evaluation |
|---|---|---|
| Home Scan | Pretrained ViT from Hugging Face | Prediction confidence is shown; real-world accuracy has not been independently verified |
| Maintenance Detection | Custom CNN (current app model) | 56.0% test accuracy; 0.554 Macro F1 |
| Property Estimate | Trained regression model | R² 0.726; MAE approximately Rp151.9 million |

These metrics reflect the available evaluation data and are not guarantees for every image or property.

## Experimental Model Comparison

A separate ResNet-18 experiment is included to test whether transfer learning improves Maintenance Detection. **The current application continues to use the original Custom CNN.** Running the experiment will not overwrite `models/maintenance_cnn.pth` or `models/maintenance_metrics.json`.

Run the experiment from the project folder:

```bash
python train_maintenance_resnet18.py
```

The script downloads the dataset and ImageNet pretrained weights if needed. It saves its results separately:

- `models/maintenance_resnet18_experiment.pth`
- `models/maintenance_resnet18_experiment_metrics.json`

The experiment reports test accuracy and Macro F1. An 80% accuracy result is a target, not a guarantee. Compare both models using the same dataset split and metrics before deciding whether to integrate ResNet-18 into the live app. Do not describe the new model as reaching 80% unless the measured test result supports that claim.

To retrain the existing Custom CNN, use the original script:

```bash
python train_maintenance.py
```

**Note:** `train_maintenance.py` updates the checkpoint used by the current app. Keep a backup before running it. The separate ResNet-18 experiment does not.

## Run Locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, install dependencies, and start the app.

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux**

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit usually serves the app at `http://localhost:8501`.

## Deployment

Deploy on Streamlit Community Cloud by connecting the GitHub repository and selecting `app.py` as the main file. Keep the model files in `models/` in the repository. Home Scan downloads its pretrained model from Hugging Face on first use, so internet access is required.

## Project Structure

```text
HABIT/
├── app.py
├── model.py
├── options.py
├── requirements.txt
├── train_maintenance.py
├── train_maintenance_resnet18.py
├── train_property.py
├── assets/
│   └── style.css
└── models/
    ├── maintenance_cnn.pth
    ├── maintenance_metrics.json
    ├── property_model.joblib
    └── property_metrics.json
```

The ResNet-18 experiment outputs its checkpoint and metrics into separate files under `models/`.

## Limitations

- This is an educational project, not a professional building inspection or official property appraisal.
- Maintenance detection can miss or misclassify defects.
- Home Scan cannot determine structural safety from images alone.
- Property estimates depend on the training data and may not reflect current market prices or individual property conditions.
- Home Report stores results only in the current app session; it is not a permanent report database.

## Data and Model Sources

- Home Scan model: [DejanX13/vit-house-classifier](https://huggingface.co/DejanX13/vit-house-classifier)
- Maintenance dataset: [chandrabhuma/building_defect_vqa](https://huggingface.co/datasets/chandrabhuma/building_defect_vqa)
- Property dataset: [web3hungry/indonesia-affordable-housing](https://huggingface.co/datasets/web3hungry/indonesia-affordable-housing)

Check the licenses and terms of linked models and datasets before redistribution or commercial use.
