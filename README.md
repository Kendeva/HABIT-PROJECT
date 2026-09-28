# HABIT — Smart Home Living Assistant

HABIT has three main AI/ML features:

1. Home Condition — pretrained ViT
2. Maintenance Detection — Custom CNN trained for this project
3. Indonesian Property Estimate — regression model trained for this project

## Datasets and Models

### Home Condition
Pretrained Hugging Face model:

`DejanX13/vit-house-classifier`

No local training is needed for this feature.

### Maintenance Detection
Dataset:

`chandrabhuma/building_defect_vqa`

The dataset contains 3,965 building-surface images with seven defect labels.
The project trains a simple Custom CNN.

Train:

```bash
python train_maintenance.py
```

Output:

```text
models/maintenance_cnn.pth
```

### Indonesian Property Estimate
Dataset:

`web3hungry/indonesia-affordable-housing`

The source contains Indonesian housing records. HABIT uses a maximum sample of
5,000 cleaned records so local training stays realistic for a student project.

Train:

```bash
python train_property.py
```

Output:

```text
models/property_model.joblib
```

The property prediction input is passed as a pandas DataFrame with the same
column names used during training. This avoids the ColumnTransformer error:
"Specifying the columns using strings is only supported for dataframes."

## Project Structure

```text
HABIT_final_fixed/
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
```

## Install and Run

Create virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Train Maintenance:

```bash
python train_maintenance.py
```

Train Property Estimate:

```bash
python train_property.py
```

Run HABIT:

```bash
streamlit run app.py
```

## UI Colors

The CSS uses only four priority colors:

- `#2F4A3C` — deep green
- `#F4F4EF` — off-white
- `#FFFFFF` — white
- `#171A18` — dark text

The footer and bottom area use light backgrounds.

## Notes

- Home Scan uses a pretrained model.
- Maintenance CNN is trained by you.
- Property regression model is trained by you.
- HABIT outputs are AI-assisted references, not professional inspections or appraisals.


## Property Range Improvement

The property model now predicts `log(price)` instead of raw price. This reduces
the effect of extreme property prices.

The estimated range is no longer calculated as only `price ± MAE`.
It is calibrated from real validation residuals:

- 80% of data is used for training.
- 10% is used to calibrate the estimated range.
- 10% is used as a final test set.
- The lower and upper range use the 10th and 90th percentile of calibration
  residuals.

`property_metrics.json` also stores the actual test-set range coverage so the
range can be evaluated instead of guessed.

After this update, retrain the property model:

```bash
python train_property.py
```

Then run:

```bash
streamlit run app.py
```
