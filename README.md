# HABIT

HABIT is a smart living assistant powered by Machine Learning that helps users evaluate rental prices, understand monthly living expenses, analyze electricity usage, and estimate maintenance waiting times through simple everyday inputs.

> Portfolio Project — Computer Science | Semester 5

---

## Features

- **Rent Check** — evaluates a listed rental price and provides a recommended monthly range
- **Monthly Bills** — calculates and visualizes monthly living expenses
- **Energy Usage** — checks whether electricity consumption appears typical or unusual
- **Maintenance** — estimates waiting time based on queue conditions and technician availability
- Reference datasets are used in the background, so users do not need to upload CSV files or understand Machine Learning concepts

## Machine Learning Models

| Feature | Method |
|---|---|
| Rent Check | Random Forest Regressor |
| Energy Usage | Isolation Forest + Random Forest Regressor |
| Maintenance Wait Estimate | Random Forest Regressor |
| Monthly Bills | Data Analytics |

## Workflow

```text
User Input → Data Processing → Machine Learning / Analytics → Result → Simple Explanation
```

## Dataset

HABIT uses synthetic reference datasets created for educational and demonstration purposes.

| Dataset | Used For |
|---|---|
| `rental_reference.csv` | Rental price range estimation |
| `energy_reference.csv` | Electricity usage pattern analysis |
| `maintenance_reference.csv` | Maintenance waiting time estimation |

The datasets are generated with structured relationships between features so the application can demonstrate realistic Machine Learning behavior. They are not official datasets from the property market, electricity providers, or maintenance companies.

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The application will usually open at:

```text
http://localhost:8501
```

## Author

Keanu Stadeva  
Computer Science Student

## Limitations

- **Synthetic datasets** — model results depend on reference data created specifically for this project
- **Rental recommendation** — the recommended range is not an official market valuation and should not replace professional property assessment
- **Energy anomaly detection** — unusual electricity usage does not automatically indicate an electrical fault or device problem
- **Maintenance estimate** — waiting time is an estimate and does not guarantee an exact service time

## Future Work

- Use more representative property and energy-consumption datasets
- Add monthly history so users can compare changes in living expenses over time
- Add local profile storage for household information
- Develop spending recommendations based on the user's historical expenses
