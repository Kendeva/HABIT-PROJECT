# HABIT

HABIT is a student portfolio project that uses Machine Learning and simple data analytics to support everyday housing decisions. The application helps users review rental prices, summarize monthly bills, check electricity usage patterns, and estimate maintenance waiting time.

> Portfolio Project — Computer Science | Semester 5

LIVE APP : https://habit-project.streamlit.app/

---

## Features

- **Rent Check** — estimates a reasonable monthly rental range and compares it with the listed rent.
- **Monthly Bills** — calculates total monthly living expenses and shows the expense distribution in a chart.
- **Energy Usage** — compares entered electricity usage with an expected value and checks for unusual usage patterns.
- **Maintenance** — estimates maintenance waiting time from queue conditions, request type, and technician availability.

## Technologies

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly

## Machine Learning

| Feature | Method |
|---|---|
| Rent Check | Random Forest Regressor |
| Energy Usage | Random Forest Regressor + Isolation Forest |
| Maintenance | Random Forest Regressor |
| Monthly Bills | Basic Data Analytics |

## Dataset

HABIT currently uses three synthetic reference datasets created for educational and demonstration purposes:

- `data/rental_reference.csv`
- `data/energy_reference.csv`
- `data/maintenance_reference.csv`

These datasets are not official property-market, electricity-provider, or maintenance-company data. Because the datasets are synthetic, the results should be treated as examples of how the Machine Learning workflow works rather than real-world official recommendations.

## Project Structure

```text
HABIT/
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   └── style.css
├── data/
│   ├── rental_reference.csv
│   ├── energy_reference.csv
│   └── maintenance_reference.csv
└── utils/
    ├── analytics.py
    ├── data_loader.py
    ├── energy_model.py
    ├── maintenance_model.py
    └── rental_model.py
```

## How to Run

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:

```bash
streamlit run app.py
```

## Limitations

- Rental results are estimates based on the provided reference dataset, not official market valuations.
- Energy anomaly detection only identifies unusual patterns and does not diagnose electrical problems.
- Maintenance waiting time is an estimate and does not guarantee an exact service time.
- Model behavior depends on the synthetic datasets used in this project.

## Author

Keanu Stadeva  
Computer Science Student
