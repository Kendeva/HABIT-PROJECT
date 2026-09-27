# HABIT

HABIT is a data-driven application that uses Machine Learning and simple data analytics to support everyday housing decisions. The application helps users review rental prices, summarize monthly bills, check electricity usage patterns, and estimate maintenance waiting time.

## Project Objective

The main goal of HABIT is to explore how simple Machine Learning models can be applied to common living and housing problems. The project focuses on creating a practical application that is easy to understand, while still showing the basic workflow of data processing, model training, prediction, and visualization.

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

## How It Works

HABIT uses a simple workflow for each Machine Learning feature:

1. Load the reference dataset.
2. Separate numerical and categorical features.
3. Encode categorical values using `OneHotEncoder`.
4. Train the required model when the application starts.
5. Accept input from the user through Streamlit.
6. Process the input and generate an estimate or analysis result.
7. Display the result in a simple and readable format.

The Energy Usage feature also uses Isolation Forest to check whether the entered electricity usage pattern appears typical or unusual compared with the reference data.

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
├── .streamlit/
│   └── config.toml
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
- The current version trains the models when the application starts instead of loading pre-trained model files.

## Future Work

Possible improvements for future versions of HABIT include:

- Replace or compare the synthetic datasets with suitable public or real-world datasets.
- Add model evaluation using metrics such as MAE, RMSE, and R².
- Compare Random Forest with other suitable regression models to see which performs better on the available data.
- Add a simple history feature so users can review previous rent, bill, energy, or maintenance checks.
- Improve the Energy Usage feature by showing monthly usage trends when historical user data is available.
- Add more property and maintenance categories as the dataset becomes more complete.
- Improve the interface and mobile responsiveness while keeping the application simple and easy to use.

## What I Learned

Through this project, I practiced:

- Preparing numerical and categorical data for Machine Learning.
- Using Scikit-learn pipelines and preprocessing tools.
- Training regression and anomaly detection models.
- Connecting Machine Learning results to a Streamlit interface.
- Presenting predictions and analytics in a form that is easier for users to understand.
- Organizing a small Machine Learning project into simple and reusable Python files.

## Author

Keanu Stadeva  
Computer Science Student
