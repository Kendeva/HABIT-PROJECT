# HABIT

HABIT is a data-driven student project that brings several everyday living needs into one practical application. It helps users explore rental price estimates, monthly expense distribution, unusual electricity usage patterns, and estimated maintenance waiting times through a simple Streamlit interface.

> Computer Science Portfolio Project — Semester 5

## Overview

HABIT is designed to make several living-related data tasks easier to understand without requiring users to work directly with datasets or Machine Learning settings. Users enter familiar information related to housing, expenses, electricity usage, or maintenance conditions, and the application presents the result as an estimate or analytical reference.

## Features

### Rental Price Estimation

Uses property information such as area, property type, room details, size, and furnishing status to estimate a monthly rental range. The entered listing price can then be compared with that estimated range.

### Monthly Bills Analysis

Summarizes regular monthly expenses such as rent, electricity, water, internet, and other costs. It also shows the expense distribution so users can see which category contributes the most to their monthly spending.

### Electricity Usage Detection

Uses household information and monthly electricity usage to compare the entered consumption with an expected usage value and identify patterns that appear typical or unusual in relation to the reference data.

### Maintenance Waiting-Time Estimation

Estimates how long a maintenance request may take based on information such as issue type, priority, queue length, technician availability, request hour, and day of the week.

## Machine Learning and Data Analytics

HABIT uses simple Machine Learning and data analytics approaches that match the needs of each feature:

| Feature | Method |
|---|---|
| Rental Price Estimation | Random Forest Regressor |
| Monthly Bills Analysis | Basic Data Analytics |
| Electricity Usage Detection | Random Forest Regressor + Isolation Forest |
| Maintenance Waiting-Time Estimation | Random Forest Regressor |

The Machine Learning flow remains straightforward: load the reference data, prepare the required numerical and categorical features, train the model, process user input, and display the result. The electricity feature also uses Isolation Forest to help detect unusual consumption patterns.

## Project Purpose

HABIT was developed as an educational portfolio project to demonstrate how Machine Learning and data analysis can be applied to practical everyday living problems. The focus is on building a clean and understandable student project that connects data processing, basic Machine Learning, analysis, visualization, and a user interface.

## Dataset

The project uses three synthetic reference datasets for learning and demonstration purposes:

- `data/rental_reference.csv`
- `data/energy_reference.csv`
- `data/maintenance_reference.csv`

These datasets are not official property-market, electricity-provider, or maintenance-company data.

## Technologies

- Python
- Streamlit
- Pandas
- Scikit-learn

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

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Start the Streamlit application:

```bash
streamlit run app.py
```

## Disclaimer

HABIT is a student-developed portfolio project intended for learning and demonstration purposes. Its outputs should be treated as estimates and analytical references rather than official rental market values, professional electrical diagnoses, or guaranteed maintenance completion times.

## What I Learned

Through this project, I practiced:

- Preparing numerical and categorical data for Machine Learning.
- Using Scikit-learn preprocessing and models in a small application.
- Applying regression and anomaly detection to different types of problems.
- Using Pandas and Streamlit charts for simple data analysis and visualization.
- Connecting Machine Learning results to a Streamlit interface.
- Organizing a student project into a small number of readable files.

## Author

Keanu Stadeva  
Computer Science Student
