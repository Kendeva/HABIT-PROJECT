from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def load_rental_data():
    return pd.read_csv(DATA_DIR / "rental_reference.csv")


def load_energy_data():
    return pd.read_csv(DATA_DIR / "energy_reference.csv")


def load_maintenance_data():
    return pd.read_csv(DATA_DIR / "maintenance_reference.csv")
