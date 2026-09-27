from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.analytics import bill_summary, format_idr
from utils.data_loader import load_energy_data, load_maintenance_data, load_rental_data
from utils.energy_model import analyze_energy, train_energy_models
from utils.maintenance_model import estimate_wait, train_maintenance_model
from utils.rental_model import estimate_rent, train_rental_model

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="HABIT", page_icon="🏠", layout="wide")

# Load a small amount of custom styling.
with open(BASE_DIR / "assets" / "style.css", "r", encoding="utf-8") as file:
    st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_models():
    rental_model = train_rental_model(load_rental_data())
    energy_models = train_energy_models(load_energy_data())
    maintenance_model = train_maintenance_model(load_maintenance_data())
    return rental_model, energy_models, maintenance_model


def show_home():
    st.title("HABIT")
    st.caption("A data-driven application for everyday living needs.")

    st.subheader("Overview")
    st.write(
        "HABIT helps users understand rental costs, monthly expenses, electricity usage, "
        "and maintenance waiting times through one simple application."
    )

    st.subheader("Main Features")
    st.markdown(
        """
        - **Rental Price Estimation** — estimates a monthly rental range from property information.
        - **Monthly Bills Analysis** — summarizes regular expenses and their distribution.
        - **Electricity Usage Detection** — checks whether electricity usage looks typical or unusual.
        - **Maintenance Waiting-Time Estimation** — estimates waiting time from maintenance conditions.
        """
    )


def show_rental(rental_model):
    st.title("Rental Price Estimation")
    st.caption("Enter property details to estimate a monthly rental range.")

    with st.form("rental_form"):
        col1, col2 = st.columns(2)

        with col1:
            area = st.selectbox(
                "Area",
                [
                    "Central Jakarta",
                    "West Jakarta",
                    "South Jakarta",
                    "East Jakarta",
                    "North Jakarta",
                ],
            )
            property_type = st.selectbox(
                "Property type",
                ["Studio", "Apartment", "Boarding House", "Small House"],
            )
            bedrooms = st.number_input("Bedrooms", 1, 5, 1)

        with col2:
            bathrooms = st.number_input("Bathrooms", 1, 4, 1)
            size_m2 = st.number_input("Size (m²)", 15.0, 250.0, 32.0, 1.0)
            furnished = st.selectbox("Furnished", ["No", "Semi", "Yes"])

        listed_rent = st.number_input(
            "Current / listed monthly rent (Rp)",
            500_000,
            50_000_000,
            4_000_000,
            100_000,
        )
        submitted = st.form_submit_button("Estimate Rental Price")

    if submitted:
        values = {
            "area": area,
            "property_type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "size_m2": size_m2,
            "furnished": furnished,
            "listed_rent": listed_rent,
        }
        result = estimate_rent(rental_model, values)

        st.subheader("Result")
        st.write(f"**Status:** {result['status']}")

        col1, col2 = st.columns(2)
        col1.metric("Listed Rent", format_idr(listed_rent))
        col2.metric(
            "Estimated Monthly Range",
            f"{format_idr(result['lower'])} - {format_idr(result['upper'])}",
        )

        difference = result["difference_pct"]
        if result["status"] == "Above Estimated Range":
            st.info(f"The listed rent is about {difference:.1f}% above the estimated range.")
        elif result["status"] == "Below Estimated Range":
            st.info(f"The listed rent is about {difference:.1f}% below the estimated range.")
        else:
            st.success("The listed rent is within the estimated monthly range.")


def show_bills():
    st.title("Monthly Bills Analysis")
    st.caption("Enter regular monthly expenses to review your spending distribution.")

    col1, col2 = st.columns(2)
    with col1:
        rent = st.number_input("Rent", 0, 50_000_000, 4_000_000, 100_000)
        electricity = st.number_input("Electricity", 0, 10_000_000, 400_000, 50_000)
        water = st.number_input("Water", 0, 5_000_000, 100_000, 25_000)

    with col2:
        internet = st.number_input("Internet", 0, 5_000_000, 350_000, 50_000)
        other = st.number_input("Other", 0, 10_000_000, 150_000, 50_000)

    if st.button("Analyze Monthly Bills"):
        result = bill_summary(rent, electricity, water, internet, other)

        col1, col2 = st.columns(2)
        col1.metric("Total Monthly Cost", format_idr(result["total"]))
        col2.metric("Largest Expense", result["largest"])

        chart_data = pd.DataFrame(
            {
                "Category": result["values"].keys(),
                "Amount": result["values"].values(),
            }
        ).set_index("Category")

        st.bar_chart(chart_data)


def show_energy(energy_models):
    st.title("Electricity Usage Detection")
    st.caption("Check whether your monthly electricity usage appears typical or unusual.")

    with st.form("energy_form"):
        col1, col2 = st.columns(2)

        with col1:
            residents = st.number_input("Residents", 1, 10, 2)
            home_type = st.selectbox("Home type", ["Boarding Room", "Apartment", "House"])
            month = st.selectbox("Month", list(range(1, 13)), index=datetime.now().month - 1)

        with col2:
            ac_hours = st.number_input("Average AC use per day (hours)", 0.0, 24.0, 4.0, 0.5)
            usage = st.number_input("Monthly electricity usage (kWh)", 1.0, 3000.0, 150.0, 1.0)

        submitted = st.form_submit_button("Detect Usage Pattern")

    if submitted:
        regression_model, anomaly_preprocessor, anomaly_model = energy_models
        result = analyze_energy(
            regression_model,
            anomaly_preprocessor,
            anomaly_model,
            {
                "residents": residents,
                "home_type": home_type,
                "month": month,
                "ac_hours_per_day": ac_hours,
                "monthly_usage_kwh": usage,
            },
        )

        st.subheader("Result")
        st.write(f"**Status:** {result['status']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Your Usage", f"{usage:.1f} kWh")
        col2.metric("Expected Usage", f"{result['expected']:.1f} kWh")
        col3.metric("Typical Range", f"{result['lower']:.0f} - {result['upper']:.0f} kWh")

        difference = result["difference_pct"]
        direction = "above" if difference >= 0 else "below"
        st.info(
            f"Your usage is about {abs(difference):.1f}% {direction} the expected usage "
            "for a similar household profile."
        )
        st.caption(
            f"Anomaly score: {result['anomaly_score']:.3f}. "
            "This score is not a probability or an electrical diagnosis."
        )


def show_maintenance(maintenance_model):
    st.title("Maintenance Waiting-Time Estimation")
    st.caption("Estimate waiting time based on request and service conditions.")

    with st.form("maintenance_form"):
        col1, col2 = st.columns(2)

        with col1:
            issue_type = st.selectbox(
                "Issue type",
                ["Air Conditioner", "Plumbing", "Electricity", "Internet", "Door/Lock", "Other"],
            )
            priority = st.selectbox("Priority", ["Low", "Normal", "High"], index=1)
            queue_length = st.number_input("Requests ahead", 0, 30, 4)

        with col2:
            technicians = st.number_input("Technicians available", 1, 10, 2)
            request_hour = st.slider("Request hour", 8, 20, 13)
            day_of_week = st.selectbox(
                "Day",
                [
                    (0, "Monday"),
                    (1, "Tuesday"),
                    (2, "Wednesday"),
                    (3, "Thursday"),
                    (4, "Friday"),
                    (5, "Saturday"),
                    (6, "Sunday"),
                ],
                format_func=lambda day: day[1],
            )

        submitted = st.form_submit_button("Estimate Waiting Time")

    if submitted:
        result = estimate_wait(
            maintenance_model,
            {
                "issue_type": issue_type,
                "priority": priority,
                "queue_length": queue_length,
                "technicians_available": technicians,
                "request_hour": request_hour,
                "day_of_week": day_of_week[0],
            },
        )

        st.subheader("Result")
        st.metric("Estimated Waiting Time", f"{result['minutes']:.0f} minutes")
        st.metric("Estimated Range", f"{result['low']:.0f} - {result['high']:.0f} minutes")
        st.write(result["label"])
        st.caption("This is an estimate, not a guaranteed service time.")


def show_about():
    st.title("About HABIT")
    st.write(
        "HABIT is a student-developed data-driven application that applies Machine Learning "
        "and data analytics to practical everyday living problems."
    )
    st.write(
        "Rental Price Estimation, Electricity Usage Detection, and Maintenance Waiting-Time "
        "Estimation use Machine Learning. Monthly Bills Analysis uses straightforward data analytics."
    )
    st.info(
        "HABIT is intended for learning and demonstration purposes. Its outputs are estimates "
        "and analytical references, not official market values, professional diagnoses, or "
        "guaranteed service times."
    )


rental_model, energy_models, maintenance_model = get_models()

st.sidebar.title("HABIT")
st.sidebar.caption("Data-driven living application")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Rental Price Estimation",
        "Monthly Bills Analysis",
        "Electricity Usage Detection",
        "Maintenance Waiting-Time Estimation",
        "About",
    ],
)

if page == "Home":
    show_home()
elif page == "Rental Price Estimation":
    show_rental(rental_model)
elif page == "Monthly Bills Analysis":
    show_bills()
elif page == "Electricity Usage Detection":
    show_energy(energy_models)
elif page == "Maintenance Waiting-Time Estimation":
    show_maintenance(maintenance_model)
else:
    show_about()
