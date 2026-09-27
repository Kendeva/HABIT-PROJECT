from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.analytics import bill_summary, format_idr
from utils.data_loader import load_energy_data, load_maintenance_data, load_rental_data
from utils.energy_model import analyze_energy, train_energy_models
from utils.maintenance_model import estimate_wait, train_maintenance_model
from utils.rental_model import estimate_rent, train_rental_model

BASE_DIR = Path(__file__).resolve().parent
STYLE_PATH = BASE_DIR / "assets" / "style.css"

st.set_page_config(
    page_title="HABIT",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    with STYLE_PATH.open("r", encoding="utf-8") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_models():
    rental_data = load_rental_data()
    energy_data = load_energy_data()
    maintenance_data = load_maintenance_data()

    rental_model = train_rental_model(rental_data)
    energy_models = train_energy_models(energy_data)
    maintenance_model = train_maintenance_model(maintenance_data)

    return rental_model, energy_models, maintenance_model


def show_header(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="kicker">Smart living assistant</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_home():
    show_header(
        "HABIT",
        "A simple AI-powered assistant for everyday living decisions — from rent and bills to energy use and maintenance waiting time.",
    )

    st.markdown("### Living overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rent Tool", "Ready")
    col2.metric("Bills Tool", "Ready")
    col3.metric("Energy Check", "Ready")
    col4.metric("Maintenance", "Ready")

    st.markdown("")
    st.markdown(
        """
        <div class="panel">
        <strong>What you can do with HABIT</strong><br><br>
        • Check whether a rental price looks reasonable.<br>
        • Estimate a suitable rental price range.<br>
        • Calculate and review monthly living expenses.<br>
        • Check whether electricity usage looks typical or unusual.<br>
        • Estimate maintenance waiting time based on current conditions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_rent_check(rental_model):
    show_header(
        "Rent Check",
        "Enter basic property details and HABIT will estimate a reasonable monthly rent based on similar reference properties.",
    )

    with st.form("rent_form"):
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
        submitted = st.form_submit_button("Analyze Rent")

    if not submitted:
        return

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

    st.markdown(
        f"""
        <div class="result-hero">
            <div class="result-label">Rental price status</div>
            <div class="result-value">{result['status']}</div>
            <div class="result-note">
                HABIT compares this listing with rental patterns learned from the reference dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1.7])
    col1.metric("Listed Rent", format_idr(listed_rent))

    with col2:
        st.markdown(
            f"""
            <div class="rent-range-card">
                <div class="rent-range-label">Recommended Monthly Range</div>
                <div class="rent-range-value">
                    {format_idr(result['lower'])} – {format_idr(result['upper'])}
                </div>
                <div class="rent-range-note">
                    Based on similar property patterns in the reference dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    difference = result["difference_pct"]
    if result["status"] == "Higher Than Recommended":
        st.info(
            f"The listed rent is about {difference:.1f}% higher than the top of the recommended range "
            f"({format_idr(result['upper'])})."
        )
    elif result["status"] == "Lower Than Recommended":
        st.info(
            f"The listed rent is about {difference:.1f}% lower than the bottom of the recommended range "
            f"({format_idr(result['lower'])})."
        )
    else:
        st.success("The listed rent falls within HABIT's recommended monthly range.")


def show_monthly_bills():
    show_header(
        "Monthly Bills",
        "Add your regular monthly costs and HABIT will summarize where your living expenses go.",
    )

    col1, col2 = st.columns(2)
    with col1:
        rent = st.number_input("Rent", 0, 50_000_000, 4_000_000, 100_000)
        electricity = st.number_input("Electricity", 0, 10_000_000, 400_000, 50_000)
        water = st.number_input("Water", 0, 5_000_000, 100_000, 25_000)

    with col2:
        internet = st.number_input("Internet", 0, 5_000_000, 350_000, 50_000)
        other = st.number_input("Other", 0, 10_000_000, 150_000, 50_000)

    if not st.button("Calculate Monthly Cost"):
        return

    result = bill_summary(rent, electricity, water, internet, other)

    st.markdown(
        f"""
        <div class="result-hero">
            <div class="result-label">Estimated monthly living cost</div>
            <div class="result-value">{format_idr(result['total'])}</div>
            <div class="result-note">Largest expense category: {result['largest']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_data = pd.DataFrame(
        {
            "Category": list(result["values"].keys()),
            "Amount": list(result["values"].values()),
        }
    )

    chart = px.bar(chart_data, x="Category", y="Amount", text_auto=".2s")
    chart.update_traces(marker_color="#47655a")
    chart.update_layout(
        height=360,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.65)",
        yaxis_title="Rp",
        xaxis_title="",
    )
    st.plotly_chart(chart, use_container_width=True)


def show_energy_usage(energy_models):
    show_header(
        "Energy Usage",
        "Enter your household profile and monthly electricity usage. HABIT will check whether the usage looks typical or unusual.",
    )

    with st.form("energy_form"):
        col1, col2 = st.columns(2)

        with col1:
            residents = st.number_input("Residents", 1, 10, 2)
            home_type = st.selectbox(
                "Home type",
                ["Boarding Room", "Apartment", "House"],
            )
            month = st.selectbox(
                "Month",
                list(range(1, 13)),
                index=datetime.now().month - 1,
            )

        with col2:
            ac_hours = st.number_input(
                "Average AC use per day (hours)",
                0.0,
                24.0,
                4.0,
                0.5,
            )
            usage = st.number_input(
                "Monthly electricity usage (kWh)",
                1.0,
                3000.0,
                150.0,
                1.0,
            )

        submitted = st.form_submit_button("Check Energy Usage")

    if not submitted:
        return

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

    st.markdown(
        f"""
        <div class="result-hero">
            <div class="result-label">Energy usage status</div>
            <div class="result-value">{result['status']}</div>
            <div class="result-note">
                Isolation Forest checks whether this usage pattern differs from similar reference observations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Your Usage", f"{usage:.1f} kWh")
    col2.metric("Expected Usage", f"{result['expected']:.1f} kWh")
    col3.metric(
        "Typical Range",
        f"{result['lower']:.0f}–{result['upper']:.0f} kWh",
    )

    difference = result["difference_pct"]
    direction = "above" if difference >= 0 else "below"
    st.info(
        f"Your entered usage is about {abs(difference):.1f}% {direction} "
        "the expected usage for a similar household profile."
    )
    st.caption(
        f"Anomaly score: {result['anomaly_score']:.3f}. "
        "This score is not a probability and does not diagnose electrical problems."
    )


def show_maintenance(maintenance_model):
    show_header(
        "Maintenance",
        "Estimate how long a maintenance request may take based on queue conditions, request type, and technician availability.",
    )

    with st.form("maintenance_form"):
        col1, col2 = st.columns(2)

        with col1:
            issue_type = st.selectbox(
                "Issue type",
                [
                    "Air Conditioner",
                    "Plumbing",
                    "Electricity",
                    "Internet",
                    "Door/Lock",
                    "Other",
                ],
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

    if not submitted:
        return

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

    st.markdown(
        f"""
        <div class="result-hero">
            <div class="result-label">Estimated waiting time</div>
            <div class="result-value">{result['minutes']:.0f} minutes</div>
            <div class="result-note">
                {result['label']} based on similar historical maintenance conditions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric(
        "Estimated Range",
        f"{result['low']:.0f}–{result['high']:.0f} minutes",
    )
    st.caption("This is an estimate, not a guaranteed service time.")


def show_about():
    show_header(
        "About HABIT",
        "A smart living assistant designed to make everyday housing decisions easier to understand.",
    )
    st.markdown(
        """
        HABIT is a student-built application that brings several common living needs into one simple interface.
        Instead of asking users to work with raw datasets or Machine Learning settings, HABIT lets users enter
        familiar information such as rental details, monthly expenses, electricity usage, or maintenance conditions
        and then presents the result in a form that is easier to understand.

        Machine Learning works behind the application to support rental price estimation, unusual electricity usage
        detection, and maintenance waiting-time estimation. The Monthly Bills feature uses straightforward data
        analytics to help users understand how their regular living expenses are distributed.

        HABIT is designed as an educational portfolio project, so its results should be treated as helpful estimates
        and analytical references rather than official market prices, electrical diagnoses, or guaranteed service times.
        """
    )


def main():
    load_css()
    rental_model, energy_models, maintenance_model = get_models()

    st.sidebar.markdown(
        """
        <div class="habit-brand">
            <div class="habit-name">HABIT</div>
            <div class="habit-sub">Smart living assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Rent Check",
            "Monthly Bills",
            "Energy Usage",
            "Maintenance",
            "About",
        ],
    )

    if page == "Home":
        show_home()
    elif page == "Rent Check":
        show_rent_check(rental_model)
    elif page == "Monthly Bills":
        show_monthly_bills()
    elif page == "Energy Usage":
        show_energy_usage(energy_models)
    elif page == "Maintenance":
        show_maintenance(maintenance_model)
    else:
        show_about()


if __name__ == "__main__":
    main()
