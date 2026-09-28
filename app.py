import streamlit as st
from PIL import Image, ImageOps

from model import (
    analyze_home,
    analyze_maintenance,
    maintenance_model_ready,
    estimate_property_price,
    property_model_ready,
)
from options import AREAS, PROVINCES

st.set_page_config(page_title="HABIT", page_icon="🏠", layout="wide")

with open("assets/style.css", encoding="utf-8") as file:
    st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
if "home_result" not in st.session_state:
    st.session_state.home_result = None
if "maintenance_result" not in st.session_state:
    st.session_state.maintenance_result = None
if "property_result" not in st.session_state:
    st.session_state.property_result = None
def preview_image(image):
    return ImageOps.fit(
        image.convert("RGB"),
        (640, 480),
        method=Image.Resampling.LANCZOS,
    )
def condition_text(condition):
    messages = {
        "Good": "The uploaded areas appear generally well maintained.",
        "Fair": "Some uploaded areas may need closer attention.",
        "Needs Attention": "One or more uploaded areas show a weaker condition.",
        "Unknown": "The model could not confidently determine the condition.",
    }
    return messages.get(condition, "Home condition result.")
with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <div class="brand-icon">⌂</div>
            <div>
                <h2>HABIT</h2>
                <p>SMART HOME LIVING ASSISTANT</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        ["Home Scan", "Maintenance", "Property Estimate", "Home Report"],
        label_visibility="collapsed",
    )
if page == "Home Scan":
    st.markdown("<p class='eyebrow'>HOME SCAN</p>", unsafe_allow_html=True)
    st.title("Analyze Your Home")
    st.caption(
        "Upload up to five home photos and select the area manually. "
        "HABIT will analyze the visual condition of each image."
    )
    upload_col, result_col = st.columns([0.95, 1.35], gap="large")
    with upload_col:
        with st.container(border=True):
            st.subheader("Home Images")
            files = st.file_uploader(
                "Upload images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="home_images",
            )
            if len(files) > 5:
                st.warning("Use a maximum of 5 images.")
                files = files[:5]
            images = []
            areas = []
            for index, file in enumerate(files):
                image = Image.open(file).convert("RGB")
                images.append(image)
                st.image(preview_image(image), use_container_width=True)
                area = st.selectbox(
                    "Area",
                    AREAS,
                    key=f"home_area_{index}_{file.name}",
                )
                areas.append(area)
            if files and st.button(
                "Analyze Home",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Analyzing home images..."):
                    st.session_state.home_result = analyze_home(images, areas)
            st.caption(
                "The pretrained model downloads once on the first run "
                "and is cached afterward."
            )
    with result_col:
        result = st.session_state.home_result
        with st.container(border=True):
            st.subheader("Overall Home Condition")
            if result is None:
                st.write("Upload images and click **Analyze Home**.")
            else:
                condition_col, score_col = st.columns([3, 1])
                with condition_col:
                    st.markdown(f"### {result['overall_condition']}")
                    st.write(condition_text(result["overall_condition"]))
                with score_col:
                    st.caption("AVG. CONFIDENCE")
                    st.markdown(f"### {result['average_confidence']:.0%}")
                    st.progress(result["average_confidence"])
        with st.container(border=True):
            st.subheader("Area Results")
            if result is None:
                st.write("Area results will appear here in the same 4:3 format.")
            else:
                items = result["areas"]
                for start in range(0, len(items), 2):
                    columns = st.columns(2)
                    for position, item in enumerate(items[start:start + 2]):
                        with columns[position]:
                            st.image(
                                preview_image(item["image"]),
                                use_container_width=True,
                            )
                            st.markdown(
                                f"**{item['area']} · {item['condition']}**"
                            )
                            st.caption(
                                f"Confidence {item['confidence']:.0%}"
                            )
        with st.container(border=True):
            st.subheader("Result Details")
            if result is None:
                st.write("Short result details will appear after analysis.")
            else:
                st.write(
                    f"**Images analyzed:** {len(result['areas'])}  \n"
                    f"**Overall result:** {result['overall_condition']}  \n"
                    f"**Average confidence:** {result['average_confidence']:.0%}"
                )
elif page == "Maintenance":
    st.markdown("<p class='eyebrow'>MAINTENANCE</p>", unsafe_allow_html=True)
    st.title("Check Visible Building Defects")
    st.caption(
        "Upload a wall, ceiling, or surface image for visual defect detection."
    )
    upload_col, result_col = st.columns(2, gap="large")
    with upload_col:
        with st.container(border=True):
            st.subheader("Maintenance Image")
            area = st.selectbox("Area", AREAS, key="maintenance_area")
            file = st.file_uploader(
                "Upload a wall or surface image",
                type=["jpg", "jpeg", "png"],
                key="maintenance_image",
            )
            image = None
            if file is not None:
                image = Image.open(file).convert("RGB")
                st.image(preview_image(image), use_container_width=True)
            if not maintenance_model_ready():
                st.info("Train once with `python train_maintenance.py`.")
            if st.button(
                "Analyze Maintenance",
                type="primary",
                disabled=image is None or not maintenance_model_ready(),
                use_container_width=True,
            ):
                with st.spinner("Checking visible defect..."):
                    result = analyze_maintenance(image)
                result["area"] = area
                st.session_state.maintenance_result = result
    with result_col:
        result = st.session_state.maintenance_result
        with st.container(border=True):
            st.subheader("Maintenance Result")
            if not maintenance_model_ready():
                st.write("The maintenance model is not trained yet.")
            elif result is None:
                st.write("Upload an image and click **Analyze Maintenance**.")
            else:
                st.caption(result["area"].upper())
                st.markdown(f"### {result['label']}")
                st.progress(result["confidence"])
                st.caption(f"Confidence {result['confidence']:.0%}")
        with st.container(border=True):
            st.subheader("Maintenance Details")
            if result is None:
                st.write("A short maintenance note will appear here.")
            else:
                st.write(f"**Area:** {result['area']}")
                st.write(f"**Visual finding:** {result['label']}")
                st.write(f"**Suggested check:** {result['note']}")
elif page == "Property Estimate":
    st.markdown(
        "<p class='eyebrow'>PROPERTY ESTIMATE</p>",
        unsafe_allow_html=True,
    )
    st.title("Estimate Indonesian Property Price")
    st.caption(
        "Uses Indonesian housing-market data. "
        "The result is a price reference, not a professional appraisal."
    )
    with st.container(border=True):
        st.subheader("Property Information")
        first_col, second_col = st.columns(2)
        with first_col:
            province = st.selectbox("Province", PROVINCES)
            subsidy_status = st.selectbox(
                "Housing category",
                ["komersil", "subsidi"],
            )
            land_area = st.number_input(
                "Land area (m²)",
                min_value=1.0,
                value=72.0,
            )
            building_area = st.number_input(
                "Building area (m²)",
                min_value=1.0,
                value=36.0,
            )
        with second_col:
            bedrooms = st.number_input("Bedrooms", min_value=0, value=2)
            bathrooms = st.number_input("Bathrooms", min_value=0, value=1)
            floors = st.number_input("Floors", min_value=1, value=1)
            property_type = st.selectbox("Property type", ["Rumah Tapak"])
        if not property_model_ready():
            st.info("Train once with `python train_property.py`.")
        if st.button(
            "Estimate Property Price",
            type="primary",
            disabled=not property_model_ready(),
            use_container_width=True,
        ):
            info = {
                "province": province,
                "subsidy_status": subsidy_status,
                "land_area_m2": land_area,
                "building_area_m2": building_area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "floors": floors,
                "property_type": property_type,
            }
            try:
                with st.spinner("Estimating Indonesian property price..."):
                    result = estimate_property_price(info)
                st.session_state.property_result = result
            except Exception as error:
                st.error(
                    "Property estimation could not run. "
                    "Please retrain the property model and try again."
                )
                st.caption(str(error))
    result = st.session_state.property_result
    if result is not None:
        price_col, range_col = st.columns(2)
        with price_col:
            with st.container(border=True):
                st.caption("ESTIMATED PRICE")
                st.markdown(f"## Rp{result['price']:,.0f}")
                st.write("Estimated Indonesian property price.")
        with range_col:
            with st.container(border=True):
                st.caption("ESTIMATED RANGE")
                st.markdown(
                    f"### Rp{result['min']:,.0f} – Rp{result['max']:,.0f}"
                )
                st.write(f"Validation MAE: Rp{result['mae']:,.0f}")
else:
    st.markdown("<p class='eyebrow'>HOME REPORT</p>", unsafe_allow_html=True)
    st.title("Home Report")
    st.caption(
        "Summary of the latest Home Scan, Maintenance, "
        "and Property Estimate."
    )
    home = st.session_state.home_result
    maintenance = st.session_state.maintenance_result
    property_result = st.session_state.property_result
    columns = st.columns(3)
    with columns[0]:
        with st.container(border=True):
            st.caption("HOME CONDITION")
            if home is None:
                st.markdown("### Not analyzed")
            else:
                st.markdown(f"### {home['overall_condition']}")
                st.write(f"{len(home['areas'])} area(s) analyzed")
    with columns[1]:
        with st.container(border=True):
            st.caption("MAINTENANCE")
            if maintenance is None:
                st.markdown("### Not checked")
            else:
                st.markdown(f"### {maintenance['label']}")
                st.write(maintenance["area"])
    with columns[2]:
        with st.container(border=True):
            st.caption("PROPERTY")
            if property_result is None:
                st.markdown("### Not estimated")
            else:
                st.markdown(f"### Rp{property_result['price']:,.0f}")
                st.write("Estimated property price")
    with st.container(border=True):
        st.subheader("Summary")
        if home is None and maintenance is None and property_result is None:
            st.write("Complete one of the analyses to generate a report.")
        else:
            if home is not None:
                st.write(
                    f"• Home condition: **{home['overall_condition']}**"
                )
            if maintenance is not None:
                st.write(
                    f"• Maintenance: **{maintenance['label']}** "
                    f"in **{maintenance['area']}**"
                )
            if property_result is not None:
                st.write(
                    f"• Property reference: "
                    f"**Rp{property_result['price']:,.0f}**"
                )
