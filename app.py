
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="College Cutoff Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "model.pkl"


# ==========================================
# 2. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error("Model load nahi hua.")
    st.code(str(e))
    st.stop()


# ==========================================
# 3. EXTRACT TRAINED INPUT OPTIONS
# ==========================================
try:
    preprocessor = model.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["cat"]

    feature_names = list(model.feature_names_in_)
    encoded_features = [
        "university_id",
        "category",
        "seat_type",
        "figure_kind"
    ]

    # Get categories from the trained OneHotEncoder
    available_options = {
        feature: list(values)
        for feature, values in zip(
            encoded_features,
            encoder.categories_
        )
    }

    # Confirm expected input columns
    expected_features = [
        "university_id",
        "year",
        "category",
        "seat_type",
        "figure_kind"
    ]

    if feature_names != expected_features:
        st.error(
            "Model ke input columns expected columns se match nahi karte."
        )
        st.write("Model columns:", feature_names)
        st.stop()

except Exception as e:
    st.error("Model ki input configuration read nahi hui.")
    st.code(str(e))
    st.stop()


# ==========================================
# 4. CUSTOM DESIGN
# ==========================================
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        padding-top: 8px;
        padding-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 16px;
        margin-bottom: 24px;
    }

    .stForm {
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🎓 Cutoff Predictor")
    st.write("Machine Learning Application")
    st.divider()
    st.info(
        "Enter your admission details to estimate "
        "the expected rank or score."
    )
    st.caption("Model: Random Forest Regressor")


# ==========================================
# 6. MAIN HEADER
# ==========================================
st.markdown(
    '<div class="main-title">🎓 College Cutoff Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict an estimated admission rank or score using your details.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader("📋 Enter Admission Details")

st.write(
    "Choose the available values from your trained model."
)


# ==========================================
# 7. INPUT FORM
# ==========================================
with st.form("cutoff_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        university_id = st.selectbox(
            "University ID",
            options=available_options["university_id"],
            format_func=str
        )

        category = st.selectbox(
            "Category",
            options=available_options["category"],
            format_func=str
        )

        year = st.number_input(
            "Admission Year",
            min_value=2000,
            max_value=2100,
            value=2026,
            step=1
        )

    with col2:
        seat_type = st.selectbox(
            "Seat Type",
            options=available_options["seat_type"],
            format_func=str
        )

        figure_kind = st.selectbox(
            "Prediction Type",
            options=available_options["figure_kind"],
            format_func=str
        )

        st.write("")
        st.caption(
            "Select rank or score according to the available options."
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Now",
        type="primary",
        use_container_width=True
    )


# ==========================================
# 8. MAKE PREDICTION
# ==========================================
if submitted:

    input_data = pd.DataFrame(
        [{
            "university_id": university_id,
            "year": year,
            "category": category,
            "seat_type": seat_type,
            "figure_kind": figure_kind
        }],
        columns=feature_names
    )

    try:
        prediction = model.predict(input_data)
        predicted_value = float(prediction[0])

        st.divider()
        st.subheader("📊 Prediction Result")

        result_col, details_col = st.columns(2)

        with result_col:
            st.metric(
                label=f"Predicted {str(figure_kind).title()}",
                value=f"{predicted_value:,.2f}"
            )

        with details_col:
            st.success("Prediction generated successfully!")

        with st.expander("View Your Input Details"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "This is an ML-based estimate, not a guaranteed "
            "official admission cutoff."
        )

    except Exception as e:
        st.error("Prediction generate nahi hui.")
        st.code(str(e))


# ==========================================
# 9. FOOTER
# ==========================================
st.divider()

st.caption(
    "College Cutoff Predictor | "
    "Python • Scikit-learn • Random Forest • Streamlit"
)
