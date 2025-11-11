import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import sys

# allow importing from src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import load_pickle
from predict import predict_single, predict_batch

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "heart_model.pkl")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------------------------- THEME/CSS ----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff !important;
        color: #000 !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    h1 { color: #B30000 !important; font-weight: 800 !important; }
    h2, h3, h4 { color: #B30000 !important; font-weight: 700 !important; }
    p { font-size:15px !important;
         color: black; }
    /* Make only the three tab labels black */
div[data-baseweb="tab"] button p {
    color: #000000 !important;
}

 
/* unify inputs and select wrapper appearance */
input[type="text"],
input[type="number"],
.stTextInput input,
.stNumberInput input,
div[data-baseweb="select"] input {
    background-color: #ffffff !important;
    color: #000000 !important;        /* ✅ text color fixed */
    border: 1px solid #000000 !important;
    box-shadow: none !important;
    font-size: 15px !important;
    padding: 8px 10px !important;
}

    .stNumberInput > div {
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        padding: 4px !important;
        background: transparent !important;
    }
    .stNumberInput > div > div:first-child input {
        border: 1px solid #e6e6e6 !important;
        background: transparent !important;
        padding: 8px 10px !important;
        color: #ffffff !important;
    }
    .stNumberInput > div > button {
        margin: 0 !important;
        padding: 6px 8px !important;
        min-width: 34px !important;
        height: 36px !important;
        border-radius: 8px !important;
        background: #b30000 !important;
        color: #fff !important;
        border: none !important;
        font-weight: 700 !important;
    }
    .stNumberInput > div > button + button { margin-left: 6px !important; }

    /* remove native spin buttons */
    input[type=number]::-webkit-inner-spin-button,
    input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }

    /* button */
    .stButton > button {
        background: linear-gradient(90deg,#c00000,#b30000) !important;
        color:#fff !important;
        font-size:17px !important;
        padding:10px 26px !important;
        border-radius:10px !important;
        font-weight:700 !important;
        border:none !important;
        box-shadow: 0 6px 14px rgba(179,0,0,0.15);
    }
    .stButton > button:hover { transform: translateY(-2px); }

    /* result cards */
    .risk-card { padding:16px; border-radius:10px; border-left:6px solid #009900; background:#f0fff0; color:#135c13; }
    .danger-card { padding:16px; border-radius:10px; border-left:6px solid #b30000; background:#fff0f0; color:#5c1111; }

    /* ✅ FIXED: Do NOT hide dropdown internal elements */
    /* Removed the buggy block:
       div[data-baseweb="select"] > div > div:first-of-type {
          display:none !important;
       }
    */

    /* dropdown options */
    div[role="listbox"] div { font-size:15px !important; padding:8px 12px !important; color:#111 !important; }
    div[role="option"]:hover { background:#ffeaea !important; }

    /* tabs */
    div[data-baseweb="tab"] > button {
        background-color: #fff !important;
        color: #000000 !important;
        border-bottom: 3px solid rgba(179,0,0,0.12) !important;
        font-weight: 600 !important;
        font-size:16px !important;
        border-radius: 8px 8px 0 0 !important;
    }

    @media (max-width:600px) {
        .stButton > button { width:100% !important; }
        .stNumberInput > div { gap:6px !important; }
    }
    .stToolbar {
    filter: invert(1) hue-rotate(180deg) !important;
}
/* ✅ Make "Single Prediction / Batch / Model Info" BLACK */
div[data-baseweb="tab"] button div p {
    color: #000000 !important;
}

/* ✅ Make selected tab black as well */
div[data-baseweb="tab"] [aria-selected="true"] div p {
    color: #000000 !important;
    font-weight: 700 !important;
}

/* ✅ Make subtitle text black */
.stMarkdown p {
    color: #000000 !important;
}
/* Only tab labels */
div[data-baseweb="tab"] button p {
    color: #000000 !important;
}
header * {
    color: #ffffff !important;
    fill: #ffffff !important;
}

    
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- PAGE ----------------------------
st.title("❤️ Heart Disease Prediction (Logistic Regression)")
st.caption("Modern red-white medical UI — ML powered prediction system")

@st.cache_resource(show_spinner=False)
def _load_model():
    return load_pickle(MODEL_PATH)

try:
    model = _load_model()
    feature_names = list(getattr(model, "feature_names_in_", []))
except Exception as e:
    st.error(f"Failed to load model. Train first:\npython src/train_model.py\n\n{e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📦 Batch (CSV)", "📈 Model Info"])

# ---------------------- TAB 1 ------------------------------------
with tab1:
    st.subheader("💓 Enter Patient Details")
    st.write("Fill all fields. Dropdowns appear automatically for categorical features.")

    cols = st.columns(2)
    inputs = {}

    category_options = {
        "Sex": ["M", "F"],
        "ChestPainType": ["ATA", "NAP", "ASY", "TA"],
        "RestingECG": ["Normal", "ST", "LVH"],
        "ExerciseAngina": ["Y", "N"],
        "ST_Slope": ["Up", "Flat", "Down"],
    }

    for i, feat in enumerate(feature_names):
        with cols[i % 2]:
            if feat in category_options:
                val = st.selectbox(f"{feat}", category_options[feat], key=f"cat_{feat}")
            else:
                val = st.number_input(f"{feat}", value=0.0, key=f"num_{feat}")
            inputs[feat] = val

    if st.button("Predict", key="predict_btn"):
        try:
            pred, proba = predict_single(model, inputs)
            percent = f"{proba*100:.2f}%"

            if pred == 1:
                st.markdown(
                    f"""
                    <div class="danger-card">
                        <h3>❤️ High Risk Detected!</h3>
                        <p>Probability: <b>{percent}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="risk-card">
                        <h3>✅ No Heart Disease Detected</h3>
                        <p>Probability: <b>{percent}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            fig, ax = plt.subplots()
            ax.bar(["No Disease", "Heart Disease"], [1 - proba, proba])
            ax.set_ylabel("Probability")
            ax.set_title("Prediction Probabilities")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------- TAB 2 ------------------------------------
with tab2:
    st.subheader("📦 Batch Prediction from CSV")
    file = st.file_uploader("Choose CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())
        out = predict_batch(model, df)
        st.success("Predictions completed:")
        st.dataframe(out.head())

        st.download_button(
            "Download Predictions CSV",
            out.to_csv(index=False).encode("utf-8"),
            "heart_predictions.csv",
            "text/csv",
        )

        counts = out["prediction"].value_counts().sort_index()
        fig2, ax2 = plt.subplots()
        ax2.bar(["No Disease (0)", "Heart Disease (1)"], counts.values)
        ax2.set_title("Prediction Distribution")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

# ---------------------- TAB 3 ------------------------------------
with tab3:
    st.subheader("📈 Model Information")
    st.write("This model uses Logistic Regression with preprocessing:")
    st.code("ColumnTransformer -> OneHotEncoder -> StandardScaler -> LogisticRegression")
    st.write("**Features used during training:**")
    st.write(feature_names)
    st.info("To retrain: python src/train_model.py")
