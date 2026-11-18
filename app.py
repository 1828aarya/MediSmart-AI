from pathlib import Path
from datetime import date
import pandas as pd
import streamlit as st
import joblib

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "medicine_dataset.csv"
MODEL_PATH = ROOT / "medicine_model.joblib"

st.set_page_config(page_title="MediSmart AI", page_icon="💊", layout="wide")
st.title("💊 MediSmart AI")
st.caption("Medicine category recognition and information assistant — educational prototype")

st.warning(
    "This project is for educational information only. It does not diagnose conditions, "
    "recommend treatment, confirm a medicine's identity, or replace a pharmacist/doctor. "
    "Never start, stop, substitute, or change a medicine based on this app."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. First run: python train_model.py")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
algorithm = bundle["algorithm"]
catalog = pd.read_csv(DATA_PATH).fillna("")

tab1, tab2, tab3 = st.tabs(["Medicine recognition", "Expiry checker", "Dataset"])

with tab1:
    st.subheader("Enter the details printed on the medicine package")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Medicine name", placeholder="e.g. Paracetamol 500")
        ingredient = st.text_input("Active ingredient", placeholder="e.g. Paracetamol")
    with c2:
        form = st.selectbox("Dosage form", ["Unknown", "Tablet", "Capsule", "Syrup", "Suspension",
                                             "Chewable tablet", "Injection", "Extended-release tablet"])
        strength = st.text_input("Strength as printed", placeholder="e.g. 500 mg")

    if st.button("Recognize category", type="primary"):
        if not any([name.strip(), ingredient.strip()]):
            st.info("Enter at least a medicine name or active ingredient.")
        else:
            text = f"{name} {ingredient} {form if form != 'Unknown' else ''} {strength}"
            prediction = model.predict([text])[0]
            st.success(f"Predicted information category: **{prediction.replace('_', ' ')}**")
            st.caption(f"Model used: {algorithm}")
            if hasattr(model, "predict_proba"):
                confidence = float(model.predict_proba([text]).max())
                st.metric("Model confidence (not medical certainty)", f"{confidence:.1%}")
            st.info("This is a machine-learning category estimate from a small synthetic dataset. Verify the printed label with a pharmacist.")

            # Show matching catalog rows only when the entered ingredient matches exactly.
            if ingredient.strip():
                matches = catalog[catalog["active_ingredient"].str.casefold() == ingredient.strip().casefold()]
                if not matches.empty:
                    st.write("Matching educational catalogue records:")
                    st.dataframe(matches, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Check a medicine's expiry date")
    expiry = st.date_input("Expiry date", value=date.today())
    if st.button("Check expiry"):
        days = (expiry - date.today()).days
        if days < 0:
            st.error(f"Expired {abs(days)} day(s) ago. Do not use it; ask a pharmacist about safe disposal.")
        elif days <= 30:
            st.warning(f"Expires in {days} day(s). Check the package and consult a pharmacist.")
        else:
            st.success(f"Expiry date is {days} day(s) away. Always follow the package storage instructions.")
    st.caption("This simple date check does not validate storage conditions or product authenticity.")

with tab3:
    st.subheader("Educational dataset")
    st.write(f"Rows: {len(catalog)} | Categories: {catalog['category'].nunique()}")
    st.dataframe(catalog, use_container_width=True, hide_index=True)
    st.caption("Synthetic sample records created for demonstrating an ML workflow; not a clinical reference database.")
