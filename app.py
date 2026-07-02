import streamlit as st
import tempfile
import joblib

from features import load_image, extract_features

st.set_page_config(
    page_title="Real vs Fake Detector",
    page_icon="🖼️",
    layout="centered"
)

MODEL_PATH = "real_fake_lgbm_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🖼️ Real vs Fake Image Detector")

st.write(
    "Upload an image to check whether it is a **REAL photo** or a **screen recapture (FAKE)**."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("Analyzing image..."):

        img = load_image(temp_path)

        features = extract_features(img).reshape(1, -1)

        probability = model.predict_proba(features)[0][1]

        prediction = "FAKE" if probability >= 0.5 else "REAL"

    st.divider()

    if prediction == "REAL":
        st.success("✅ Prediction: REAL")
    else:
        st.error("🚨 Prediction: FAKE")

    st.metric(
        "Fake Probability",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    st.caption(
        f"Confidence: {max(probability, 1-probability)*100:.2f}%"
    )