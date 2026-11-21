"""app.py - the web interface.  Start it with:  streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st

from features import decode_image, extract_features
from rules import rule_based_label, suggestions

st.set_page_config(page_title="AI Image Quality Analyzer", page_icon="📷")
st.title("📷 AI-Based Image Quality Analyzer")
st.caption("Educational prototype. The result is an estimate based on a small "
           "synthetic dataset and five measurements - not a universal measure "
           "of photographic quality.")

MEANINGS = {
    "sharpness": "Edge crispness (higher = sharper)",
    "brightness": "Average light level, 0 black to 255 white",
    "contrast": "Spread between dark and light areas",
    "noise": "Estimated grain/noise (lower = cleaner)",
    "megapixels": "Image size in millions of pixels",
}


@st.cache_resource          # load the model once, not on every click
def load_model():
    return joblib.load("models/quality_model.joblib")


try:
    bundle = load_model()
except FileNotFoundError:
    st.error("Model file not found. Run train.py first.")
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = decode_image(uploaded.getvalue())
    if img is None:
        st.error("That file could not be read as an image.")
        st.stop()

    height, width = img.shape[:2]
    feats = extract_features(img)

    # Ask the trained model for a prediction
    model = bundle["model"]
    row = pd.DataFrame([feats])[bundle["features"]]
    label = model.predict(row)[0]
    probabilities = pd.Series(model.predict_proba(row)[0], index=model.classes_)

    left, right = st.columns(2)
    with left:
        st.image(img, channels="BGR", caption=uploaded.name)   # OpenCV uses BGR
        st.write(f"**Dimensions:** {width} x {height} pixels")
    with right:
        st.metric("Predicted quality", label)
        st.metric("Model confidence", f"{probabilities.max():.0%}")
        st.bar_chart(probabilities)

    st.subheader("Measured features")
    table = pd.DataFrame({
        "Feature": list(feats.keys()),
        "Value": [round(v, 2) for v in feats.values()],
        "Meaning": [MEANINGS[k] for k in feats],
    })
    st.table(table)

    st.subheader("Suggestions")
    for tip in suggestions(feats):
        st.markdown(f"- {tip}")

    st.caption(f"For comparison, the hand-made rule-based baseline (no machine "
               f"learning) says: {rule_based_label(feats)}")
