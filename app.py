import joblib
import pandas as pd
import streamlit as st

from dataset_config import DATASETS

st.set_page_config(page_title="Heart Disease Risk Prototype")


@st.cache_resource
def load_model():
    return joblib.load("models.joblib")


st.title("Heart Disease Risk Prototype")
st.warning("Academic prototype. Not a medical diagnostic tool.")

dataset_key = st.selectbox(
    "Choose a dataset",
    options=list(DATASETS),
    format_func=lambda key: DATASETS[key]["label"],
)
dataset = DATASETS[dataset_key]
source = pd.read_csv(dataset["path"])

st.caption(f"Prediction target: {dataset['target']}")
inputs = {}
columns = st.columns(2)
for index, feature in enumerate(dataset["numeric"] + dataset["categorical"]):
    with columns[index % 2]:
        label = dataset["field_labels"].get(feature, feature)
        if feature in dataset["numeric"]:
            values = pd.to_numeric(source[feature], errors="coerce").dropna()
            minimum = float(values.min())
            maximum = float(values.max())
            if minimum == maximum:
                minimum -= 1
                maximum += 1
            median = float(values.median())
            step = 0.1 if not values.mod(1).eq(0).all() else 1.0
            inputs[feature] = st.number_input(
                label,
                min_value=minimum,
                max_value=maximum,
                value=min(max(median, minimum), maximum),
                step=step,
            )
        else:
            choices = sorted(source[feature].dropna().unique().tolist(), key=str)
            category_labels = dataset["category_labels"].get(feature, {})
            inputs[feature] = st.selectbox(
                label,
                options=choices,
                format_func=lambda value, labels=category_labels: labels.get(
                    value, str(value)
                ),
                key=f"{dataset_key}_{feature}",
            )

if st.button("Predict", type="primary"):
    row = pd.DataFrame([inputs], columns=dataset["numeric"] + dataset["categorical"])
    model = load_model()[dataset_key]
    probabilities = model.predict_proba(row)[0]
    prediction = model.classes_[probabilities.argmax()]
    probability = float(probabilities.max())
    target_label = dataset["target_labels"].get(prediction, str(prediction))
    st.success(f"Predicted result: {target_label} ({probability:.1%} model confidence)")
    st.progress(probability)
