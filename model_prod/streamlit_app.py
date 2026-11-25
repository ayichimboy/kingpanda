import os
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("MODEL_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.title("Drone Battery Temperature")
upload = st.file_uploader("Upload a telemetry CSV", type=["csv"])

if upload is not None and st.button("Predict"):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": (upload.name, upload.getvalue(), "text/csv")},
            timeout=30,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        st.error(f"Prediction failed ({exc.response.status_code}).")
        st.code(exc.response.text)
    except requests.RequestException as exc:
        st.error(f"Could not reach the model API. {exc}")
    else:
        result = pd.DataFrame(response.json())
        st.line_chart(result, y="predicted_temperature_f")
        st.dataframe(result)
        st.download_button(
            "Download predictions",
            result.to_csv(index=False),
            file_name="battery_temperature_predictions.csv",
            mime="text/csv",
        )