import joblib
import os

import numpy as np
import pandas as pd
from typing_extensions import Dict,Any
from pydantic import BaseModel, Field
import streamlit as st

BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR,'insurance_regressor_model.joblib')

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]

#Patterns for InsuranceSample categorical values
patterns = {key:f"^({'|'.join(model_data['categorical_columns'][key])})$" for key,value in model_data["categorical_columns"].items()}

class InsuranceSample(BaseModel):
    age : float = Field(None,gt=15,lt=100)
    sex : str = Field(None, min_length=3,max_length=17,pattern=patterns["sex"])
    bmi : float = Field(None,lt=100)
    children : int = Field(None,lt=7) #max in csv is 5
    smoker : str = Field(None,min_length=1,max_length=5,pattern=patterns["smoker"])
    region : str = Field(None,min_length=1,max_length=50,pattern=patterns["region"])

def predict_insurance(insurance_sample: InsuranceSample)->Dict[str, Any]:
    df_sample = pd.DataFrame({key:[value] for key,value in insurance_sample})
    print(f"Sample:\n\r{df_sample}\n")
    log_predict = model.predict(df_sample)
    predict = max(np.expm1(log_predict),0)
    print(f"Prediction:\n\r{predict}\n")
    return {"message":"Success",
            "prediction":predict,
            "model_score":model_data["score"]}

######################## TO RUN:
# 1 INSTALL requirements.txt

st.set_page_config(
    page_title="Insurance Calculator(Regressor)",
    page_icon="🎯🚗"
)

# Title
st.title("🎯🚗 Insurance Calculator(Regressor)")

# If model missing -> show waiting message and stop
if not os.path.exists(MODEL_PATH):
     st.warning("The system is initializing, please wait")
     st.code(MODEL_PATH)
     st.stop()


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

pipe_model = load_model(MODEL_PATH)

model = pipe_model["model"]

model_score = pipe_model["score"] if pipe_model["score"] else None

if model is None or model_score is None:
    st.warning("Model or Score not found")
    st.stop()


# ---- Input form (screenshot-friendly UI) ----
with st.form("insurance_form"):
    st.subheader("Insurance Calculator Details")

    Age = st.number_input(
        "Your Age",
        min_value=18,
        value=18,
        step=1
    )

    Sex = st.selectbox(
        "Gender Status",
        options=[None, "female", "male"],
    )

    Bmi = st.number_input(
        "Bmi",
        min_value=0.0,
        value=0.0,
        step=5.0
    )

    Children = st.number_input(
        "Children Count",
        min_value=0,
        value=0,
        step=1
    )


    Smoker = st.selectbox(
        "Smoker",
        options=[None,"yes", "no"],
        index=0
    )

    Region = st.selectbox(
        "Region",
        options=[None,"northwest", "northeast", "southeast", "southwest"],
        index=0
    )

    check_model_score = st.form_submit_button("Check Model score(R2)")

    submitted = st.form_submit_button("Check Insurance Result")


# Runs only after clicking the button
if submitted:
    if any([Age is None,Sex is None,Bmi is None,Children is None,Smoker is None,Region is None]):
        st.error("All fields are required!")
        st.stop()

    in_sample = InsuranceSample(age=int(Age), sex=str(Sex), bmi=float(Bmi),children=int(Children), smoker=str(Smoker), region=str(Region))
    st.dataframe(pd.DataFrame({key:[value] for key,value in in_sample}))

    prediction = predict_insurance(in_sample)

    if prediction is None:
        st.warning("Prediction Error")
        st.stop()
    st.success(f"Insurance Calculator Result =========  {prediction['prediction'][0]:.2f}$")

if check_model_score:
    st.dataframe(model_score)







