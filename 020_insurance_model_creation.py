import os
import zipfile

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error,r2_score

BASE_DIR = os.getcwd()
ZIP_PATH = os.path.join(BASE_DIR,'insurance.zip')
EXTRACT_ZIP_PATH = os.path.join(BASE_DIR,'insurance_dataset')

#Extrect ZIP
if not os.path.exists(EXTRACT_ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_ZIP_PATH)
        print("Success")

CSV_PATH = os.path.join(EXTRACT_ZIP_PATH,'insurance.csv')
df = pd.read_csv(CSV_PATH)

#df stats
df_stats = {
    "label":"charges",
    "columns": df.columns.tolist(),
    "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
    "categorical_columns":df.select_dtypes(exclude=[np.number]).columns.tolist(),
}
#print(f"DF Data:\n","\n".join([f"\r{key}:{value}" for key,value in df_stats.items()]))
#print(f"\n\n")

#label
df_stats["numerical_columns"] = [col for col in df_stats["numerical_columns"] if col != df_stats["label"]]
#print(df_stats["numerical_columns"])

categorical_pipeline = Pipeline([
    ("imputer" , SimpleImputer(strategy="most_frequent")),
    ("encoder"  , OneHotEncoder(drop="first", handle_unknown="ignore"))
])

numerical_pipeline = Pipeline([
    ("imputer" , SimpleImputer(strategy="mean")),
    ("scaler" , StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("num" , numerical_pipeline  , df_stats["numerical_columns"]),
    ("cat" , categorical_pipeline , df_stats["categorical_columns"]),
])

pipeline_model = Pipeline([
    ("preprocessor" , preprocessor),
    ("regressor" , LinearRegression())
])

X = df.drop(columns=[df_stats["label"]])
y = df[df_stats["label"]]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#y scaling to avoid negative predictions
y_train_log = np.log1p(y_train)

pipeline_model.fit(X_train, y_train_log)

y_prediction_log = pipeline_model.predict(X_test)

y_prediction = np.expm1(y_prediction_log)

model_score = {
    "r2": r2_score(y_test,y_prediction),
}
categorical_columns_summary = { value:list(set(df[value])) for value in set(df_stats["categorical_columns"])}
print(categorical_columns_summary)

model_data = {
    "model":pipeline_model,
    "score":model_score,
    "categorical_columns":categorical_columns_summary
}

model_name = "insurance_regressor_model.joblib"


joblib.dump(model_data,model_name)
print(f"\n\r{model_name} Saved Successfully\n\rR2:{model_score['r2']}")

















