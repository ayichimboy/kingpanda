import pandas as pd   
import numpy as np       
import matplotlib.pyplot as plt 
from pathlib import Path 
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# load data 
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "temp_data.csv"

if not DATA_DIR.exists():
    raise FileExistsError("Data Doesn't exist")
if not DATA_DIR.suffix in [".csv", ".xlsx"]:
    raise TypeError("Expected File Type Missing")

df = pd.read_csv(DATA_DIR)

if df.empty:
    raise ValueError("Loaded Dataframe is Empty")
if len(set(df.columns)) != len(df.columns):
    print("Data has duplicated columns\n"
          "Dropping Duplicate Columns ...")
    df = df.loc[:, df.columns.drop_duplicates()]  
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

for col in [col for col in df.columns if "date" in col]:
    df[col] = pd.to_datetime(df[col])
df = df.rename(columns={
     'date-time_(cst/cdt)': "datetime", 
     'temperature___(°f)': "temperature_f", 
     'rh___(%)': "relative_humidity",
       'dew_point___(°f)': "dew_point_f", 
    })

df = df.drop(columns=["#", "host_connected", "end_of_file"])
if "datetime" not in df.columns:
    df = df.apply(pd.to_numeric, errors="coerce")

df = df.dropna()

cols = ["temperature_f", "dew_point_f", "relative_humidity"]
dates = pd.to_datetime(df["datetime"], errors="raise").dt.normalize()
daily = (
    df.groupby(dates.rename("date"))[cols]
    .mean()
    .reset_index()
)
selected_cols = ["date","temperature_f"]
new_data = daily.loc[:, selected_cols]

target = "temperature_f"
df_model = new_data.copy()

df_model[target] = pd.to_numeric(df_model[target], errors="coerce")
season_by_month = {
    11: "fall",
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
        }

if df_model.empty or df_model["date"].isna().any():
    raise ValueError("Expected an leas one row or valid date for every row.")
if df_model["date"].duplicated().any():
    raise ValueError("Expected One Temperature per Day")

temperature = df_model.set_index("date")[target].sort_index().asfreq("D")
def build_features(temperature):
    """ build features for training """
    X = pd.DataFrame(index=temperature.index)
    for lag in [1, 2, 3, 7, 14, 30]:
        X[f"temp_lag_{lag}"] = temperature.shift(lag)
        
    calendar_features = ['day','month']
    for feature in calendar_features:
        X[feature] = getattr(X.index, feature)
        
    past = temperature.shift(1)
    X["temp_mean_3d"] = past.rolling(3).mean()
    X["temp_mean_7d"] = past.rolling(7).mean()
    X["temp_change_1d"] = X["temp_lag_1"] - X["temp_lag_2"]
    
    angle = 2 * np.pi * (X.index.dayofyear - 1) / 365.25
    X["annual_sin"] = np.sin(angle)
    X["annual_cos"] = np.cos(angle)
    return X

results = build_features(temperature)
results["season"] = results["month"].map(season_by_month)
winter = results[results["season"].eq("winter")] 
winter_data = winter.drop(columns=["season"]) 
  
X = winter_data.copy()
y = temperature.loc[winter_data.index]

print(X.shape)
print(y.shape)
print(" Data Processsing and Handling Complet ")


