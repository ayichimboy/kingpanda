import pandas as pd     
import matplotlib.pyplot as plt                 
import numpy as np  
from pathlib import Path              

def load_data(df):
    """ load csv file """
    return pd.read_csv(df)

def clean_colnames(df):
    """ validate data cols"""
    df.columns = (df.columns.
                str.lower().
                str.strip().
                str.replace("(" ," ").
                str.replace(" ", "_").
                str.replace(")", " ").
                str.strip()
                )
    return df

def check_duplicates(df):
    """ check and validate columns """
    dfx = df.copy()
    dfn= dfx.loc[:, dfx.columns.drop_duplicates()]
    
    dropped_cols = [
        "time_millisecond",
        "wind_speed_mph",
        "wind_direction_degrees",
        "isvideo",
        "isphoto",
        "voltagecell1",
        "voltagecell2",
        "voltagecell3",
        "voltagecell4",
        "voltagecell5",
        "voltagecell6",
        "message", 
    ]
    
    results = dfn.drop(columns=dropped_cols)
    return results

def final_data(df):
    df = df.copy()
    
    selected_cols = [
    "datetime_utc", 
    "voltage_v",
    "battery_percent", 
    "battery_temperature_f",
    "speed_mph",
    "current_a",
    "flycstate", 
    ]
    
    missing = [col for col in selected_cols if col not in df.columns]
    if missing:
        raise ValueError("Selected columns not in data columns")
    
    return df[selected_cols]

def main():
    BASE_DIR= Path(__file__).parent
    DATA_DIR=BASE_DIR/"sample_drone_flights.csv"
    
    df = load_data(DATA_DIR)
    df_clean = clean_colnames(df)
    df_final = check_duplicates(df_clean)
    data = final_data(df_final)
    print(f"final data shape: {data.shape}\n\n"
          f"final data count: {len(data)}")
    
if __name__ == "__main__":
    main()