import pandas as pd     
import numpy as np             

    
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

def check_data(df):
    """ validate columns features and others """
    
    df = df.copy()
    selected_cols = ['speed(mph)',
                     'battery_temperature(f)',
                     'flycState'
                     ]
    missing = [col for col in selected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing These columns in Dataframe: {missing}")
    if len(set(df.columns)) - len(df.colums):
        df = df.loc[:, df.columns.drop_duplicate()]   
    df = df.drop_duplicates()   
    
    return df 
