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