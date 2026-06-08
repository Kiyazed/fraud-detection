import pandas as pd

def clean_data(df):
    df = df.drop_duplicates()
    return df

def convert_datetime(df):
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    return df