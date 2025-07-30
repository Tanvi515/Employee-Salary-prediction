import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.dropna(subset=["salary_in_usd"])
    
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in ["job_title", "employee_residence", "company_location"]:
        if col in df.columns:
            top = df[col].value_counts().nlargest(15).index
            df[col] = df[col].where(df[col].isin(top), other="Other")

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df
