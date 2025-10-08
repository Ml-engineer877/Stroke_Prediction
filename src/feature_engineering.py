import os
import pandas as pd
import numpy as np

project_path=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
process_path=os.path.join(project_path,"Data","Processed","Stroke Processed Data.csv")

df=pd.read_csv(process_path)
print(df.head())

def label(df):
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    var_col=df[["gender","ever_married","work_type","Residence_type","smoking_status"]]
    for col in var_col:
        df[col]=le.fit_transform(df[col])
    print(df[["gender","ever_married","work_type","Residence_type","smoking_status"]].head())


def save(df):
    df.to_csv(process_path,index=False)
    print("Cleaned File Is Saved To The Path:",process_path)


if __name__ == "__main__":
    label(df)
    save(df)