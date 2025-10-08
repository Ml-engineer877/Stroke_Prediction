import os
import pandas as pd 
import numpy as np

project_path=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
raw_path=os.path.join(project_path,"Data","Raw","raw stroke data.csv")
process_dir=os.path.join(project_path,"Data","Processed")
os.makedirs(process_dir,exist_ok=True)
process_path=os.path.join(project_path,"Stroke Processed Data.csv")

df=pd.read_csv(raw_path)
print(df.head())

def missing_value(df):
    print("Checking Missing Values:")
    print(df.isnull().sum())
    df['bmi']=df['bmi'].fillna(df['bmi'].median())
    print(df.isnull().sum())

def duplicate(df):
    dup=df.duplicated().sum()
    print("Check for duplicates:",dup)
    if dup > 0:
        df=df.drop_duplicates()
        print("Duplicates Removed:",df.shape)

def outliers(df):
    num_col=df[["age","avg_glucose_level"]]
    for col in num_col:
        q1=df[col].quantile(0.25)
        q3=df[col].quantile(0.75)
        iqr=q3-q1
        outlier=df[(df[col]<q1-1.5*iqr) | (df[col]>q3+1.5*iqr)]
    print("Outliers",outlier)


def handle_outlier(df):
    num_col=df[["age","avg_glucose_level"]]
    for col in num_col:
        q1=df[col].quantile(0.25)
        q3=df[col].quantile(0.75)
        iqr=q3-q1
        df[col]=df[col].clip(lower=q1-1.5*iqr,upper=q3+1.5*iqr)
        outlier=df[(df[col]<q1-1.5*iqr) | (df[col]>q3+1.5*iqr)]
    print("outliers:",outlier)

def visualize_data(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    img_path=os.path.join(project_path,"images")
    plt.figure(figsize=(6,4))
    sns.boxplot(df[["age","avg_glucose_level"]])
    plt.title("After Outliers")
    plt.savefig(os.path.join(img_path,"Without Outliers.png"))
    plt.show()

def save(df):
    df.to_csv(process_path,index=False)
    print("Cleaned File Is Saved To The Path:",process_path)


if __name__ == "__main__":
    missing_value(df)
    duplicate(df)
    outliers(df)
    handle_outlier(df)
    visualize_data(df)
    save(df)