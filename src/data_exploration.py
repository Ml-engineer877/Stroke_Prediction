import os
import numpy as np
import pandas as pd

project_path=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
raw_path=os.path.join(project_path,"Data","Raw","raw stroke data.csv")
images_dir="../images"
os.makedirs(images_dir,exist_ok=True)

df=pd.read_csv(raw_path)
print(df.head())

def info(df):
    print(df.head(5))
    print(df.info())
    print(df.describe())

def missing_value(df):
    print("Missing Values Per Column:",df.isnull().sum())
    print("Stroke Count")
    print(df["stroke"].value_counts(dropna=False))

def visualize_data(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(6,4))
    sns.countplot(data=df,x="stroke")
    plt.title("Distribution Of Strokes")
    plt.savefig(os.path.join(images_dir,"Dis Of Strokes.jpg"))
    plt.show()

    plt.figure(figsize=(6,4))
    sns.boxplot(df[["age","avg_glucose_level"]])
    plt.title("Box Plot Of Numeric Columns")
    plt.savefig(os.path.join(images_dir,"Box Plot Of Num Col.jpg"))
    plt.show()


if __name__ =="__main__":
    info(df)
    missing_value(df)
    visualize_data(df)
    
