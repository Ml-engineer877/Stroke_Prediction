import os
import pandas as pd
import numpy as np

project_path=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
test_path=os.path.join(project_path,"Data","Test","Test_Data.csv")

df=pd.read_csv(test_path)
print(df.shape)

def load_model():
    model1_path=os.path.join(project_path,"Models","Gaussian_Model.pkl")
    import joblib
    model1=joblib.load(model1_path)
    print("Model1 Is Loaded Sucessfully")
    model2_path=os.path.join(project_path,"Models","Label_Encoder.pkl")
    model2=joblib.load(model2_path)
    print("Model2 Is Loaded Sucessfully")
    return model1,model2

def encode(model2,df):
    print("Checking Missing Values:",df.isnull().sum())
    print("Droping Missing Values:")
    df=df.dropna()
    var_col=df[["gender","ever_married","work_type","Residence_type","smoking_status"]]
    for col in var_col:
        df[col]=model2.fit_transform(df[col])
    print("After Encoding:",df[["gender","ever_married","work_type","Residence_type","smoking_status"]].head())
    return df

def seggregate(df):
    x=df.drop(columns="stroke")
    y=df["stroke"]
    print(x.shape)
    print(y.shape)
    return x,y

def split(x,y):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    print("Shape of x_train:",x_train.shape)
    print("Shape of y_train:",y_train.shape)
    print("Shape of x_test:",x_test.shape)
    print("Shape of y_test:",y_test.shape)
    return x_train,x_test,y_train,y_test

def model_evaluate(model1,x_test,y_test):
    print(x_test.head())
    pred=model1.predict(x_test)
    from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
    print("Accuracy Of The Model Is:",accuracy_score(y_test,pred)*100)
    print("Classification Report:",classification_report(y_test,pred))
    cm=confusion_matrix(y_test,pred)
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.heatmap(cm,annot=True,cmap="coolwarm")
    img_path=os.path.join(project_path,"images")
    plt.show()
    return pred


if __name__ == "__main__":
    model1,model2=load_model()
    df=encode(model2,df)
    x,y=seggregate(df)
    x_train,x_test,y_train,y_test=split(x,y)
    model_evaluate(model1,x_test,y_test)