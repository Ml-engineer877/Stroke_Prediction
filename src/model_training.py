import os
import pandas as pd
import numpy as np

project_path=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
process_path=os.path.join(project_path,"Data","Processed","Stroke Processed Data.csv")
model_path=os.path.join(project_path,"Models","Gaussian_model.pkl")

df=pd.read_csv(process_path)
print(df.head())

def seggrigate(df):
    x=df.iloc[:,1:11]
    y=df["stroke"]
    print(x.shape)
    print(y.shape)
    return x,y

def split(x,y):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    print(x_train.shape)
    print(x_test.shape)
    return x_train,x_test,y_train,y_test

def train(x_train,y_train):
    from sklearn.naive_bayes import GaussianNB
    model=GaussianNB()
    model.fit(x_train,y_train)
    return model

def predict(x_test,y_test,model):
    pred=model.predict(x_test)
    from sklearn.metrics import accuracy_score,confusion_matrix
    print("Accuracy:",accuracy_score(y_test,pred)*100)
    cm=confusion_matrix(y_test,pred)
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.heatmap(cm,annot=True,cmap="coolwarm")
    img_path=os.path.join(project_path,"images")
    plt.savefig(os.path.join(img_path,"Confusion Matrix.png"))
    plt.show()
    return pred

def save(model,model_path):
    import pickle
    with open(model_path,"wb") as f:
        pickle.dump(model,f)
    print("Model Is Saved To The Path:",model_path)

def load_model(model_path):
    import pickle
    with open(model_path,"rb") as f:
        model=pickle.load(f)
    print("Model Is Loaded From:",model_path)
    return model

if __name__ == "__main__":
    x,y=seggrigate(df)
    x_train,x_test,y_train,y_test=split(x,y)
    model=train(x_train,y_train)
    predict(x_test,y_test,model)
    save(model,model_path)
    load_model(model_path)
    