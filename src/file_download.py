from sklearn.datasets import load_breast_cancer
import pandas as pd

def download_breast_cancer_data():
    data=load_breast_cancer()
    df=pd.DataFrame(data=data.data,columns=data.feature_names)
    df['target']=data.target
    df.to_csv("../data/breast_cancer_data.csv")

if __name__== "__main__":
    download_breast_cancer_data()    