# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:11:48 2023
This script must be run at least one time before runing app.py for streamlit APP.

In this script models are trained and saved : results may be improved.

This application is for learn purpose only.

@author: Zied
"""
from numba import jit, cuda

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import r2_score 
from joblib import dump, load
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import seaborn as sns
import os 
import re 

def set_path():
    
    path=os.getcwd()
    #path='C:/Users/pc/Nextcloud/Python/streamlit_ex'
    os.chdir(path)
    model_path=path+"/models/"

    isExist = os.path.exists(model_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(model_path)
    
    return path, model_path
        
#@jit(target_backend='cuda')     
def load_data():
    dataset= load_diabetes(as_frame=True, scaled=False)
    return(dataset)

#@jit(target_backend='cuda')     
def save_descreption(data):
    """ cette fonction permet d'ajouter une descreption 
    pour la variable sex et enregistrer 
    la descreption dans un fichier txt """

    text=str(data)
    pattern="      - sex\n"
    new_text=re.sub(pattern, r'      - sex     2 female, 1 male\n',text)
    
    # enregistrer la descreption de la base de données pour l'afficher à l'utilisateur par la suite 
    file=open("descreption.txt", "w")
    file.write ( new_text)
    file.close()
    return(new_text)
    
def split (X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    x_mean=X_train.loc[:,X_train.columns!='sex'].mean(axis=0)
    x_std =X_train.loc[:,X_train.columns!='sex'].std(axis=0)
    
    X_scaled= (X_train.loc[:,X_train.columns!='sex'] - x_mean)/x_std
    X_scaled.insert(0,'sex',X_train['sex'])
       
    y_mean=y_train.mean()
    y_std =y_train.std()
    
    y_scaled=(y_train-y_mean)/y_std
    
    dict_stat={"x_mean": x_mean,
               'x_std':x_std,
               "y_mean":y_mean,
               "y_std":y_std}
    
    return(X_scaled, X_test, y_scaled, y_test, dict_stat)

def affichage (x):
    fig = plt.figure(figsize = (8,10))
    ax = fig.gca()
    x.hist(ax=ax, color = "skyblue")
    plt.show()
    
    
def evaluate(y_test, y_pred):
    mean_abs=mean_absolute_error(y_test, y_pred)
    mean_squ=mean_squared_error(y_test, y_pred)
    r2=r2_score(y_test, y_pred)
    return([mean_abs,mean_squ,r2])
     
#@jit(target_backend='cuda')
def train(models,params,X,y,name,model_path):
    
    X_scaled, X_test, y_scaled, y_test, dict_stat=split(X,y)
    
    x_mean=dict_stat["x_mean"]
    x_std=dict_stat["x_std"]
    y_mean=dict_stat["y_mean"]
    y_std=dict_stat["y_std"]
    
    rslt=[]
    i=0

    for i in range(3):
        model=models[i]
        param=params[i]
        clf=GridSearchCV(model, param)
        clf.fit(X_scaled,y_scaled)
        
        
        X_test_scaled= (X_test.loc[:,X_test.columns!='sex'] - x_mean)/x_std
        X_test_scaled.insert(0,'sex',X_test['sex'])
        
        y_pred=clf.predict(X_test_scaled)
        #clf.best_estimator_
        #clf.best_params_
        dump(clf.best_estimator_, model_path+name[i ]+".joblib") 
        
        y_std_array=np.full((y_pred.shape[0],1), y_std)
        y_mean_array=np.full((y_pred.shape[0],1),y_mean)

        y_pred=np.ravel(y_pred)* np.ravel(y_std_array )+np.ravel(y_mean_array)
        score=evaluate (np.ravel(y_test),y_pred)
        rslt.append(  {"model" : name[i],
                       "mean_abs": score[0],
                       "mean_squ": score[1],
                       "r2":score[2]})
    return(rslt)
    
def main():
   
    path,model_path=set_path()
    
    data=load_data()
    print(save_descreption(data['DESCR']))
 

    X=data["data"]
    y=data["target"].to_frame()
    
    y.describe()
    
    for col in list(X.columns):
        print(X[col].describe())
    
    name_x=list(X.columns)
    name_y=list(y.columns)  
  

    affichage(X.loc[:,X.columns!='sex'])
    
    model1=LinearRegression()
    param1={"fit_intercept": [False, True],
            "positive": [False, True]}   
    model2=Ridge()
    param2={"alpha": [1,10,20,30],
            "fit_intercept" :[False, True] ,
            
            "solver" : ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]} 
    model3=SGDRegressor()
    param3={"loss": ["squared_error","huber"],
            "penalty": ["ll1","l2","elasticnet"],
            "alpha": [0.1,1,10,0.2,0.3],
            "fit_intercept":[False,True]}
    
    models=[model1,model2,model3]
    params=[param1,param2,param3]
    name=["Linear_models", "Ridge_model","SGDR_model"]
    end_score= train(models,params,X,y,name,model_path)
    
    l=list( end_score[1].keys())
   
    results=pd.DataFrame([],index=name, columns= l[-3:])
    
    for i in range(3):
        ll=[]
        dic=end_score[i]
        for key, values in dic.items():
            if type(values)!=str:
                ll.append(values)
        results.iloc[i,:]=ll
    
    results.to_csv(path+"/rslt.txt", header=True, index=True, sep=',', mode='w')
    
    return(results)

main()

