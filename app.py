import streamlit as st
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import r2_score 
from joblib import dump, load

from numba import jit, cuda

import os 
import re 

from annexe_app import *

def main():
    
    path, model_path=set_path()
    
    data=load_data()
    X=data["data"]
    y=data["target"].to_frame()
    
    X_train, X_test, y_train, y_test, dict_stat=split(X,y)
    
    x_mean=dict_stat["x_mean"]
    x_std=dict_stat["x_std"]
    y_mean=dict_stat["y_mean"]
    y_std=dict_stat["y_std"]
    
  
    st.title("Diabet disease progression one year prediction, exemple")
    st.subheader("By Zied GHARBI")
    f=open(path+"/descreption.txt", "r")
    desc=f.read()
    f.close()
    st.markdown( """ 
    <br>  <br/>
    This application include 3 classics models for prediction namelly : simple linear regression, Ridge regression and Linear model fitted by minimizing a regularized empirical loss with SGD.
    These models were trained on data set descrebied as folow :
    """, unsafe_allow_html=True
    )
    
    st.text(desc)
    
    st.sidebar.markdown('# Choose your parameters')
    
    age=st.sidebar.number_input('Age of patient: ',min_value=1,max_value=100,step=1)
    
    sex = st.sidebar.selectbox("Sex of patient:", ("Male","Female"))
    
    bmi=st.sidebar.number_input("Bmi value :", min_value=16.00, max_value=100.00, step=0.01)
    
    st.sidebar.text("")
    bp=st.sidebar.number_input('Bp index :',min_value=60.00, max_value=133.00,step=0.01)
    
    tc=st.sidebar.number_input('Tc value :',min_value=90.00, max_value=310.00,step=0.01)
    
    ldl=st.sidebar.number_input('Ldl value :',min_value=39.00, max_value=245.00,step=0.01)
    
    hdl=st.sidebar.number_input('Hdl value :',min_value=19.00, max_value=105.00,step=0.01)
    
    tch=st.sidebar.number_input('Tch value :',min_value=0.00, max_value=15.00,step=0.01)
    
    ltg=st.sidebar.number_input('Ltg value :',min_value=0.00, max_value=10.00,step=0.01)
    
    glu=st.sidebar.number_input('Glu value :',min_value=50.00, max_value=124.00,step=0.01)
    
    st.sidebar.markdown('# Choose your model(s) for prediction :')
    model=st.sidebar.multiselect("Models : multiselect authorised :", options=["Linear regression","Ridge regression","SGD regression" ],default=["Linear regression"] )
  
    if st.sidebar.button("Calculate",key="butt"):
        
        rslt={}
        
        if sex=='Male':
            sex=1
        else :
            sex=2
    
        for model in model :
            if model=="Linear regression":
                lr=load(model_path+'/Linear_models.joblib')

                x=np.float_([sex,age,bmi,bp,tc,ldl,hdl,tch,ltg,glu])
                x=x.reshape(1,-1)
                x=pd.DataFrame(x,columns=['sex', 'age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])

                data = (x.loc[:,x.columns!='sex'] - x_mean)/x_std

                data.insert(0,"sex",x["sex"])

                data=data.to_numpy(dtype=float)

                rs=lr.predict(data)
                rslt["Linear regression"]=float(rs)*y_std+y_mean
                
            if model=="Ridge regression":
                lr=load(model_path+'/Ridge_model.joblib')
                x=np.float_([sex,age,bmi,bp,tc,ldl,hdl,tch,ltg,glu])
                x=x.reshape(1,-1)
                x=pd.DataFrame(x,columns=['sex', 'age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])

                data = (x.loc[:,x.columns!='sex'] - x_mean)/x_std

                data.insert(0,"sex",x["sex"])

                data=data.to_numpy(dtype=float)

                rs=lr.predict(data)
                rslt["Ridge regression"]=float(rs)*y_std+y_mean
            
            if model=="SGD regression":
                lr=load(model_path+'/SGDR_model.joblib')
                x=np.float_([sex,age,bmi,bp,tc,ldl,hdl,tch,ltg,glu])
                x=x.reshape(1,-1)
                x=pd.DataFrame(x,columns=['sex', 'age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])

                data = (x.loc[:,x.columns!='sex'] - x_mean)/x_std

                data.insert(0,"sex",x["sex"])

                data=data.to_numpy(dtype=float)

                rs=lr.predict(data)
                rslt["SGD regression"]=float(rs)*y_std+y_mean
        text=""
        
        for key , value in rslt.items():
            text=text+" \n "+ str(key)+" ---- "+str(value)
        
        st.text(text)

if __name__ == '__main__':

    main()
