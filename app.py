# -*- coding: utf-8 -*-
"""
Created on Mon May 23 19:18:26 2022

@author: home
"""

from flask import Flask,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from datetime import datetime
app = Flask(__name__)


@app.route('/svr')
def svr():
    #SVR MODEL-SOH
    df = pd.read_csv("soh_1.csv")    
    #df.describe()
    X_data=df.drop(['soh'],axis=1)
    y_data=df['soh']
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_data, y_data, test_size=0.2,random_state=1)
    x_test_orig=X_test1
  
    x_test_orig=x_test_orig.values.tolist()
    
    X_train1=X_train1.drop(['TimeStamp'],axis=1)
    X_test1=X_test1.drop(['TimeStamp'],axis=1)
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_train1 = sc_x.fit_transform(X_train1)
    X_test1 = sc_x.transform(X_test1)
    y_train1 = sc_y.fit_transform(np.array(y_train1).reshape(-1,1))
    y_test1 = sc_y.transform(np.array(y_test1).reshape(-1,1))
    
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train1,y_train1)
    
    
    
    y_prediction1 =  regressor.predict(X_test1)
    
#     y_prediction2 = sc_y.inverse_transform(y_prediction1)
    y_prediction2 = sc_y.inverse_transform(np.array(y_prediction1).reshape(-1,1))

    actual_soh = sc_y.inverse_transform(np.array(y_test1).reshape(-1,1))
    
    sortto=x_test_orig

    for i in range(len(y_prediction2)):
        sortto[i].append(actual_soh[i][0])
        sortto[i].append(y_prediction2[i])
    
    #for i in range(len(sortto)):
     #       sortto[i][0]=datetime.strftime(datetime.strptime(sortto[i][0],'%Y-%m-%d %H:%M:%S.%f'),'%Y-%m-%d %H:%M:%S.%f')
            
    from operator import itemgetter
    outputlist=sorted(sortto,key=itemgetter(0))
   
            
    #return render_template('home.html',soh=actual_soh,data=x_test_orig,pred=y_prediction2,length=len(y_prediction2))
    return render_template('home.html',sortto=outputlist,length=len(sortto))
@app.route('/svrsoc')
def svrsoc(): 
    #SVR MODEL-SOC
    df = pd.read_csv("soc.csv")
    
    X_data=df.drop(['Relative State of Charge'],axis=1)
    y_data=df['Relative State of Charge']
        
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_data, y_data, test_size=0.2,random_state = 1)
    X_testwithTimeStamp1=X_test1
    x_test_origSOC1=X_testwithTimeStamp1.values.tolist()
    X_train1=X_train1.drop(['TimeStamp'],axis=1)
    X_test1=X_test1.drop(['TimeStamp'],axis=1)
    
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_train1 = sc_x.fit_transform(X_train1)
    X_test1 = sc_x.transform(X_test1)
    y_train1 = sc_y.fit_transform(np.array(y_train1).reshape(-1,1))
    y_test1 = sc_y.transform(np.array(y_test1).reshape(-1,1))
    
    
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train1,y_train1)
    y_prediction1 =  regressor.predict(X_test1)
    
    
    actual_soc = sc_y.inverse_transform(np.array(y_test1).reshape(-1,1))
    
    print(y_test1)



    y_prediction1 = sc_y.inverse_transform(np.array(y_prediction1).reshape(-1,1))
    print(y_prediction1)
    sortto=x_test_origSOC1
    print(len(y_prediction1))
    print(len(x_test_origSOC1[0]))
    for i in range(len(y_prediction1)):
        sortto[i].append(actual_soc[i][0])
        sortto[i].append(y_prediction1[i])
    print(sortto)
    
    #for i in range(len(sortto)):
     #       sortto[i][0]=datetime.strftime(datetime.strptime(sortto[i][0],'%Y-%m-%d %H:%M:%S.%f'),'%Y-%m-%d %H:%M:%S.%f')
            
    from operator import itemgetter
    outputlist=sorted(sortto,key=itemgetter(0))
    print("out:::",outputlist)
    return render_template('svrsoc.html',sortto=outputlist,length=len(sortto))
    #return render_template('home.html',soh=y_test1,data=x_test_origSOC1,pred=y_prediction1,length=len(y_prediction1))
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

@app.route('/nnsoc')
def nnsoc():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pyswarms as ps
    import random
    from pandas import DataFrame
    from pandas import concat
    from sklearn.preprocessing import MinMaxScaler
    
    from scipy.optimize import differential_evolution
    
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    
    Rand = 42
    random.seed(Rand)
    np.random.seed(Rand)
    
        
    from pandas import read_csv
    dataset = read_csv('soc1.csv', header=0, index_col=0)
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_x = MinMaxScaler(feature_range=(0, 1))

    y = values[:, -1]
    x = values[:,:-1]
    df = read_csv('soc1.csv')
    X_data=df.drop(['Relative State of Charge'],axis=1)
    x_data_orig=X_data[1202:]
    x_data_orig=x_data_orig.values.tolist()

    x = scaler_x.fit_transform(x)
    y = scaler.fit_transform(np.array(y).reshape(-1,1))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle = False)
    
    def cal(n_inputs, n_hidden, n_classes):
        i_weights = n_inputs*n_hidden
        i_bias = n_hidden
        h_weights = n_hidden * n_classes
        h_bias = n_classes
        n_params = i_weights + i_bias + h_weights + h_bias
        
        return i_weights, i_bias, h_weights, h_bias, n_params
    n_inputs = 6
    n_hidden = 20
    n_classes = 1
    i_weights , i_bias, h_weights, h_bias, n_params = cal(n_inputs, n_hidden, n_classes)
    def forward(params):
        w1 = params[:i_weights].reshape((n_inputs, n_hidden))
        b1 = params[i_weights:i_weights+i_bias].reshape((n_hidden,))
        w2 = params[i_weights+i_bias:i_weights+i_bias+h_weights].reshape((n_hidden, n_classes))
        b2 = params[i_weights+i_bias+h_weights:].reshape((n_classes,))
        
        z1 = x_train.dot(w1) + b1
        a1 = np.where(z1 > 0, z1, z1*0.01)
        z2 = a1.dot(w2) + b2
        
        loss = mean_squared_error(y_train, z2)
        
        return loss
    def f(x):
        n_particles = x.shape[0]
        j = [ forward(x[i]) for i in range(n_particles)]
        return np.array(j)
    
    def train(options):
        optimizer = ps.single.GlobalBestPSO(n_particles = 100, dimensions = n_params, options = options)
        cost,pos = optimizer.optimize(f, iters = 50)
        return cost, pos, optimizer.cost_history
    
    def predict(X, pos):
        w1 = pos[:i_weights].reshape((n_inputs, n_hidden))
        b1 = pos[i_weights:i_weights+i_bias].reshape((n_hidden,))
        w2 = pos[i_weights+i_bias:i_weights+i_bias+h_weights].reshape((n_hidden, n_classes))
        b2 = pos[i_weights+i_bias+h_weights:].reshape((n_classes,))
        
        z1 = X.dot(w1) + b1
        a1 = np.where(z1 > 0, z1 , z1*0.01)
        z2 = a1.dot(w2)
        
        ypred = z2
        return ypred
    checkpoint_state = np.random.get_state()
    np.random.set_state(checkpoint_state)
    options = {'c1':0.9,'c2':0.1,'w': 0.9}
    cost , pos, history = train(options)
    y_pred = predict(x_test, pos)
    y_pred= scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
    return render_template('nnsoc.html',soh=y_test,data=x_data_orig,pred=y_pred,length=len(y_pred))

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


@app.route('/nnsoh')
def nnsoh():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pyswarms as ps
    import random
    from pandas import DataFrame
    from pandas import concat
    from sklearn.preprocessing import MinMaxScaler
    
    from scipy.optimize import differential_evolution
    
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    
    Rand = 42
    random.seed(Rand)
    np.random.seed(Rand)
    
    from pandas import read_csv
   
    dataset = read_csv('soh_1.csv', header=0, index_col=0)
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    df = read_csv('soh_1.csv', header=0)
    X_data=df.drop(['soh'],axis=1)
    print(X_data[101:])
    x_data_orig=X_data.values.tolist()
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    
    y = values[:, -1]
  
    x = values[:,:-1]

    x = scaler_x.fit_transform(x)
    
    y = scaler.fit_transform(np.array(y).reshape(-1,1))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle = False)
    print(x_test)
    def cal(n_inputs, n_hidden, n_classes):
        i_weights = n_inputs*n_hidden
        i_bias = n_hidden
        h_weights = n_hidden * n_classes
        h_bias = n_classes
        n_params = i_weights + i_bias + h_weights + h_bias
        
        return i_weights, i_bias, h_weights, h_bias, n_params
            
    
    n_inputs = 7
    n_hidden = 20
    n_classes = 1
    i_weights , i_bias, h_weights, h_bias, n_params = cal(n_inputs, n_hidden, n_classes)
    
    def forward(params):
        w1 = params[:i_weights].reshape((n_inputs, n_hidden))
        b1 = params[i_weights:i_weights+i_bias].reshape((n_hidden,))
        w2 = params[i_weights+i_bias:i_weights+i_bias+h_weights].reshape((n_hidden, n_classes))
        b2 = params[i_weights+i_bias+h_weights:].reshape((n_classes,))
        
        z1 = x_train.dot(w1) + b1
        a1 = np.where(z1 > 0, z1, z1*0.01)
        z2 = a1.dot(w2) + b2
        
        loss = mean_squared_error(y_train, z2)
        
        return loss
    
    def f(x):
        n_particles = x.shape[0]
        j = [ forward(x[i]) for i in range(n_particles)]
        return np.array(j)
    
    def train(options):
        optimizer = ps.single.GlobalBestPSO(n_particles = 100, dimensions = n_params, options = options)
        cost,pos = optimizer.optimize(f, iters = 50)
        return cost, pos, optimizer.cost_history
    
    def predict(X, pos):
        w1 = pos[:i_weights].reshape((n_inputs, n_hidden))
        b1 = pos[i_weights:i_weights+i_bias].reshape((n_hidden,))
        w2 = pos[i_weights+i_bias:i_weights+i_bias+h_weights].reshape((n_hidden, n_classes))
        b2 = pos[i_weights+i_bias+h_weights:].reshape((n_classes,))
        
        z1 = X.dot(w1) + b1
        a1 = np.where(z1 > 0, z1 , z1*0.01)
        z2 = a1.dot(w2)
        
        ypred = z2
        return ypred
    
    
    
    
    checkpoint_state = np.random.get_state()
    np.random.set_state(checkpoint_state)
    options = {'c1':0.9,'c2':0.1,'w': 0.9}
    cost , pos, history = train(options)
    
    
    y_pred = predict(x_test, pos)
    y_pred= scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
    
    return render_template('nnsoc.html',soh=y_test,data=x_data_orig,pred=y_pred,length=len(y_pred))
#////////////////////////////////////////////////////////////////////////////////////////////////////////

@app.route('/gasoc')
def gasoc():
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import pickle
    from pandas import read_csv
    from keras.models import load_model

    dataset = pd.read_csv('soc1.csv', header=0, index_col=0)
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    y = values[:, -1]
    x = values[:,:-1]
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(np.array(y).reshape(-1,1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle = False)
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    modelGA=load_model('modelGA.h5')
    predictY = modelGA.predict(x_test)
    
    predictY = scaler.inverse_transform(predictY)
    df = read_csv('soc1.csv')
    X_data=df.drop(['Relative State of Charge'],axis=1)
    x_data_orig=X_data[1202:]
    x_data_orig=x_data_orig.values.tolist()
    y_test = scaler.inverse_transform(y_test)
    return render_template('nnsoc.html',soh=y_test,data=x_data_orig,pred=predictY,length=len(predictY))

@app.route('/gasoh')
def gasoh():
    
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import pickle
    from pandas import read_csv
    from keras.models import load_model
    dataset = pd.read_csv('soh_1.csv', header=0, index_col=0)
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    y = values[:, -1]
    x = values[:,:-1]
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(np.array(y).reshape(-1,1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle = False)
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    modelGAsoh=load_model('modelGASOH.h5')
    y_test = scaler.inverse_transform(y_test)
    predictY = modelGAsoh.predict(x_test)
    predictY = scaler.inverse_transform(predictY)
    df = read_csv('soh_1.csv', header=0)
    
    X_data=df.drop(['soh'],axis=1)
    print(X_data[101:])
    x_data_orig=X_data.values.tolist()
    return render_template('nnsoc.html',soh=y_test,data=x_data_orig,pred=predictY,length=len(predictY))
   
@app.route('/rfsoh')
def rfsoh():
    from sklearn import preprocessing
    df = pd.read_csv("soh_1.csv")    
    #df.describe()
    X_data=df.drop(['soh'],axis=1)
    y_data=df['soh']
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_data, y_data, test_size=0.2,random_state=1,shuffle=False)
    x_test_orig=X_test1
    x_test_orig=x_test_orig.values.tolist()
    
    X_train1=X_train1.drop(['TimeStamp'],axis=1)
    X_test1=X_test1.drop(['TimeStamp'],axis=1)
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_train1 = sc_x.fit_transform(X_train1)
    X_test1 = sc_x.transform(X_test1)
    y_train1 = sc_y.fit_transform(np.array(y_train1).reshape(-1,1))
    y_test1 = sc_y.transform(np.array(y_test1).reshape(-1,1))
    from sklearn.ensemble import RandomForestRegressor
    rf=RandomForestRegressor()
    model = rf.fit(X_train1,y_train1)

    y_test_predict=model.predict(X_test1)
    y_prediction2 = sc_y.inverse_transform(y_test_predict)
    actual_soh = sc_y.inverse_transform(y_test1)
    return render_template('home.html',soh=actual_soh,data=x_test_orig,pred=y_prediction2,length=len(y_prediction2))



if __name__ == "__main__":
    app.run(debug=True)
       
