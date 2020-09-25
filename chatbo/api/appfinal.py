from flask import Flask,request
import requests
import pandas as pd
from datetime import datetime
from pandas_datareader import data

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import os
import requests
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries

from sklearn.preprocessing import MinMaxScaler
import json
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,Flatten
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from pandas_datareader import data
import codecs, json 
from flask_jsonpify import jsonpify
Exporter = Flask(__name__)

##############################################################################################
#BIRCOIN : BTC-USD

# pylint: disable=too-many-function-args
@Exporter.route("/modelBTC-USD",methods=['POST','GET'])
def ScrapingBTC():
    print("****************")
    crypto = request.values.get("crypto") 
    print(crypto)
    #Setting the end date to today
    end = datetime.today()

    #Start date set to one year back
    start = datetime(2018,1,1)
    #using yahoo finance to grab cryptocurrency data
    bit_data=data.DataReader(crypto ,'yahoo',start,end)
    bit_data =pd.DataFrame(bit_data)

    group=bit_data.groupby("Date")
    dataa=group['Close'].mean()
    dataa = pd.DataFrame(dataa)
    close_train=dataa.iloc[:len(dataa)-30]
    close_test=dataa.iloc[len(close_train):]
    
    #feature scalling (set values between 0-1)
    close_train=np.array(close_train)
    close_train=close_train.reshape(close_train.shape[0],1)
    scaler=MinMaxScaler(feature_range=(0,1))
    close_scaled=scaler.fit_transform(close_train)
    timestep=50
    x_train=[]
    y_train=[]

    for i in range(timestep,close_scaled.shape[0]):
        x_train.append(close_scaled[i-timestep:i,0])
        y_train.append(close_scaled[i,0])

    x_train,y_train=np.array(x_train),np.array(y_train)
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1) #reshaped for RNN

    model=Sequential()
    model.add(LSTM(10,input_shape=(None,1),activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    model.fit(x_train,y_train,epochs=100,batch_size=32)
    inputs=dataa[len(dataa)-len(close_test)-timestep:]
    inputs=inputs.values.reshape(-1,1)
    inputs=scaler.transform(inputs)
    x_test=[]
    for i in range(timestep,inputs.shape[0]):
        x_test.append(inputs[i-timestep:i,0])
    x_test=np.array(x_test)
    x_test= x_test.reshape(x_test.shape[0],x_test.shape[1],1)
    predicted_data=model.predict(x_test)
    predicted_data=scaler.inverse_transform(predicted_data)
    pred = predicted_data[29]
    pred = pred.tolist()

    json_data = json.dumps(pred)
    return json_data


##############################################################################################
#ETHERUM : ETH-USD

# pylint: disable=too-many-function-args
@Exporter.route("/modelETH-USD",methods=['POST','GET'])
def ScrapingETH():
    print("****************")
    crypto = request.values.get("crypto") 
    print(crypto)
    #Setting the end date to today
    end = datetime.today()

    #Start date set to one year back
    start = datetime(2020,2,1)
    #using yahoo finance to grab cryptocurrency data
    bit_data=data.DataReader(crypto ,'yahoo',start,end)
    bit_data =pd.DataFrame(bit_data)

    group=bit_data.groupby("Date")
    dataa=group['Close'].mean()
    dataa = pd.DataFrame(dataa)
    close_train=dataa.iloc[:len(dataa)-30]
    close_test=dataa.iloc[len(close_train):]
    
    #feature scalling (set values between 0-1)
    close_train=np.array(close_train)
    close_train=close_train.reshape(close_train.shape[0],1)
    scaler=MinMaxScaler(feature_range=(0,1))
    close_scaled=scaler.fit_transform(close_train)
    timestep=50
    x_train=[]
    y_train=[]

    for i in range(timestep,close_scaled.shape[0]):
        x_train.append(close_scaled[i-timestep:i,0])
        y_train.append(close_scaled[i,0])

    x_train,y_train=np.array(x_train),np.array(y_train)
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1) #reshaped for RNN

    model=Sequential()
    model.add(LSTM(10,input_shape=(None,1),activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    model.fit(x_train,y_train,epochs=100,batch_size=32)
    inputs=dataa[len(dataa)-len(close_test)-timestep:]
    inputs=inputs.values.reshape(-1,1)
    inputs=scaler.transform(inputs)
    x_test=[]
    for i in range(timestep,inputs.shape[0]):
        x_test.append(inputs[i-timestep:i,0])
    x_test=np.array(x_test)
    x_test= x_test.reshape(x_test.shape[0],x_test.shape[1],1)
    predicted_data=model.predict(x_test)
    predicted_data=scaler.inverse_transform(predicted_data)
    pred = predicted_data[29]
    pred = pred.tolist()

    json_data = json.dumps(pred)
    return json_data



##############################################################################################
#FACEBOOK : FB

# pylint: disable=too-many-function-args
@Exporter.route("/modelFB",methods=['POST','GET'])
def ScrapingFB():
    print("****************")
    crypto = request.values.get("crypto") 
    print(crypto)
    #Setting the end date to today
    end = datetime.today()

    #Start date set to one year back
    start = datetime(2020,2,1)
    #using yahoo finance to grab cryptocurrency data
    bit_data=data.DataReader(crypto ,'yahoo',start,end)
    bit_data =pd.DataFrame(bit_data)

    group=bit_data.groupby("Date")
    dataa=group['Close'].mean()
    dataa = pd.DataFrame(dataa)
    close_train=dataa.iloc[:len(dataa)-30]
    close_test=dataa.iloc[len(close_train):]
    
    #feature scalling (set values between 0-1)
    close_train=np.array(close_train)
    close_train=close_train.reshape(close_train.shape[0],1)
    scaler=MinMaxScaler(feature_range=(0,1))
    close_scaled=scaler.fit_transform(close_train)
    timestep=50
    x_train=[]
    y_train=[]

    for i in range(timestep,close_scaled.shape[0]):
        x_train.append(close_scaled[i-timestep:i,0])
        y_train.append(close_scaled[i,0])

    x_train,y_train=np.array(x_train),np.array(y_train)
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1) #reshaped for RNN

    model=Sequential()
    model.add(LSTM(10,input_shape=(None,1),activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    model.fit(x_train,y_train,epochs=100,batch_size=32)
    inputs=dataa[len(dataa)-len(close_test)-timestep:]
    inputs=inputs.values.reshape(-1,1)
    inputs=scaler.transform(inputs)
    x_test=[]
    for i in range(timestep,inputs.shape[0]):
        x_test.append(inputs[i-timestep:i,0])
    x_test=np.array(x_test)
    x_test= x_test.reshape(x_test.shape[0],x_test.shape[1],1)
    predicted_data=model.predict(x_test)
    predicted_data=scaler.inverse_transform(predicted_data)
    pred = predicted_data[29]
    pred = pred.tolist()

    json_data = json.dumps(pred)
    return json_data


##############################################################################################

#GOOGle : GOOG

# pylint: disable=too-many-function-args
@Exporter.route("/modelGOOG",methods=['POST','GET'])
def ScrapingGOOG():
    print("****************")
    crypto = request.values.get("crypto") 
    print(crypto)
    #Setting the end date to today
    end = datetime.today()

    #Start date set to one year back
    start = datetime(2020,1,1)
    #using yahoo finance to grab cryptocurrency data
    bit_data=data.DataReader(crypto ,'yahoo',start,end)
    bit_data =pd.DataFrame(bit_data)

    group=bit_data.groupby("Date")
    dataa=group['Close'].mean()
    dataa = pd.DataFrame(dataa)
    close_train=dataa.iloc[:len(dataa)-30]
    close_test=dataa.iloc[len(close_train):]
    
    #feature scalling (set values between 0-1)
    close_train=np.array(close_train)
    close_train=close_train.reshape(close_train.shape[0],1)
    scaler=MinMaxScaler(feature_range=(0,1))
    close_scaled=scaler.fit_transform(close_train)
    timestep=50
    x_train=[]
    y_train=[]

    for i in range(timestep,close_scaled.shape[0]):
        x_train.append(close_scaled[i-timestep:i,0])
        y_train.append(close_scaled[i,0])

    x_train,y_train=np.array(x_train),np.array(y_train)
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1) #reshaped for RNN

    model=Sequential()
    model.add(LSTM(10,input_shape=(None,1),activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    model.fit(x_train,y_train,epochs=50,batch_size=32)
    inputs=dataa[len(dataa)-len(close_test)-timestep:]
    inputs=inputs.values.reshape(-1,1)
    inputs=scaler.transform(inputs)
    x_test=[]
    for i in range(timestep,inputs.shape[0]):
        x_test.append(inputs[i-timestep:i,0])
    x_test=np.array(x_test)
    x_test= x_test.reshape(x_test.shape[0],x_test.shape[1],1)
    predicted_data=model.predict(x_test)
    predicted_data=scaler.inverse_transform(predicted_data)
    pred = predicted_data[29]
    pred = pred.tolist()

    json_data = json.dumps(pred)
    return json_data


##############################################################################################

#FOREX EURO : EURUSD=X

# pylint: disable=too-many-function-args
@Exporter.route("/modelEURUSD=X",methods=['POST','GET'])
def ScrapingEUR():
    print("****************")
    crypto = request.values.get("crypto") 
    print(crypto)
    #Setting the end date to today
    end = datetime.today()

    #Start date set to one year back
    start = datetime(2018,1,1)
    #using yahoo finance to grab cryptocurrency data
    bit_data=data.DataReader(crypto ,'yahoo',start,end)
    bit_data =pd.DataFrame(bit_data)

    group=bit_data.groupby("Date")
    dataa=group['Close'].mean()
    dataa = pd.DataFrame(dataa)
    close_train=dataa.iloc[:len(dataa)-30]
    close_test=dataa.iloc[len(close_train):]
    
    #feature scalling (set values between 0-1)
    close_train=np.array(close_train)
    close_train=close_train.reshape(close_train.shape[0],1)
    scaler=MinMaxScaler(feature_range=(0,1))
    close_scaled=scaler.fit_transform(close_train)
    timestep=50
    x_train=[]
    y_train=[]

    for i in range(timestep,close_scaled.shape[0]):
        x_train.append(close_scaled[i-timestep:i,0])
        y_train.append(close_scaled[i,0])

    x_train,y_train=np.array(x_train),np.array(y_train)
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1) #reshaped for RNN

    model=Sequential()
    model.add(LSTM(10,input_shape=(None,1),activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    model.fit(x_train,y_train,epochs=50,batch_size=32)
    inputs=dataa[len(dataa)-len(close_test)-timestep:]
    inputs=inputs.values.reshape(-1,1)
    inputs=scaler.transform(inputs)
    x_test=[]
    for i in range(timestep,inputs.shape[0]):
        x_test.append(inputs[i-timestep:i,0])
    x_test=np.array(x_test)
    x_test= x_test.reshape(x_test.shape[0],x_test.shape[1],1)
    predicted_data=model.predict(x_test)
    predicted_data=scaler.inverse_transform(predicted_data)
    pred = predicted_data[29]
    pred = pred.tolist()

    json_data = json.dumps(pred)
    return json_data


##############################################################################################
#FOREX JPY : JPY=X

# pylint: disable=too-many-function-args
@Exporter.route("/modelJPY=X",methods=['POST','GET'])
def ScrapingJPY():
    print("****************")
    crypto = request.values.get("crypto") 
    print(crypto)
    #Setting the end date to today
    end = datetime.today()

    #Start date set to one year back
    start = datetime(2018,1,1)
    #using yahoo finance to grab cryptocurrency data
    bit_data=data.DataReader(crypto ,'yahoo',start,end)
    bit_data =pd.DataFrame(bit_data)

    group=bit_data.groupby("Date")
    dataa=group['Close'].mean()
    dataa = pd.DataFrame(dataa)
    close_train=dataa.iloc[:len(dataa)-30]
    close_test=dataa.iloc[len(close_train):]
    
    #feature scalling (set values between 0-1)
    close_train=np.array(close_train)
    close_train=close_train.reshape(close_train.shape[0],1)
    scaler=MinMaxScaler(feature_range=(0,1))
    close_scaled=scaler.fit_transform(close_train)
    timestep=50
    x_train=[]
    y_train=[]

    for i in range(timestep,close_scaled.shape[0]):
        x_train.append(close_scaled[i-timestep:i,0])
        y_train.append(close_scaled[i,0])

    x_train,y_train=np.array(x_train),np.array(y_train)
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1) #reshaped for RNN

    model=Sequential()
    model.add(LSTM(10,input_shape=(None,1),activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    model.fit(x_train,y_train,epochs=50,batch_size=32)
    inputs=dataa[len(dataa)-len(close_test)-timestep:]
    inputs=inputs.values.reshape(-1,1)
    inputs=scaler.transform(inputs)
    x_test=[]
    for i in range(timestep,inputs.shape[0]):
        x_test.append(inputs[i-timestep:i,0])
    x_test=np.array(x_test)
    x_test= x_test.reshape(x_test.shape[0],x_test.shape[1],1)
    predicted_data=model.predict(x_test)
    predicted_data=scaler.inverse_transform(predicted_data)
    pred = predicted_data[29]
    pred = pred.tolist()

    json_data = json.dumps(pred)
    return json_data

##############################################################################################
##############################################################################################
################################################################################################

# CRYPTO : 

# REAL PRICE CRYPTO : BTC , ETH , BCH (Bitcoin Cash) , USDT (Tether) ,ADA (CARDANO)
#https://coinmarketcap.com/all/views/all/

AV_API_KEY = '8V9AQOQL5AZ2KZPM'


@Exporter.route("/realCrypto",methods=['POST','GET'])
def realCrypto():
   
    crypto = request.values.get("crypto") 
    
    resp = requests.get('https://www.alphavantage.co/query', params={
    'function': 'CURRENCY_EXCHANGE_RATE',
    'from_currency': crypto,
    'to_currency': 'USD',
    'apikey': AV_API_KEY})
    Bitcoin =  resp.json()
    real = Bitcoin["Realtime Currency Exchange Rate"]
    
    price = real["5. Exchange Rate"]
    t = real["6. Last Refreshed"]
    d = dict()
    d['price today'] = price
    d['time'] = t
    return d

   


###########################################################
#Price Actions today : 

@Exporter.route("/realForex",methods=['POST','GET'])
def realForex():
    symbole1 = request.values.get("symbole1") 
    
    resp = requests.get('https://www.alphavantage.co/query', params={
    'function': 'CURRENCY_EXCHANGE_RATE',
    'from_currency': symbole1,
    'to_currency': 'USD' ,
    'apikey': AV_API_KEY})
    forex =  resp.json()
    
    real = forex["Realtime Currency Exchange Rate"]
    from_currency = real["2. From_Currency Name"]
    to_currency = real["4. To_Currency Name"]
    price = real["5. Exchange Rate"]
    t = real["6. Last Refreshed"]
    d = dict()
    d['from_currency'] = from_currency
    d['to_currency'] = to_currency
    d['price today'] = price
    d['time'] = t
    return d

#################################################################################################
###Malek
@Exporter.route('/powerbi', methods=['POST','GET'])
def powerbi():
    
    new_freq = request.get_data()
    print(new_freq)
    new_data=new_freq.decode('utf-8')

    ### Scraping
    start = datetime(2010,1,1)
    end=datetime.today()
    #df=pd.DataFrame([data.DataReader(new_data,'yahoo',start,end)['Adj Close']]).T
    df=data.DataReader(new_data ,'yahoo',start,end)
    df.to_csv('PowerBI/A.csv')

    print('Kamell')
    return '''<h3>POWERBI Jawou behi !! </h3>'''
###############################################################
## Malek statique

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pandas_datareader import data
import numpy as np
import seaborn as sns
plt.style.use('fivethirtyeight')
import json
from sklearn.cluster import KMeans 
#from scipy import cluster
import asyncio
import time
from flask import jsonify
def scraping_Adj_Close(lst):
    start = datetime(2010,1,1)
    end=datetime.today()

    df=pd.DataFrame([data.DataReader(i,'yahoo',start,end)['Adj Close'] for i in lst]).T

     # Supprimer NaN avec les methodes :
     # On remplit les valeurs vides par les valeurs suivantes 
    df = df.fillna(method='ffill')

    # On remplit les valeurs vides par les valeurs précedents 
    df = df.fillna(method="bfill")

    df.columns=lst
    return df

# Simulation de portfolio optimale 

def calc_portfolio_perf(weights, mean_returns, cov, rf):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio
# To ignore numpy errors:
#     pylint: disable=E1101
def simulate_random_portfolios(num_portfolios, mean_returns, cov, rf,lst):
    # les noms d'entreprise que je travaille avec dans df
    lst3=scraping_Adj_Close(lst).columns.to_list()

    results_matrix = np.zeros((len(mean_returns)+3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, sharpe_ratio = calc_portfolio_perf(weights, mean_returns, cov, rf)
        results_matrix[0,i] = portfolio_return
        results_matrix[1,i] = portfolio_std
        results_matrix[2,i] = sharpe_ratio
        #iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results_matrix[j+3,i] = weights[j]
            
    results_df = pd.DataFrame(results_matrix.T,columns=['ret','stdev','sharpe'] + [ticker for ticker in lst3])

    return results_df

def Optimization_portfo(lst,risque,num_simulation):
    df=scraping_Adj_Close(lst)
    mean_returns = df.pct_change().mean()
    cov = df.pct_change().cov()

    num_portfolios = num_simulation
    rf = risque
    results_frame = simulate_random_portfolios(num_portfolios, mean_returns, cov, rf, lst) 

    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
    
    #locate positon of portfolio with minimum standard deviation
    #min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
    ##############
    # convert to json
    #result = results_frame.to_json(orient="split")
    #results_frame1 = json.loads(result)
    # convert to json
    result = (max_sharpe_port * 100).to_json(orient="split")
    
    #max_sharpe_port1 = json.loads(result)
    # convert to json
    #result = min_vol_port.to_json(orient="split")
    #min_vol_port1 = json.loads(result)

    return result


@Exporter.route('/test', methods=['POST','GET'])
def test():
    new_freq = request.get_data()
    print(new_freq)
    
    ###### Cleaning parametere 
    #start = '['
    #end = ']'
    #dataCleaning=data[data.find(start)+len(start):data.rfind(end)]
    
    #dataCleaning = dataCleaning.replace('\"', '')
    #lst=dataCleaning.split(',')
    #lst = ' '.join(lst).replace('\\','').split()
    #print(lst)
    ##############
    l = []
    lst2=['FB','GOOG','ASFT','ADBE']
    res=Optimization_portfo(lst2,0.0,10000)
    l = res.split(',')

    print('heeeeeeeeeeeeeeeeeeeeee')
    print(l)
    print(len(l))
    a = l[11] + "%" + "--" +  l[12] + "%" + "--" + l[13] + "%" + "--" + l[14] + "%" 
    resul = json.dumps(a)
    print(resul)
    return resul



if __name__ == "__main__":        # on running python app.py
    Exporter.run(debug=False, host='0.0.0.0')  