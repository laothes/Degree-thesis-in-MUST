import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout

from numpy.fft import rfft,irfft



def createXY(dataset, n_past,n_fu):
    '''
    The OT is decomposed as y= OT of the NTH point and X= OT of the first h time points
    '''
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)-n_fu):
        dataX.append(dataset[i - n_past:i])
        dataY.append([dataset[i+n_fu]])
    return np.array(dataX), np.array(dataY)

def denoise_fft(data,threshold,is_draw = False):
    '''
    data dimension:1
    threshold: filter out those value under threshold
    is_draw
    '''
    yf = rfft(data)
    yf_abs = np.abs(yf) 
    indices = yf_abs > threshold
    yf_clean = indices * yf
    new_f_clean = irfft(yf_clean)
    if is_draw == True:
        plt.plot(new_f_clean)
        plt.show()
    return new_f_clean

def input_fit_lstm(X):
    '''
    Change the 2D input to lstm's 3D input format
    '''
    return np.reshape(X, (X.shape[0],1,X.shape[1]))

def input_fit_cnn(X):
    '''
    Change the 2D input to cnn 3D input format
    '''
    return np.reshape(X, (X.shape[0],X.shape[1],1))

def data_dvd(data,size = [0.6,0.2,0.2]):
    '''
    Split dataset
    '''
    test_split_1 = round(len(data) * (1-size[0]))
    test_split_2 = round(len(data) * size[-1])
    return data[:-test_split_1],data[-test_split_1:-test_split_2],data[-test_split_2:]

def build_model():
    model = Sequential()
    '''
    code for CNN
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu',
                      input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(MaxPooling1D(pool_size=1))
    
    '''
    model.add(LSTM(200,return_sequences=True)) 
    model.add(LSTM(200,return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse',metrics=['mae'],optimizer='Adam')
    return model

# MAE
def mae(target, predict):
    return (abs(target-predict)).mean()
 
# MSE
def mse(target, predict):
    return ((target - predict)**2).mean()
 
# RMSE
def rmse(target,predict):
    return np.sqrt(((predict - target) ** 2).mean())
 
# R2
def r2(target, predict):
    SSE = sum((target - predict) ** 2)
    SST = sum((target - (target).mean()) ** 2)
    return 1 - (SSE / SST)

def res_show(target_, predict_,draw_error = False):
    Mae = mae(target_, predict_)
    Mse = mse(target_, predict_)
    Rmse = rmse(target_, predict_)
    R2 = r2(target_, predict_)
    print('mae:',Mae,'mse:',Mse, 'rmse:',Rmse,'r2:',R2)
    if draw_error == False:
        plt.plot(target_, color='red', label='Real OT')
        plt.plot(predict_, color='blue', label='Predicted OT')
        plt.title('OT Prediction')
        plt.xlabel('Time')
        plt.ylabel('OT')
        plt.legend()
        plt.show()
    else:
        pred_error = target_ - predict_
        plt.plot(pred_error, color='green', label='Pred Error')
        plt.title('Error')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

# input dataset
df_o = pd.read_csv(".\ETT_h1\ETTh1.csv",index_col = 0)
df_v = pd.read_csv(".\ETT_h1\VMD26.csv")
df_f = df_o.iloc[:,:-1]
df_true = df_o.iloc[:,-1]

# train :test = 6:2:2
df_f_train,df_f_val,df_f_test = data_dvd(df_f)
df_v_train,df_v_val,df_v_test = data_dvd(df_v)

# normalization
scaler_f = MinMaxScaler(feature_range=(0,1))
scaler_v = MinMaxScaler(feature_range=(0,1))
df_f_tr_s = scaler_f.fit_transform(df_f_train)
df_v_tr_s = scaler_v.fit_transform(df_v_train)
df_f_va_s = scaler_f.transform(df_f_val)
df_v_va_s = scaler_v.transform(df_v_val)
df_f_te_s = scaler_f.transform(df_f_test)
df_v_te_s = scaler_v.transform(df_v_test)

# parameters
vmd_idx = len(df_v.loc[0])
threshold = 5
n_past = 24
n_fu = 0
epochs = 100
batch_size = 24 * 30

# train
model_lst = []
for i in range(vmd_idx):
    print(i,vmd_idx)
    
    #fft
    tr_clean = denoise_fft(df_v_tr_s[:,i],threshold=threshold)
    va_clean = denoise_fft(df_v_va_s[:,i],threshold=threshold)
    te_clean = denoise_fft(df_v_te_s[:,i],threshold=threshold)
    
    X_tr,y_train = createXY(tr_clean, n_past = n_past, n_fu = n_fu)
    X_va,y_val = createXY(va_clean, n_past = n_past, n_fu = n_fu)
    X_te,y_test = createXY(te_clean, n_past = n_past, n_fu = n_fu)
    X_train = np.hstack((df_f_tr_s[n_past+n_fu:,:],X_tr))
    X_val = np.hstack((df_f_va_s[n_past+n_fu:,:],X_va))
    X_test = np.hstack((df_f_te_s[n_past+n_fu:,:],X_te))
    X_train = input_fit_lstm(X_train)
    X_val = input_fit_lstm(X_val)
    X_test = input_fit_lstm(X_test)
    
    '''
    code for cnn
    X_train = input_fit_cnn(X_train)
    X_val = input_fit_cnn(X_val)
    X_test = input_fit_cnn(X_test)
    
    '''
    model = build_model()
    model.fit(X_train, y_train,validation_data=(X_val, y_val), epochs=epochs,batch_size =batch_size)
    model_lst.append(model)

# prediction
pred = []
for i in range(vmd_idx):
    prediction = model_lst[i].predict(X_test_lst[i])
    pred.append(prediction)

pred = np.array(pred).T# n * vmd_idx
pred = np.reshape(pred,(-1,pred.shape[2])) #Remove the outermost layer

pred_re = scaler_v.inverse_transform(pred)
pred_sum = np.array([[sum(pred_re[i]) for i in range(len(pred_re))]]).T

Tr = df_true[-len(pred_sum):]
Tr = np.array([Tr]).T

res_show(Tr, pred_sum,draw_error = False)

res_show(Tr, pred_sum,draw_error = True)

for i in range(model_lst):
    model_lst[i].save('LSTM'&i, save_format='tf')
print('F')
