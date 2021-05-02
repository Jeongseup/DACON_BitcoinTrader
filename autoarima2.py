import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import gc
import os.path
import time
from datetime import timedelta, datetime
from dateutil import parser
from tqdm import tqdm
import copy
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape, GRU
from pmdarima import auto_arima
import pandas_profiling as pp
from keras.layers.wrappers import TimeDistributed
from numpy import array
import random as ran
from keras.utils import to_categorical
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.layers import *
from keras.callbacks import ModelCheckpoint,Callback
from pandas import DataFrame


test_x_df=pd.read_csv('./data/test_x_df.csv')
sample_submission=pd.read_csv('./data/sample_submission.csv')
test_x_df = test_x_df.astype('float')
sample_submission=sample_submission.astype('float')


def df2d_to_array3d(df_2d):

    feature_size = df_2d.iloc[:,2:].shape[1]
    time_size = len(df_2d.time.value_counts())
    sample_size = len(df_2d.sample_id.value_counts())
    array_3d = df_2d.iloc[:,2:].values.reshape([sample_size, time_size, feature_size])
    return array_3d

test_x_array = df2d_to_array3d(test_x_df)


def getWeights_FFD(d, size, thres):
    w = [1.]  # w의 초깃값 = 1

    for k in range(1, size):

        w_ = -w[-1] * (d - k + 1) / k  # 식 2)를 사용했다.

        if abs(w[-1]) >= thres and abs(w_) <= thres:

            break

        else:

            w.append(w_)

    # w의 inverse

    w = np.array(w[::-1]).reshape(-1, 1)

    return w


def fracDiff_FFD(series, d, thres=0.002):
    '''

    Constant width window (new solution)

    Note 1: thres determines the cut-off weight for the window

    Note 2: d can be any positive fractional, not necessarily bounded [0,1]

    '''

    # 1) Compute weights for the longest series

    w = getWeights_FFD(d, series.shape[0], thres)

    width = len(w) - 1

    # 2) Apply weights to values

    df = []

    seriesF = series

    for iloc in range(len(w), seriesF.shape[0]):
        k = np.dot(w.T[::-1], seriesF[iloc - len(w):iloc])
        df.append(k)

    df = np.array(df)

    return df, w



def array_to_submission(x_array, pred_array):

    submission = pd.DataFrame(np.zeros([pred_array.shape[0], 2], np.int64),
                              columns=['buy_quantity', 'sell_time'])
    submission = submission.reset_index()
    submission.loc[:, 'buy_quantity'] = 0.1

    buy_price = []
    for idx, sell_time in enumerate(np.argmax(pred_array, axis=1)):
        buy_price.append(pred_array[idx, sell_time])
    buy_price = np.array(buy_price)
    # 105% 이상 상승한하고 예측한 sample에 대해서만 100% 매수
    submission.loc[:, 'buy_quantity'] = (buy_price - pred_array[:,0] > 0.06) * 1
    # 모델이 예측값 중 최대 값에 해당하는 시간에 매도
    submission['sell_time'] = np.argmax(pred_array, axis=1)
    submission.columns = ['sample_id', 'buy_quantity', 'sell_time']
    return submission



def COIN(y_df, submission, df2d_to_answer=df2d_to_answer):
    # 2차원 데이터프레임에서 open 시점 데이터만 추출하여 array로 복원
    # sample_id정보를 index에 저장
    y_array, index = df2d_to_answer(y_df)

    # index 기준으로 submission을 다시 선택
    submission = submission.set_index(submission.columns[0])
    submission = submission.iloc[index, :]

    # 초기 투자 비용은 10000 달러
    total_momey = 10000  # dolors
    total_momey_list = []

    # 가장 처음 sample_id값
    start_index = submission.index[0]
    for row_idx in submission.index:
        sell_time = submission.loc[row_idx, 'sell_time']
        buy_price = y_array[row_idx - start_index, 0]
        sell_price = y_array[row_idx - start_index, sell_time]
        buy_quantity = submission.loc[row_idx, 'buy_quantity'] * total_momey
        residual = total_momey - buy_quantity
        ratio = sell_price / buy_price
        total_momey = buy_quantity * ratio * 0.9995 * 0.9995 + residual
        total_momey_list.append(total_momey)

    return total_momey, total_momey_list


# in testset
new_array = np.zeros((535,1339,1))

#new_array[0] = fdiff
for x in range(535):
    fdiff, w = fracDiff_FFD(test_x_array[x,:,1], d=0.2, thres=0.002)
    new_array[x]= fdiff
new_array.shape

test_pred_array =  np.zeros([new_array.shape[0], 120])

for idx in tqdm(range(new_array.shape[0])):
    try:

        x_series = new_array[idx,:]
        x_series = x_series.reshape(1339,)
        pp=auto_arima(x_series, stepwise=True, error_action='ignore', seasonal=False)
        model = ARIMA(x_series, order=pp.order)
        fit = model.fit()
        preds = fit.predict(1, 120, typ='levels')
        valid_pred_array[idx,:] = preds# - (preds[0]-x_series[-1])

    except:
        print(idx, " 샘플은 수렴하지 않습니다.")
        # ARIMA의 (p,d,q) 값이 (5,1,1), (4,1,1)에서 수렴하지 않을 경우
        # 모두 0으로 채움
        pass



for x in range(535):
    valid_pred_arrayy[x] = valid_pred_array[x]/valid_pred_array[x,0]


submission = array_to_submission(test_x_array, valid_pred_arrayy)
submission.loc[0,['sample_id']] = 7661

for i in range(535):
    submission.loc[i, ['sample_id']] = 7661+i

submission.to_csv("submission_exp1.csv", index = False)