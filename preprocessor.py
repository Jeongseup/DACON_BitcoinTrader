import numpy as np
import pandas as pd

### ------------ Data preprocess part ------------ ###

def df2d_to_array3d(df_2d):

    feature_size = df_2d.iloc[:,2:].shape[1]
    time_size = len(df_2d.time.value_counts())
    sample_size = len(df_2d.sample_id.value_counts())
    array_3d = df_2d.iloc[:,2:].values.reshape([sample_size, time_size, feature_size])
    return array_3d


def getWeights_FFD(d, size, thres):
    ''' 함수 설명 : 실수 차분을 위한 get weights '''
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
    함수 설명 : 실수 차분
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


def FFD_smoothing(train_x_array):
    ''' 함수 설명 : 차분 데이터 뽑기 '''

    FFD_array = np.zeros((383, 1339, 1))

    for x in range(383):
        fdiff, w = fracDiff_FFD(train_x_array[x, :, 1], d=0.2, thres=0.002)
        FFD_array[x]= fdiff
    
    return FFD_array

### ---------------------------------------------- ###