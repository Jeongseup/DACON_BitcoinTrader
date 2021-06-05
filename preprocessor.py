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

# study 해보고 제대로 안 쓴 preprocess

# ====== Generating Dataset ====== #

def open_data_generate(dataframe, col_name, x_frames, y_frames, print_mode = False):

    '''
    print mode = True로 바꾸면 데이터 어떻게 분할되는지 볼 수 있음
    
    example) 
    x_frames = 60
    train_x, train_y = open_data_generate(train_df,col_name ='open', x_frames = x_frames, y_frames = 1, print_mode = False)

    
    데이터 전처리 후 shape
    train x shape is (288000, 60) 
    train y shape is (288000, 1)
    '''

    # grouping
    grouped_df = dataframe.groupby('sample_id')
    
    # export unique sample ids
    unique_sample_id_list = grouped_df.sample_id.unique()

    # create new lists
    X, y = list(), list()

    ''' 샘플 하나 선택 loop '''
    for sample_id in unique_sample_id_list:
        
        # get one sample_id in sample list
        temp_sample_id = sample_id.item()

        # get one group by temp_sample_id
        temp_series = grouped_df.get_group(temp_sample_id)[col_name]

        # 한 샘플당 몇 개의 arrset가 나오는 지 확인
        count = 0
        split_length = len(temp_series) - (x_frames + y_frames) + 1
        
        ''' 한 샘플 내 데이터 split loop '''
        for time_idx in range(split_length):
            
            # index 변경
            time_idx += x_frames
            
            # temp_data select
            temp_arr = temp_series[time_idx - x_frames : time_idx + y_frames]

            # get values
            temp_x, temp_y = temp_arr.iloc[:x_frames].values, temp_arr.iloc[x_frames:].values

            # appending
            X.append(temp_x)
            y.append(temp_y)
            
            # counter printing
            count += 1
            if (count == split_length) & (print_mode == True):
                print(f'현재 sample id : {temp_sample_id}')
                print(f'{temp_sample_id}번째 sample의 생성 array수 : {count}')
            
    return np.array(X), np.array(y)


def targetframes_predict(model = model,dataframe = test_df, test_id = 1207, x_frames = 60 , target_len = 120):

    '''
    test_df의 test_id = 1207이 실제 데이터에서의 7657
    target_len는 반복 예측해야 할 구간 길이
    
    example)
    y_pred_arr, y_true_arr = targetframes_predict(model = model, dataframe = test_df, test_id = 1207, x_frames = x_frames , target_len = 120)
    '''
    
    # list 만들고, 마지막 데이터 30개 추출
    y_pred_list = list()
    y_true_list = dataframe[dataframe.sample_id == test_id].reset_index(drop=True).open[1380:].values
    
    x_input = dataframe[dataframe.sample_id == test_id].reset_index(drop=True).open[1380 - x_frames :1380].values

    for i in range(target_len):

        yhat = model.predict(x_input.reshape((1, x_frames, 1)), verbose=0)
        
        # list append
        y_pred = round(yhat.item(), 8)
        y_pred_list.append(y_pred)

        # exchage input data
        x_input = np.append(x_input, y_pred)
        x_input = np.delete(x_input, 0)


    return np.array(y_pred_list), np.array(y_true_list)
