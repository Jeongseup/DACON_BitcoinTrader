import numpy as np
import pandas as pd
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.preprocessing import KBinsDiscretizer

### ------------ Data preprocess part ------------ ###


def coin_index_export(input_array, coin_num):
    ''' 코인별 인덱스 뽑기 '''
    index = []
    sample_id_len = input_array.shape[0]
    coin_num_col = 0 

    for sample_id in range(sample_id_len):
        if input_array[sample_id, 0, coin_num_col] == coin_num:
            #print(sample_id)
            index.append(sample_id)
    return index


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

def simple_exponetial_smoothing(arr, alpha=0.3):
    
    y_series = list()
    
    for temp_arr in arr:
        target_series = temp_arr[:, 1].reshape(-1) # open col is 1 index

        smoother = SimpleExpSmoothing(target_series, initialization_method="heuristic").fit(smoothing_level=0.3,optimized=False)
        smoothing_series = smoother.fittedvalues
        
        y_series.append(smoothing_series)
    
    return np.array(y_series)

# ================================================= #

def simple_exponetial_smoothing_fory(arr, alpha=0.3):
    
    y_series = list()

    for temp_arr in arr:
        target_series = temp_arr[:, 1].reshape(-1) # open col is 1 index

        smoother = SimpleExpSmoothing(target_series, initialization_method="heuristic").fit(smoothing_level=alpha,optimized=False)
        smoothing_series = smoother.fittedvalues

        y_series.append(smoothing_series)
            
    return np.array(y_series)

# ================================================= #

def simple_exponetial_smoothing_forX(arr, alpha=0.3):
    
    # initialization
    sample_size = int(arr.shape[0])
    time_size = int(arr.shape[1])
    feature_size = int(arr.shape[2])
    
    # create empty array
    smoothing_arr = np.zeros((sample_size, time_size, feature_size - 1))

    for idx, temp_arr in enumerate(arr):
        for col in range(1, feature_size): # open col is 1 index
            if col < 5:

                temp_series = temp_arr[:, col].reshape(-1) 
                smoother = SimpleExpSmoothing(temp_series, initialization_method="heuristic").fit(smoothing_level=0.3,optimized=False)
                temp_smoothing_series = smoother.fittedvalues
                smoothing_arr[idx, :, col-1] = temp_smoothing_series

            else:
                
                pass_series = temp_arr[:, col].reshape(-1)
                smoothing_arr[idx, :, col-1] = pass_series

    return smoothing_arr

# ================================================= #

def moving_average(arr, window_size = 20):
    
    #length = ma 몇 할지
    length = window_size
    ma = np.zeros((arr.shape[0], arr.shape[1] - length, arr.shape[2]))

    for idx in range(arr.shape[0]):
        for i in range(length, arr.shape[1]):
            for col in range(arr.shape[2]):
                ma[idx, i-length, col] = arr[idx,i-length:i, col].mean() #open
            
    return ma[:, :, 1] # open col is 1

def time_split(input_array, split_size = 6):
    ''' n분봉으로 데이터 나누는 함수 '''
    # origin size define
    index_size = input_array.shape[0]
    origin_time_size = input_array.shape[1]
    variable_size = input_array.shape[2]

    # new array size define
    new_time_size = int(origin_time_size/split_size) # 1380 / 6
    new_array = np.zeros((index_size, new_time_size, variable_size))

    for idx in range(index_size):
        for time_idx in range(new_time_size):
            

            first_time_idx = time_idx * split_size
            last_time_idx = ((time_idx+1) * split_size) -1

            new_array[idx, time_idx, 0] = input_array[idx, first_time_idx, 0] #coin_num
            new_array[idx, time_idx, 1] = input_array[idx, first_time_idx, 1] #open
            
            new_array[idx, time_idx, 2] = np.max(input_array[idx, first_time_idx:last_time_idx, 2]) #high
            new_array[idx, time_idx, 3] = np.min(input_array[idx, first_time_idx:last_time_idx, 3]) #low

            new_array[idx, time_idx, 4] = input_array[idx, last_time_idx, 4] #close

            new_array[idx, time_idx, 5] = np.sum(input_array[idx, first_time_idx:last_time_idx, 5]) #etc
            new_array[idx, time_idx, 6] = np.sum(input_array[idx, first_time_idx:last_time_idx, 6]) #etc
            new_array[idx, time_idx, 7] = np.sum(input_array[idx, first_time_idx:last_time_idx, 7]) #etc
            new_array[idx, time_idx, 8] = np.sum(input_array[idx, first_time_idx:last_time_idx, 8]) #etc
            new_array[idx, time_idx, 9] = np.sum(input_array[idx, first_time_idx:last_time_idx, 9]) #etc

    return new_array

### ---------------------------------------------- ###

def train_val_test_spliter(arr):
    
    n = len(arr)
    num_features = arr.shape[2] - 1
    
    train_arr = arr[0:int(n*0.8), :, :]
    val_arr = arr[int(n*0.8):, :, :]
    
    n2 = len(train_arr) + len(val_arr)
    
    print(
    f'''
    ======================================================
    Origin length is {n}, then total split length is {n2}
    ======================================================
    train length is {train_arr.shape},
    val length is {val_arr.shape},
    num_features is ({num_features})
    '''
    )
    
    return train_arr, val_arr

# study 해보고 제대로 안 쓴 preprocess
def each_coin_normalization(train_x_arr):
    ''' 함수 설명 : 코인별 데이터 정규화 '''
    
    # 유니크 코인 번호
    unique_coin_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    #create empty scaled list
    scaled_train_x_arr = np.zeros((train_x_arr.shape[0], train_x_arr.shape[1], train_x_arr.shape[2]))
    
    for temp_coin_num in unique_coin_index:
        # 유니크 코인 번호 중 한 코인 번호씩 해당 코인에 맞는 인덱스 추출
        # ex) if temp_coin_num is 0, temp_coin_index = [3, 7, 8, 14...]
        temp_coin_index = coin_index_export(train_x_arr, temp_coin_num)
        
        # temp coin num array export
        temp_x_arr = train_x_arr[temp_coin_index]
        
        # initialization
        num_sample   = temp_x_arr.shape[0] # sample dim
        num_sequence = temp_x_arr.shape[1] # time-sequence dim
        num_feature  = temp_x_arr.shape[2] # feature dim

        # create emptpy scaler
        temp_scaler = MinMaxScaler()
        
        # 시계열을 선회하면서 피팅합니다
        print('Current normalizing coin number is {}'.format(temp_coin_num))
        for temp_sample, temp_index in enumerate(temp_coin_index):
            temp_scaler.partial_fit(temp_x_arr[temp_sample, :, 5:]) # open =1, high = 2, low=3, close=4, volume=5 ~...

        # 스케일링(변환)합니다.
        for temp_sample, temp_index in enumerate(temp_coin_index):
            scaled_train_x_arr[temp_index, :, 5:] = temp_scaler.transform(temp_x_arr[temp_sample, :, 5:]).reshape(1, num_sequence, 5)
            scaled_train_x_arr[temp_index, :, :5] = temp_x_arr[temp_sample, :, :5]
            
        # save scaler for test arr
        dir_name = './scaler'
        file_name = f'coin_{temp_coin_num}_scaler.pkl'
        save_path = os.path.join(dir_name, file_name)
        joblib.dump(temp_scaler, save_path)
        
    
    print("Each coin normalization, Complete!")
    return scaled_train_x_arr


def kbin_discretizer(input_array):

    kb = KBinsDiscretizer(n_bins=10, strategy='uniform', encode='ordinal')
    processed_data = np.zeros((input_array.shape[0], input_array.shape[1], 1))
    
    for i in range(input_array.shape[0]):
        # coin_index_export args : (input_array, coin_num)
        globals()['processing_array{}'.format(i)] = input_array[i,:,1]
        
        #globals()['outliery_array{}'.format(i)] = train_y_array[outlier[i],:,1]
        kb.fit(globals()['processing_array{}'.format(i)].reshape(input_array.shape[1],1))
        globals()['processed_fit{}'.format(i)] = kb.transform(globals()['processing_array{}'.format(i)].reshape(input_array.shape[1],1))
        
        #globals()['outliery_fit{}'.format(i)] = kb.transform(globals()['outliery_array{}'.format(i)].reshape(120,1))
        processed_data[i,:,:] = globals()['processed_fit{}'.format(i)]
        
    return processed_data


def outlier_detecter(raw_y_arr, outlier_criteria = 0.05):

    open_arr = raw_y_arr[:, :, 1] #open col is 1

    outlier_list = []
    openrange_list = []

    for idx, temp_arr in enumerate(open_arr):
    
        temp_min = temp_arr.min()
        temp_max = temp_arr.max()
        temp_arr_range = temp_max - temp_min
        openrange_list.append(temp_arr_range)

        if temp_arr_range > outlier_criteria:
            outlier_list.append(idx)
            print(f'{idx}번째 open series is outlier sample!')
            print(f'temp array range is {temp_arr_range:.3}\n')
            

    return outlier_list, np.array(openrange_list)

# ================================================= #
# ====== Generating Dataset ====== #

def data_generate(dataframe, x_frames, y_frames, print_mode = False):

    ''' 설명 생략 '''

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
        temp_df = grouped_df.get_group(temp_sample_id)

        # 한 샘플당 몇 개의 arrset가 나오는 지 확인
        count = 0
        split_length = len(temp_df) - (x_frames + y_frames) + 1
        
        ''' 한 샘플 내 데이터 split loop '''
        for time_idx in range(split_length):
            
            # index 변경
            time_idx += x_frames
            
            # temp_data select
            temp_arr = temp_df.iloc[time_idx - x_frames : time_idx + y_frames, 3:].values

            # get values
            temp_x = temp_arr[:x_frames, :]
            temp_y = temp_arr[x_frames:, :]

#             # 2d to 3d -> (255, 12) to (1, 255, 12) / (120, 12) to (1, 120, 12)
#             temp_3d_x = np.expand_dims(temp_2d_x, axis = 0)
#             temp_3d_y = np.expand_dims(temp_2d_y, axis = 0)
            
            # appending
            X.append(temp_x)
            y.append(temp_y)
            
            # counter printing
            count += 1
            if (count == split_length) & (print_mode == True):
                print(f'현재 sample id : {temp_sample_id}')
                print(f'{temp_sample_id}번째 sample의 생성 array수 : {count}')
            
    return np.array(X), np.array(y)

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


def targetframes_predict(model, dataframe, test_id = 1207, x_frames = 60 , target_len = 120):

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


def coindata_merger(train_x_df, train_y_df, coin_num):
    
    # df에서 지정된 코인 데이터만 추출
    coin_num_x_df = train_x_df[train_x_df['coin_index'] ==  coin_num]
    coin_num_y_df = train_y_df[train_y_df['coin_index'] ==  coin_num]

    # y dataframe time value에 1380 씩 adding
    coin_num_y_df.time = coin_num_y_df.time.copy() + 1380

    # x,y df merge하고 sample_id와 time 순으로 sorting
    merged_df = pd.concat([coin_num_x_df, coin_num_y_df])
    merged_df = merged_df.sort_values(by = ['sample_id','time']).reset_index(drop=True)

    # sample_id series orderly indexing
    sample_id_series = merged_df.sample_id.value_counts().reset_index().rename(columns = {"index" : "sample_id"})
    reset_index_series = sample_id_series.iloc[:,:1].sort_values(by = ['sample_id']).reset_index(drop=True)

    # coin index file export
    coin_index = reset_index_series.reset_index().set_index('sample_id')
    coin_index_name = f'./coin_{coin_num}_index.json'
    coin_index.to_json(coin_index_name, orient="table", indent=4)
    
    # dict index
    new_sample_dict = reset_index_series.reset_index().set_index('sample_id').to_dict()

    # sample_id value initialization
    merged_df['sample_id'] = merged_df['sample_id'].map(new_sample_dict['index'])

    merged_df.to_hdf('./data/merged_data.h5',  key = 'merged_df')
    return merged_df
