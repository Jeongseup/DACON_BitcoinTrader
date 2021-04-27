import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 

def train_val_test_split(dataframe, split_size, test_days):
    ''' train, validation, test data split '''
    
    orgin_len = len(dataframe)

    # test split
    test_df = dataframe.iloc[-test_days:]
    test_len =  len(test_df)
    dataframe = dataframe.iloc[:-test_days]

    # the rest of data split train and validaiton
    train_len = int((len(dataframe) * split_size))
    validation_len = len(dataframe) - train_len
    print(train_len, validation_len)
    train_df = dataframe.iloc[:train_len] 
    validation_df = dataframe.iloc[train_len:] 

    # split lengths check
    if orgin_len == (train_len + validation_len + test_len):
        print("Data is split normally.\n")
        print("train_len is {} days, validation_len is {} days, test_len is {} days.\n".format(train_len ,validation_len, test_len))
    else:
        print("Data is split abnormally, Please Check your split size\n")
        pass

    return train_df, validation_df, test_df


def standardnorm_scaler_v1(train_df, validation_df, test_df):
    ''' scaling version1 : 거래량 데이터만 넣을 경우 '''
    
    temp1_df = train_df.copy()
    temp2_df = validation_df.copy()
    temp3_df = test_df.copy()
    
    # create scaler and dafaframes
    scaler = StandardScaler()

    # scaler fitting
    scaler = scaler.fit(train_df)
    
    # transforms each dataframes
    scaled_tr = scaler.transform(train_df)
    scaled_train_df = pd.DataFrame(scaled_tr, columns = temp1_df.columns, index = list(temp1_df.index.values))
    
    scaled_val = scaler.transform(validation_df)
    scaled_val_df = pd.DataFrame(scaled_val, columns = temp2_df.columns, index = list(temp2_df.index.values))
    
    scaled_te = scaler.transform(test_df)
    scaled_test_df = pd.DataFrame(scaled_te, columns = temp3_df.columns, index = list(temp3_df.index.values))
    
    return scaled_train_df, scaled_val_df, scaled_test_df, scaler


def standardnorm_scaler_v2(train_df, validation_df, test_df, output_dim):
    '''  scaling version2 : 거래량 데이터외의 데이터도 넣을 경우 '''
    
    # create scaler and dafaframes
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()

    # scaler fitting
    scaler1 = scaler1.fit(train_df.iloc[:, :output_dim])
    scaler2 = scaler2.fit(train_df.iloc[:, output_dim:])

    # transforms each dataframes
    train_df.iloc[:, :output_dim] = scaler1.transform(train_df.iloc[:, :output_dim])
    train_df.iloc[:, output_dim:] = scaler2.transform(train_df.iloc[:, output_dim:])
    
    validation_df.iloc[:, :output_dim] = scaler1.transform(validation_df.iloc[:, :output_dim])
    validation_df.iloc[:, output_dim:] = scaler2.transform(validation_df.iloc[:, output_dim:])
    
    test_df.iloc[:, :output_dim] = scaler1.transform(test_df.iloc[:, :output_dim])
    test_df.iloc[:, output_dim:] = scaler2.transform(test_df.iloc[:, output_dim:])

    return train_df, validation_df, test_df, scaler1, scaler2


class DatasetGenerater(object):
    ''' generate dataset for encoder & decoder frame '''
    
    def __init__(self, dataframe, x_frames, y_frames):

        '''    
        : param dataframe:   Raw data to load for making dataset
        : param x_frames:    the number of input day size
        : param y_frames:    the number of output day size
        '''
        self.dataframe = dataframe
        self.x_frames = x_frames
        self.y_frames = y_frames

        check_null = self.dataframe.isna().sum().sum()
        print("null clear") if check_null == 0 else print("null exist")
        
        
    def __len__(self):
        return len(self.dataframe) - (self.x_frames + self.y_frames) + 1
    
    
    def __getitem__(self, idx):
        idx += self.x_frames
        
        dataframe = self.dataframe.iloc[idx-self.x_frames:idx+self.y_frames]
        dataframe = dataframe.values
        
        X = dataframe[:self.x_frames]
        y = dataframe[self.x_frames:]
        
        return X, y