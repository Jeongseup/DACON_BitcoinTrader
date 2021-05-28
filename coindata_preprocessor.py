import pandas as pd
import numpy as np

def coindata_preprocessor(train_x_df, train_y_df, coin_num):
    
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