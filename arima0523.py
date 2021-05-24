import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima


### --------------- Modeling part ---------------- ###

# 파일 경로
os.chdir('C:/Users/techtech/Downloads/open')

#파일 업로드
train_x_df=pd.read_csv('train_x_df.csv')
train_y_df=pd.read_csv('train_y_df.csv')

# 데이터 전처리 1 : dafaframe to array 
train_x_array = df2d_to_array3d(train_x_df)
train_y_array = df2d_to_array3d(train_y_df)

### 차분 데이터 뽑기###########
new_array = np.zeros((383,1339,1))

for x in range(383):
    fdiff, w = fracDiff_FFD(train_x_array[x, :, 1], d=0.2, thres=0.002)
    new_array[x]= fdiff

#미래 데이터 저장 array
valid_pred_array =  np.zeros([new_array.shape[0], 120])

#모델 돌리기 및 결과 저장
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

#구매량 및 판매 시점
valid_submission = array_to_submission(new_array, valid_pred_array, increase_rate = 0.4)

#투자 후 총 금액
valid_y_df = train_y_df[train_y_df.sample_id < 383]
total_momey, total_momey_list = COIN(valid_y_df,
                                     valid_submission)

print(total_momey)

# 투자 히스토리
plt.plot(total_momey_list)
plt.title("history")
plt.show()







#비차분 일반 아리마
new_arrayy = train_x_array[:383,:,1].reshape(383,1380,1)

#미래데이터 저장 array
valid_pred_arrayy =  np.zeros([new_arrayy.shape[0], 120])

#모델 돌리기 및 결과 저장
for idx in tqdm(range(new_arrayy.shape[0])):
    try:

        x_series = new_arrayy[idx,:]
        x_series = x_series.reshape(1380,)
        pp=auto_arima(x_series, stepwise=True, error_action='ignore', seasonal=False)
        model = ARIMA(x_series, order=pp.order)
        fit = model.fit()
        preds = fit.predict(1, 120, typ='levels')
        valid_pred_arrayy[idx,:] = preds# - (preds[0]-x_series[-1])

    except:
        print(idx, " 샘플은 수렴하지 않습니다.")
        # ARIMA의 (p,d,q) 값이 (5,1,1), (4,1,1)에서 수렴하지 않을 경우
        # 모두 0으로 채움
        pass

#구매량 및 판매 시점
valid_submission = array_to_submission(new_array, valid_pred_arrayy)

#투자 후 총 금액
valid_y_df = train_y_df[train_y_df.sample_id < 383]

total_momey, total_momey_list = COIN(valid_y_df,
                                     valid_submission)

print(total_momey)

# 투자 히스토리
plt.plot(total_momey_list)
plt.title("history")
plt.show()
