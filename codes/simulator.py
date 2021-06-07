import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#pred_array : 예측정보 담긴 array, start_idx : submission의 start idx

def array_to_submission(pred_array, start_idx=0, increase_rate = 1.04):
    submission = np.zeros((pred_array.shape[0],3))

    for x in range(int(pred_array.shape[0])):
        #시작 인덱스 설정
        idx = int(start_idx + x)
        submission[x,0] = idx
        #예측값의 최고가에 따른 buy_quantity 결정
        high_price = np.max(pred_array[x,:])/pred_array[x,0]
        if high_price >=increase_rate:
            submission[x, 1] = 1
        #예측값의 최고가의 time_number
        sell_time = int(np.argmax(pred_array[x,:]))
        submission[x, 2] = sell_time
    submission = pd.DataFrame(submission)
    submission.columns = ['sample_id','buy_quantity', 'sell_time']
    return submission

#y_array : (val set에서 사용된 idx, open가)가 들어있는 2차원 array) or 함수 df2d_to_answer의 return 값
#y_array.shape = (idxlen,120)
#submission : 함수 arrayy_to_submission의 return 값
def COIN(y_array, submission):
    # 2차원 데이터프레임에서 open 시점 데이터만 추출하여 array로 복원
    # sample_id정보를 index에 저장
    # y_array= df2d_to_answer(y_df)

    # 초기 투자 비용은 10000 달러
    total_money = 10000
    total_money_list = []


    for row_idx in range(submission.shape[0]):
        sell_time = int(submission.loc[row_idx, 'sell_time'])
        buy_price = y_array[row_idx, 0]
        sell_price = y_array[row_idx, sell_time]
        buy_quantity = submission.loc[row_idx, 'buy_quantity'] * total_money
        residual = total_money - buy_quantity
        ratio = sell_price / buy_price
        total_money = buy_quantity * ratio * 0.9995 * 0.9995 + residual
        total_money_list.append(total_money)

    return total_money, total_money_list

# 그냥 3차원 어레이에서 원하는거 꺼내쓰는게 편하지 않나 싶음,,
def df2d_to_answer(df_2d):
    # valid_y_df로부터 open 가격 정보가 포함된 [샘플 수, 120분] 크기의 2차원 array를 반환하는 함수
    time_size = len(df_2d.time.value_counts())
    sample_size = len(df_2d.sample_id.value_counts())
    array_2d = df_2d.open.values.reshape([sample_size, time_size])
    return array_2d

def investing_histroy(exp_name, total_momey_list, total_momey, save_mode):
    

    # 투자 히스토리
    plt.plot(total_momey_list)
    plt.title(exp_name + " Investing History", fontsize = 12, y = 1.02)
    plt.ylabel("total asset")
    plt.xlabel("trial time")
    plt.text(60, 10000, total_momey)
    plt.grid(True)

    if save_mode is True:
        plt.savefig(f'./images/{exp_name}.png')
    plt.show()
 