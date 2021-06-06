## 데이콘 비트코인 트레이더 시즌2 스터디
### index 
- Chapter. 1 - EDA
- Chapter. 2 - Season 1 pilot
- Chapter. 3 - Personal modeling prediction
- Chapter. 4 - Data preprocess
- Chapter. 5 - Results & Suggestion
.. Reference
___
### Chapter. 1 - EDA(Exploratory Data Analysis)
#### train_x_df EDA 과정 설명
* sample_id : 한 시퀀스 샘플, 한 시퀀스는 1380분의 시계열 데이터로 구성
 아래 예시
![resnet_image](./images/train_x_df_eda.png)

#### In one sample, dataset description
- X : 1380분(23시간)의 연속 데이터
- Y : 120분(2시간)의 연속 데이터
- 23시간 동안의 데이터 흐름을 보고 앞으로의 2시간 데이터를 예측하는 것
- sample_id는 7661개의 세트 구성, 각 세트는 독립적인 dataset 
- coin_index는 총 10개 종류로 구성(index number is 0 ~ 9)

#### 코인별 샘플 개수
- 각 코인별로 샘플 개수는 다름
- 9, 8번의 샘플 수가 가장 많음
![resnet_image](./images/coin_index.png)


#### 모르는 데이터 피쳐 조사
- 'Volume' - ' Taker buy base asset volume' = ' Maker buy base asset volume'
> source by : https://www.binance.kr/apidocs/#individual-symbol-mini-ticker-stream
- quote asset volume = coin volume / btc volume
> quote asset volume = Volume expressed in quote asset units. For pair DOGE/ BTC the volume is shown in BTC , instead of DOGE.
>>  예시--> 가상화폐/거래화폐에서 거래화폐의 양
한국돈으로 돌고돌아 계산한다고 쳐 (100만)
ex) btc/usdt 면 usdt의 가치 57000*1200
에서의 qav = 100만/1200 =>8만xxx
btc/krw면 btc의 가치 7400만  100만
에서의 qav = 100만
tb_base_av
coin / xxxxx
volume / quote_av
0 =19.xxxxx
1 = 0.028xxxxx
2 = 0.268xxxxx
3 = 0.238 xxxxx
4 = 2.1312xxxx
5 = 52.1123xxxx(**maximum coin**)
6= 0.22421
7= 19.3821
8 = 0.003426
9 = 0.00013(**minimum coin**)
====> **작을수록 비싼 코인으로 추정**

#### Open price outlier problem 
- 샘플 내 outlier 너무 빈도가 적고, regression으로 학습하기 어려움(raw, smoothing, log smoothing 별 차이 없음)
![resnet_image](./images/price_displot.png)
<center><open price distribution plot></center>

![resnet_image](./images/price_boxplot.png)
<center><open price box plot></center>

- open price outlier detection tempary method
```python
for temp_arr in outlier_arr:
    plt.plot(temp_arr, label = 'True series')
    plt.ylim(open_arr.min(), open_arr.max())
    plt.legend()
    plt.show()

filtered_y_df = raw_y_df[~raw_y_df["sample_id"].isin(outlier_list)]
```
![resnet_image](./images/outlier_image.png)
<center><outlier range boxplot></center>

#### EDA code
coin eda code link : <a href ='./coin_eda.ipynb'>"here"</a>

#### Data handling memo 
1. greedy feature add based on taker volumn data
```python
''' greedy feature handleing'''
# test_df = train_x_df[train_x_df['volume'] != 0]
# test_df['rest_asset'] = test_df['volume'] - test_df['tb_base_av']
# test_df['greedy'] = test_df['tb_base_av'] / test_df['volume']

# test_df2 = test_df[['time', 'coin_index', 'open', 'high', 'low', 'close', 'volume', 'trades', 'tb_base_av','rest_asset', 'greedy']]
# test_df2[['coin_index','trades', 'volume', 'tb_base_av','rest_asset', 'greedy']].head()
# test_df2[test_df2['greedy'] == 1][['coin_index','trades', 'volume', 'tb_base_av','rest_asset', 'greedy']].head()
```

2. 변동폭 feature add based on high and low price difference
```python
print(
    f'''
    {df.high.max()}
    {df.low.max()}
    {df.open.max()}
    {df.close.max()}
    
    
    {df.high.min()}
    {df.low.min()}
    {df.open.min()}
    {df.close.min()}
    
    '''
    
    ''' high - low = 변동폭 \n'''
    ''' 음봉양봉 구분 추가 가능'''
)
```
___
### Chapter. 2 - Season 1 model pilot
- sample id = 0, open data series로만 모델링 진행
#### ARIMA modeling
- ARIMA arg meanings : https://otexts.com/fppkr/arima-forecasting.html
```python
# ARIMA model fitting : model arguments 는 임의로 진행
model = ARIMA(x_series, order=(3,0,1))
fit  = model.fit()
pred_by_arima = fit.predict(1381, 1380+120, typ='levels')
```
#### Prophet modeling
- Time Series Forecasting — ARIMA vs Prophet : https://medium.com/analytics-vidhya/time-series-forecasting-arima-vs-prophet-5015928e402a
- facebook github : https://facebook.github.io/prophet/docs/quick_start.html
- prophet 설명 블로그 : https://zzsza.github.io/data/2019/02/06/prophet/

```python
# pprophet 모델 학습 
prophet = Prophet(seasonality_mode='multiplicative', 
                  yearly_seasonality=False,
                  weekly_seasonality=False, daily_seasonality=True,
                  changepoint_prior_scale=0.06)
prophet.fit(x_df)

future_data = prophet.make_future_dataframe(periods=120, freq='min')
forecast_data = prophet.predict(future_data)

```
#### result plot
![season_1_pilot_image](./images/season_1_pilot.png)
{: width="50" height="50"}

#### season 1 pilot code
season 1 pilot code link : <a href ='./season1_pilot.ipynb'>"here"</a>

___

### Chapter. 3 - Personal modeling prediction
- 기존의 driving 방식처럼 trian_x에서 open column만 활용하여 yhat predict함.

#### ARIMA trial
- 우선 기존 ARIMA 방법을 Baseline으로 잡고, 진행
- hyperparameter p,d,q는 임의로 잡음
```python
def train(x_series, y_series, args):
    
    model = ARIMA(x_series, order=(2,0,2))
    fit  = model.fit()
    
    y_pred = fit.predict(1381, 1380+120, typ='levels')
    error = mean_squared_error(y_series, y_pred)
    plotting(y_series, y_pred, args.sample_id)

    return error*10E5
```
![colab_arima_prediction_image](./images/colab_arima_prediction.png)

Colab link : https://colab.research.google.com/drive/1x28Mi9MSqqkSTO2a8UU0wXDzgXNy2WT9?usp=sharing


#### Prophet trial
- hyperparameter는 임의로 설정, seasonality는 코인 데이터가 addtitive 보다는 multiplicative가 적합하다고 판단
```python
prophet= Prophet(seasonality_mode='multiplicative',
                  yearly_seasonality='auto',
                  weekly_seasonality='auto', daily_seasonality='auto',
                  changepoint_range=0.9,  
                  changepoint_prior_scale=0.1  # 오버피팅, 언더피팅을 피하기 위해 조정
                )

prophet.add_seasonality(name='first_seasonality', period=1/12, fourier_order=7) # seasonality 추가
prophet.add_seasonality(name='second_seasonality', period=1/8, fourier_order=15) # seasonality 추가

prophet.fit(x_df)

future_data = prophet.make_future_dataframe(periods=120, freq='min')
forecast_data = prophet.predict(future_data)
```
- sample_id = 1, dataset 예측 결과
![colab_prophet_prediction_image](./images/colab_prophet_prediction.png)
<center><open price prophet prediction plot></center>
Colab link : https://colab.research.google.com/drive/1dDf6AIln31catWWDsrB_lbL-0M5DsZTd?usp=sharing



#### Neural Prophet trial
- hyperparameter 임의로 잡음, seasonality mode는 이전 prophet model처럼 mulplicative로 진행
```python
def prophet_preprocessor(x_series):
    
    # start time initialization
    start_time = '2021-01-01 00:00:00'
    start_dt = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')

    # datafram 만들기
    x_df = pd.DataFrame()
    # 분당 시간 데이터 시리즈 입력
    x_df['ds'] = [start_dt + datetime.timedelta(minutes = time_min) for time_min in np.arange(1, x_series.shape[0]+1).tolist()]
    # 가격 데이터 시리즈 입력
    x_df['y'] = x_series.tolist()

    return x_df


def train(x_series, y_series, **paras):
    
    x_df = prophet_preprocessor(x_series)
    
    model = NeuralProphet(
                          n_changepoints = paras['n_changepoints'],
                          changepoints_range = paras['changepoints_range'],
                          num_hidden_layers = paras['num_hidden_layers'],
            
                          learning_rate = 0.1, epochs = 40, batch_size = 32,
                          seasonality_mode = 'multiplicative', 
                          yearly_seasonality = False, weekly_seasonality = False, daily_seasonality = False,
                          normalize='minmax'
                         )
    
    model.add_seasonality(name='first_seasonality', period=1/24, fourier_order=5) 
    model.add_seasonality(name='second_seasonality', period=1/12, fourier_order=10)

    metrics = model.fit(x_df, freq="min")

    future = model.make_future_dataframe(x_df, periods=120)
    forecast = model.predict(future)
    error = mean_squared_error(y_series, forecast.yhat1.values[-120:])

    return error

```
Colab link : https://colab.research.google.com/drive/1E38kkH2mfFgnGKj89t2mLZV6xg7rPQl8?usp=sharing

#### Keras RNN models trail
- 음.. 비슷한 방식으로 open 가격 데이터만이 아닌, feature까지 활용해서 driving해보고 싶었음

### chapter.4 - Data preprocess
#### Data smoothing
- 이후, DNN 계열의 모델링을 시도했으나, 제대로 regression이 되지 않음. -> 기존 데이터는 너무 진폭이 심해서 모델이 regression을 하기 어렵다고 판단함.

- smoothing method 1 : simple exponential smoothing
- smoothing method 2 : moving average

- price data smoothing
![smoothing_image](./images/smoothing.png)


#### Data discretize
- 데이터를 계층화하는 게 어떨까? 싶었음..

- discretize method : KBinsdiscretizer library(in scikit-learn)
```python
from sklearn.preprocessing import KBinsDiscretizer
kb = KBinsDiscretizer(n_bins=10, strategy='uniform', encode='ordinal')
kb.fit(open_y_series)
#  이때 `bin_edges_` 메소드를 이용하여 저장되어진 경계값을 확인할 수 있다.
print("bin edges :\n", kb.bin_edges_ )
```

![resnet_image](./images/kbinsdiscretize_plot.png)
<center><kbinsdiscretizer before & after plot></center>


#### Data log normalization
-시계열 데이터 정규화

``` python
data = data.apply(lambda x: np.log(x+1) - np.log(x[self.x_frames-1]+1))
```
- 시계열 데이터 정규화 방법 출처 : https://github.com/heartcored98/Standalone-DeepLearning/blob/master/Lec8/Lab10_Stock_Price_Prediction_with_LSTM.ipynb(2019 KAIST 딥러닝 홀로서기 )


### chapter.3 - RNN modeling
<a href ='./rnn_modeling.ipynb'>RNN 모델링</a>
sample이 1208개 있고, 한 샘플당 1500개의 시리즈 구성 -> 여기서 또 1500개 375개로씩 짤라
한 sample 당 4개의 미니 세트로 구성된건데.. 

1208 * 4 = 6832 세트

6000세트는 학습시키고
382세트는 validation 

1208 샘플이 있으면 1100개 의 샘플은 학습데이터로 빼고 나머지를 split 


어차피 시간대 별 특성이 없다고 치면. 코인 인덱스가 같으면. 

datasetgenerate로 0 ~ 1000까지 샘플 가지고 오고.
1. 샘플을 가져와서 

1, 1500, 12 - 


### chapter. 4 - 실수차원 ARIMA
### chapter. 5 - Neural prophet
### chapter. 6 - tensorflow time series forecasting course study

* reference : <a href = 'https://www.tensorflow.org/tutorials/structured_data/time_series?hl=ko#%EB%8B%A8%EC%9D%BC_%EC%8A%A4%ED%85%9D_%EB%AA%A8%EB%8D%B8'>tensorflow tutorial</a>
* requirements 
    1. tensorflow 2.x 이상과
    2. 해당 텐서 버젼과 호환되는 numpy 
    3. 확인방법 : tensorflow과
    4. 내가 설치한 verison(텐서가 version2 부터는 gpu과 cpu 동시 호환을 default로 하는데, 노트북 사용자라서, cpu전용 버젼을 따로 설치함.)
        * pip install tensorflow 2.4.0-cpu
        * pip install numpy-1.19.2

* 보통 에러 : ㄴ엄어마ㅣㅓㅇㅁ
* 에러창 지우는 방법 : 
```python
# tensorflow import 하기 전에, define 해야 함.
import warnings
warnings.filterwarnings('ignore')
```
___
### 스터디 1단계 - 전반적인 코드 리뷰 및 시범운용
우선 9번 코인(bitcoin이라 추정) 중 한 sample만 가지고 study 진행   
* 의문점1 : Normalized를 시키는 게 맞을 지는 의문임. 이미 가격 data columns들이 time index가 1380인 시점을 기준으로 부분 정규화가 되어있음.   
* 배운점1 : 전반적인 class를 사용해서 데이터를 처리하는 방식은 크게 배울만 함. 깔끔하게 정리되고, 여러 모델에 적용하기 용이함.
* 배운점2 : Residual networks는 기본적으로 AR(auto-regression)이 기본 가정인듯. future은 이전 과거 데이터로 부터 생성된다. 즉, data간의 시게열 변화가 작아야 더욱 더 효용성 있는 model임.

![resnet_image](./images/residual.png)

> 시계열 분석에서는 다음 값을 예측하는 대신 다음 타임스텝에서 값이 어떻게 달라지는 지를 예측하는 모델을 빌드하는 것이 일반적입니다. 마찬가지로 딥러닝에서 "잔여 네트워크(Residual networks)" 또는 "ResNets"는 각 레이어가 모델의 누적 결과에 추가되는 아키텍처를 나타냅니다. **이것은 변화가 작아야 한다는 사실을 이용하는 방법입니다.**

* 결과 : 한 샘플만 가지고 적용할 때에는 크게 효과 있어보이지는 않음. 한 코인당 하나의 모델을 만드는 방식으로 샘플을 train, validation, test 구간으로 나누어 다시 재적용해 볼 필요가 있음.

매수 매도 골든크로스 전략은 가능한건가?
크로스의 주기가 너무 잦으면별로임.. 횟수가 적게 조정해야 함
예측된 구간 내 골든크로스가 발생한다고 예측이 되면 1381때 1로 구매 아니면 0 으로 패스
그리고 구매이후 데드크로스가 발생시 전부 판매하면 안정적인 전략, 차라리 데드크로스 발생 조금전에 해버리면 더 높은 가격에 팔수도 있을 거란 것으로 유추됨

120 중 하나를 30% 확률로 맞추는 모델링
최종적으로 확인할 것
분류 문제로 풀 경우 무조건 사지 말고 맞출확률에 cap을 씌워서 가능성 높은 거만 사는 방법으로 확인할 것
moving average 줘서 문제 다시 풀어볼 것

regression 가능해지는 지 확인

Conv1d(output dim 수(채널수), kenel_size ) 임

처음에 64개의 output channel로 10개씩 찍었으니까

1380 - 10 + 1 = 이 압축된 sequece 수
10 씩 하면 결국 10분 데이터 하나의 데이터로 압축시키는 걸 의미

how to use?
- 1d CNN은 더 큰 filter size를 써도 된다.
- 1d CNN은 더 큰 window size를 써도 된다.
- filter size로 일반적으로 7 or 9가 선택된다.





### 참고한 자료
1. Time-Series Forecasting: NeuralProphet vs AutoML: https://towardsdatascience.com/time-series-forecasting-neuralprophet-vs-automl-fa4dfb2c3a9e

2. Techniques to Handle Very Long Sequences with LSTMs : https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/
> A reasonable limit of 250-500 time steps is often used in practice with large LSTM models.
3. Neural prophet baseline : https://dacon.io/codeshare/2492

4. 예보 데이터 전처리와 선형보간 : https://dacon.io/competitions/official/235720/codeshare/2499?page=1&dtype=recent
> 트렌드 추출해서 interpolation 처리 후 반영 방법

5. ARIMA 원리 설명: https://youngjunyi.github.io/analytics/2020/02/27/forecasting-in-marketing-arima.html

6. facebook prophet : https://facebook.github.io/prophet/docs/quick_start.html#python-api
> **prophet changepoint range의 의미** 
100%으로 하면 오버피팅 되긴 할듯, By default changepoints are only inferred for the first 80% of the time series in order to have plenty of runway for projecting the trend forward and to avoid overfitting fluctuations at the end of the time series. This default works in many situations but not all, and can be changed using the changepoint_range argument.

> **prophet changepoint_prior_scale의 의미** 
이상치반영정도? 같은 느낌, If the trend changes are being overfit (too much flexibility) or underfit (not enough flexibility), you can adjust the strength of the sparse prior using the input argument changepoint_prior_scale. By default, this parameter is set to 0.05

7. Cryptocurrency price prediction using LSTMs | TensorFlow for Hackers (Part III) : https://towardsdatascience.com/cryptocurrency-price-prediction-using-lstms-tensorflow-for-hackers-part-iii-264fcdbccd3f

8. tensorflow time-series forecasting tutorial : https://www.tensorflow.org/tutorials/structured_data/time_series?hl=ko

9. 시계열 예측 패키지 Prophet 소개 : https://hyperconnect.github.io/2020/03/09/prophet-package.html

10. fourie order meaning in prophet : https://medium.com/analytics-vidhya/how-does-prophet-work-part-2-c47a6ceac511
> m.add_seasonality(name='first_seasonality', period= 1/24 , fourier_order = 7) 1/24 가 1일을 24등분해서 1시간 마다의 시즈널을 입히는 것
 m.add_seasonality(name='second_seasonality', period=1/6, fourier_order = 15)  1/6 하면 1일을 6등분해서 4시간 마다의 시즈널을 입히는 것

11. [ML with Python] 4.구간 분할/이산화 & 상호작용/다항식  - https://jhryu1208.github.io/data/2021/01/11/ML_segmentation/

12. A Simple LSTM-Based Time-Series Classifier : https://www.kaggle.com/purplejester/a-simple-lstm-based-time-series-classifier

13. PyTorch RNN 관련 티스토리 블로그 : https://seducinghyeok.tistory.com/8

14. [PyTorch] Deep Time Series Classification : https://www.kaggle.com/purplejester/pytorch-deep-time-series-classification/notebook

15. PyTorch로 시작하는 딥 러닝 입문 wicidocs : https://wikidocs.net/64703

16. scikit-learn kbins docs : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html

17. Pytorch로 CNN 구현하기 티스토리 블로그 : https://justkode.kr/deep-learning/pytorch-cnn

18. CNN을 활용한 주가 방향 예측 : https://direction-f.tistory.com/19

19. Bitcoin Time Series Prediction with LSTM : https://www.kaggle.com/jphoon/bitcoin-time-series-prediction-with-lstm

20. 시즌 1, CNN 모델 팀 : https://dacon.io/competitions/official/235740/codeshare/2486?page=1&dtype=recent