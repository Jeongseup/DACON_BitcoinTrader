## 데이콘 비트코인 트레이더 시즌 2


<a href ='./coin_eda.ipynb'>EDA 정리</a>


<a href ='./season1_pilot.ipynb'>시즌 1 파일럿 모델링</a>

기존의 driving 방식은 trian_x에서 open column만 가지고 
ARIMA나 Prophet으로 연산시켜서.
y값을 추정함.


음.. 비슷한 방식으로 open 가격 데이터만을 가지고
Tree 모델이나
DNN 이나 
RNN 등을 해볼 수 있을 텐데..
60분의 데이터를 보고 



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