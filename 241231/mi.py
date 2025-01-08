import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import random

# 재현성을 위해 ramdom seed설정 -랜덤으로 가져오더라도 순서를 정함
random.seed(42)
np.random.seed(42)

# 데이터 준비
# 독립변수
X=np.array([[1],[2],[3],[4],[5]])
# 종속변수
y=np.array([2,4,5,4,5])

# 데이터분활
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train,y_train)

# 예측 및 평가 -X의 값이 a일때 y의 값이 어떤게 나올것 같다
predictions = model.predict(X_test)

#결과 출력
print('학습 데이터 크기',X_train.shape[0]) 
print('테스트 데이터 크키:',X_test.shape[0])
print('\n모델 계수:')
print('기울기(slope)',model.coef_[0])
print('절편 intercept: ',model.intercept_)
print('\n테스트 세트 실제 값' ,y_test)
print('테스트 세트 예측 값',predictions)
print('\n모델 성능')
print('R 점수:',r2_score(y_test, predictions))
print('평균 제곱 오차 MSE',mean_squared_error(y_test, predictions))
