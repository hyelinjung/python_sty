import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 폰트 설정 
plt.rcParams['font.family'] = 'NanumGothic'
# - 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus']=False
# 지수 표현식 방지
pd.options.display.float_format='{:.2f}'.format

# 데이터 생성
np.random.seed(42)
n_samples=100

# 독립변수 생성
X1=np.random.normal(0,1,n_samples)
X2=np.random.normal(0,1,n_samples)
X3=np.random.normal(0,1,n_samples)

# 종속변수 생성(Y=2X1+3X2 +1.5X3 + 오차)
Y = 2+ X1 +3*X2 + 1.5 * X3 +np.random.normal(0,1,n_samples)

# 데이터프레임 생성
data = pd.DataFrame({
    'X1': X1,
    'X2':X2,
    'X3':X3,
    'Y':Y
})

# 학습용과 테스트용 데이터 분리
X=data[['X1','X2','X3']]
y=data['Y']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train,y_train)

# 예측
y_pred = model.predict(X_test)

# 결과 출력
print('회귀 계수:')
for feature, coef in zip(X.columns, model.coef_):
    print(f'{feature}:{coef:.4f}')
print(f'\n절편: {model.intercept_:.4f}')
print(f'\nR-squared 값: {r2_score(y_test,y_pred):.4f}')
print(f'\n평균제곱근오차 RMSE: {np.sqrt(mean_squared_error(y_test,y_pred)):.4f}')

# 실제값과 예측값 비교 시각화
# 그래프의 크기 지정
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred, color='blue', alpha=0.5)
plt.plot([-4,4],[-4,4],'r--')
plt.xlabel('실제값')
plt.ylabel('예측값')
plt.title('실제값 vs 예측값')
plt.grid(True)
plt.show()

# 새로운 데이터로 예측 예시
new_data = np.array([[0.5,1.0,-0.5]])
prediction = model.predict(new_data)
print(f'\n새로운 데이터 예측값: {prediction[0]:.4f}')