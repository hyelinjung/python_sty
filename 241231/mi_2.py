#피쳐에 따른 타켓발생 예측/ 피쳐별 타켓에 영향을 미치는 중요도
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('./dataSet/diabetes.csv')
print('df: ',df)

# 데이터 확인
print('데이터셋 크기:',df.shape)
print('\n처음 5개 행:')
print(df.head())
print('\n기술통계:')
print(df.describe)

# 독립변수와 종속변수(y) 분리 -x는 outcome제외 모든 컬럼,y는outcome컬럼만
# outcome을 제외한 모든 특성
X=df.drop('Outcome',axis=1)
y=df['Outcome']
print(X)
print(y)

# 데이터 분할 -
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train,y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 평가
print('\n모델 성능:')
print('R2 점수:',r2_score(y_test,y_pred))
print('평균 제곱 오차 MSE: ',mean_squared_error(y_test,y_pred))
print('평균 절대 오차 MAE:',mean_absolute_error(y_test,y_pred))

# 특성 중요도 분석
feature_importance = pd.DataFrame({
    'feature' : X.columns,
    'importance':np.abs(model.coef_)
})
print('feature_importance',feature_importance)
feature_importance = feature_importance.sort_values('importance',ascending=False)
print('\n특성 중요도:')
print(feature_importance)

# 시각화 :특성 중요도
# plt.figure(figsize=(10,6))
# sns.barplot(x='importance',y='feature',data=feature_importance)
# plt.title('Feature Importance in Diabetes Prediction')
# plt.xlabel('Absolute Coefficient Value')
# plt.tight_layout()
# plt.show()

# # 시각화 : 실제값 vs 예측값
# plt.figure(figsize=(8,6))
# plt.scatter(y_test,y_pred,alpha=0.5)
# plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--',lw=2)
# plt.show()

