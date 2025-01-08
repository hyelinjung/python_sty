import numpy as np
from sklearn.linear_model import LinearRegression

# 샘플 데이터 생성
# 공부 시간(X1)과 과외 시간(X2)에 따른 시럼 점수(Y)예측

# 하루공부시간
study_hours = np.array([2,3,4,5,4,6,7,8,5,4])
# 주간 과외 시간
tutor_hours = np.array([0,1,0,2,1,3,2,3,2,1])
# 시험 점수
test_scores= np.array([60,70,75,85,75,90,92,95,80,75])

# 입력 데이터 형태 맞추기
X=np.column_stack((study_hours,tutor_hours))
y=test_scores

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X,y)

# 결과 출력
print('회귀 계수:')
print(f'공부 시간: {model.coef_[0]:.2f}')
print(f'과외 시간: {model.coef_[1]:.2f}')

# 공부나 과외를 전혀하지 않은 학생의 점수
print(f'절편: {model.intercept_:.2f}')

# 새로운 데이터로 예측 -공부6시간 과외2시간
new_student = np.array([[6,2]])
prediction = model.predict(new_student)
print(f'\n예측 점수:{prediction[0]:.1f}')

# 모델 성능 평가 (R-squared)
r_squared = model.score(X,y)
print(f'R-squared:{r_squared:.4f}')

