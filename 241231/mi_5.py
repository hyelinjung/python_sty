import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import pandas as pd
dataSet = pd.read_csv('./dataSet/LinearRegressionData.csv')
head =dataSet.head()
print(head)

# 제일마지막 컬럼겂 뺀 나머지
X=dataSet.iloc[:,:-1].values
y=dataSet.iloc[:,-1].values
print('y의값',y)
print('X의값',X)

model = LinearRegression()
model.fit(X,y)

y_prediction = model.predict(X)
print(y_prediction)

# 시각화
# plt.scatter(X,y,color='blue')
# plt.title('Score by hours')
# plt.plot(X,y_prediction,color='green')
# plt.xlabel('hours')
# plt.ylabel('score')
# plt.show()

print('5시간 공부 시 예상 점수',model.predict([[5]]))
# 기울기
model.coef_
print(model.coef_)

# 절편
model.intercept_

