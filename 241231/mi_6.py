import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('./dataSet/LinearRegressionData.csv')

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

print(len(X))
print(len(X_train))
print(len(X_test))

print(len(y))
print(len(y_train))
print(len(y_test))

model = LinearRegression()
model.fit(X_train,y_train)

# plt.scatter(X_train,y_train,color='blue')
# plt.plot(X_train,model.predict(X_train),color='green')
# plt.title('Score by hours')
# plt.xlabel('hours')
# plt.ylabel('score')
# plt.show()

r=model.coef_
print(r)
print(model.score(X_test,y_test))
