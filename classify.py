# 분류-이진과 다중

# 세가지 분류값 중에 어떤 클래스에 속하는지 예측할 수 있도록 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
ir =load_iris()
X = ir.data
y=ir.target

X_t,X_ts,y_t,y_ts = train_test_split(X,y,test_size=0.2,random_state=42)
model_1 = DecisionTreeClassifier()
model_1.fit(X_t,y_t)
y_pred = model_1.predict(X_ts)
score =model_1.score(X_t,y_t)
accu = accuracy_score(y_ts,y_pred)
print(f'score:{score},accu:{accu}') 
print('분류 모델의 성능 평가 지표를 요약해 보여주는',classification_report(y_ts,y_pred,target_names=ir.target_names))
# precision -정밀도로 모델이 양성으로 예측한 것 중에서 실제 양성인 비율->예측을 했는데 실제로 그 예측이 맞을 확률(예시에서는 피쳐별 100%로)
# recall- 재현율로 실제 양성 중 모델이 양성으로 올바르게 예측한 비율 즉, 실제 양성을 잡는 
# support - 각 클래스에 속한 실제 샘플의 개수

##Tf-IDF -> 텍스트에서 단어의 중요도를 판단- 모델이 읽을 수 있게 행렬타입으로 반환
df = pd.read_csv('./dataset/spam.csv',encoding='ISO-8859-1')
df = df[['v1','v2']]
df.columns = ['label','text']
print(df)

df['label'] = df['label'].map({'ham':0,'spam':1})
print(df.isnull().sum())
X_t,X_ts,y_t,y_ts = train_test_split(df['text'],df['label'],test_size=0.2,random_state=42)
vec = TfidfVectorizer(stop_words='english')
X_t_tf = vec.fit_transform(X_t)
X_ts_tf = vec.transform(X_ts)
print(f'X_t:{X_t_tf}, X_ts:{X_ts_tf}')


