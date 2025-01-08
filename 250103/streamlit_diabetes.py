#!/usr/bin/env python
# coding: utf-8

# # 스트림릿을 사용한 당뇨 예측

# In[11]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# 한 번 학습하면 pickle아리는 파일을 저장할 때 사용하는 라이브러리
import joblib
import streamlit as st
import matplotlib as plt


# In[4]:


# 폰트지정
plt.rcParams['font.family'] = 'Malgun Gothic'
# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False

# 숫자가 지수표현식으로 나올 때 지정
pd.options.display.float_format = '{:.2f}'.format


# In[5]:


# 데이터 로드 및 전처리
data = pd.read_csv('../dataSet/diabetes.csv')


# In[6]:


# 선택된 feature만 사용
selected_feature = ['Glucose', 'BMI', 'Age']
X = data[selected_feature]
y=data['Outcome']


# In[8]:


# 학습데이터
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[10]:


# 랜덤포레스트 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)


# In[13]:


# 모델 저장
joblib.dump(model,'dia.pkl')


# In[14]:


# 테스트 데이테로 정확도 확인
y_predict = model.predict(X_test)
accuracy = accuracy_score(y_test,y_predict)
print(f'model Accuracy: {accuracy *100:.2f}%')


# In[15]:


# Streamlit
st.title('당뇨병 예측 시스템')
st.write('Glucose, BMI, Age 값을 입력하여 당뇨병을 예측해보시오')


# In[17]:


# 사용자 입력받기
glucose = st.slider('Glucose (혈당 수치)', min_value=0, max_value=200, value=100)
bmi = st.slider('BMI (체질량지수)', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
age = st.slider('Age (나이)', min_value=0, max_value=100, value=30)


# In[18]:


# 예측하기 버튼
if st.button('예측하기'):
    model = joblib.load('dia.pkl')
    input_data = np.array([[glucose,bmi,age]])
    prediction = model.predict(input_data)[0]
    
    if prediction ==1:
        st.write("가능성이 높다")
    else:
        st.write("가능성이 낮다")


# In[ ]:




