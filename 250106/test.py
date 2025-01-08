import streamlit as st
import pandas as pd
import numpy as np

st.title('스트림릿 앱')
name = st.text_input('이름을 입력하시오')

if st.button('확인'):
    st.write(f'안녕하세요,{name}님')
    
x=st.slider('숫자를 선택',0,100)
st.write(f'선택한 숫자는 {x}입니다')