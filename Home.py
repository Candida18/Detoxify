import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Detoxify | Home",
    page_icon="🤝",
)

st.title('Detoxify : Hate Speech Detection ')


st.markdown("""<br/>""", True)
st.markdown(""" 

<p align="justify">Humanity has benefitted greatly from the growing usage of social media and information sharing. However, as online content continues to grow, so does the spread of hate speech. 
 Thus, to solve this emerging issue in social media sites, subsequent studies used a variety of feature engineering techniques and machine learning algorithms to automatically recognize hate speech messages on different datasets. To the best of our knowledge, no study has compared the various feature engineering techniques and machine learning algorithms to determine which feature engineering methodology and machine learning algorithm perform best on a common publically available dataset.Our project's objective is to assess the performance of various machine learning algorithms and feature engineering techniques using a publicly available dataset with three distinct classes. We also aim to create a real-time Hate & Offensive Speech Detection System         
""", True)

st.image("./images/home.jpg")

st.markdown(""" 
<br/><br/>
<p align="center"> Developed with ❤ by Brendan, Aditya & Candida</p>

""", True)
