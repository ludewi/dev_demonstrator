"""Information"""
import streamlit as st

st.set_page_config(
     page_title="1. What to do?",
     page_icon="🚀",
     layout="wide",
     initial_sidebar_state="expanded"
    )

st.title("What to do?", anchor=None)

with st.sidebar:
    st.subheader("Client ID: 001")

#quick_start = st.container()
st.write('Quick Start Guide - Here are your tasks are:')
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.subheader('**1.** **Data Generation**')
    st.image("pictures/draw.PNG")

with col2:
    st.subheader('**2.** **Particitpate in Federated Learning**')
    st.image("pictures/participate.PNG")

with col3:
    st.subheader('**3.** **Test the trained Model**')
    st.image("pictures/test_model.PNG")
