import st_info as page_info
import st_demonstrator_ohneAug as page_input
import st_result as page_result
import streamlit as st
import utils as utl

st.set_page_config(
     page_title="Federated Learning Demonstrator",
     page_icon="🚀",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': '''## This demonstrator has been designed by Lukas Delbrück & Matthias Ewald'''
     }
 )

###### HEAD ########
#st.title("Federated Learning System - Dashboard")
#st.write("We designed a Federated Learning System using the Framework FLOWER")
#st.markdown('<h1 style="text-align:center;color:red;font-weight:bolder;font-size:50px;">Federated Learning<br>Demonstrator</h1>',unsafe_allow_html=True)


# https://medium.com/@u.praneel.nihar/building-multi-page-web-app-using-streamlit-7a40d55fa5b4 Quelle für Multi App

PAGES = {
    "Information": page_info,
    "Dateneingabe": page_input,
    "Ergebnisse": page_result
}

with st.sidebar:
    #rst.image('/HKA.png')
    st.write("Client ID: 001")
    #st.subheader('Navigation')
    selection = st.radio("Navigation", list(PAGES.keys()))
    page = PAGES[selection]


if __name__ == '__main__':
    page.app()
