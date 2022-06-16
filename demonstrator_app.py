import streamlit as st


def main():
    st.set_page_config(
         page_title="Federated Learning Demonstrator",
         page_icon="🚀",
         layout="centered",
         initial_sidebar_state="expanded",
         menu_items={
             'Get Help': 'https://www.extremelycoolapp.com/help',
             'Report a bug': "https://www.extremelycoolapp.com/bug",
             'About': '''## This demonstrator has been designed by Lukas Delbrück & Matthias Ewald'''
         }
    )
    st.image("pictures/HKA.png")
    st.title("Federated Learning Demonstrator")
    st.markdown("by Lukas Delbrück and Matthias Ewald")


if __name__ == '__main__':
    main()
