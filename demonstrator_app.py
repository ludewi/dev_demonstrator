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
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("Dieser Demonstrator wurde im Rahmen der Masterarbeit von Lukas Delbrück und Matthias Ewald im SS 2022"
                " an der Hochschule Karlsruhe erstellt.")
    st.markdown("Erstprüfer: Andreas Wagner")
    st.image("pictures/Credits_Lukas_Matthias.PNG", width=250)


if __name__ == '__main__':
    main()
