import streamlit as st


def main():
    st.set_page_config(
         page_title="Federated Learning Demonstrator",
         page_icon="🚀",
         layout="centered",
         initial_sidebar_state="expanded",
         menu_items={
             'About': "https://www.h-ka.de"
         }
    )
    st.image("pictures/HKA.png")
    st.title("Federated Learning Demonstrator", anchor=None)
    st.markdown("by Lukas Delbrück and Matthias Ewald")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("This demonstrator was created as part of the master's thesis by Lukas Delbrück and Matthias Ewald in the summer semester of 2022 at Karlsruhe University of Applied Sciences.")


if __name__ == '__main__':
    main()
