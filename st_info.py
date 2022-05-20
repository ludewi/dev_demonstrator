"""Information"""
import streamlit as st


def app():
    st.title("Informationsseite")



    """ In diesem Container wird die gesamte Funktionsweise des Demonstators erklärt"""
    example = st.container()
    example.subheader('Was ist MNIST?')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("pictures/MNIST_1.png")
    with col2:
        st.image("pictures/MNIST_2.png")
    with col3:
        st.image("pictures/MNIST_3.jpeg")
    example.markdown("Lorem Ipsum is simply dummy text of the printing and typesetting industry. "
                       "Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, "
                       "when an unknown printer took a galley of type and scrambled it to make a type specimen book. "
                       "It has survived not only five centuries, but also the leap into electronic typesetting, "
                       "remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets "
                       "containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker "
                       "including versions of Lorem Ipsum.")

    """ In diesem Container wird die gesamte Funktionsweise des Demonstators erklärt"""
    explain = st.container()
    explain.subheader('Was ist FL?')
    col1, col2 = st.columns(2)
    with col1:
        st.image("pictures/Fed_learn.png")
    with col2:
        #st.image("pictures/Aufbau_FL.png")
        video_file = open('pictures/fl_google.mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes, start_time=22)
    
    explain.markdown("Lorem Ipsum is simply dummy text of the printing and typesetting industry. "
                       "Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, "
                       "when an unknown printer took a galley of type and scrambled it to make a type specimen book. "
                       "It has survived not only five centuries, but also the leap into electronic typesetting, "
                       "remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets "
                       "containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker "
                       "including versions of Lorem Ipsum.")
    #explain.image("./Funktion.gif")

    """ In diesem Container wird der gesamte Aufbau des Demonstators erklärt"""
    setup = st.container()
    setup.subheader('Aufbau des Demonstrators')
    setup.markdown("Lorem Ipsum is simply dummy text of the printing and typesetting industry. "
                     "Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, "
                     "when an unknown printer took a galley of type and scrambled it to make a type specimen book. "
                     "It has survived not only five centuries, but also the leap into electronic typesetting, "
                     "remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets "
                     "containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker "
                     "including versions of Lorem Ipsum.")
    #setup.image("./Aufbau_FL.png")

    """ In diesem Container werden die Vorteile des Demonstators erklärt"""
    advantage = st.container()
    advantage.subheader('Welche Vorteile hat FL?')
    advantage.markdown("**Datensicherheit**")
    advantage.markdown("**Größere Datenbasis**")
