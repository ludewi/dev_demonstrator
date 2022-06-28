"""Information"""
import streamlit as st

st.set_page_config(
     page_title="Information",
     page_icon="🚀",
     layout="wide",
     initial_sidebar_state="expanded"
    )

st.title("Information page", anchor=None)
st.markdown("**Here you find basic information about our demonstrator.**")

example = st.container()
example.subheader('What is the MNIST dataset?')
example.markdown("MNIST is a large database of handwritten digits. It´s widely used for machine learning training and testing."
                 "In order for the numbers to be used for machine learning, the number images of the MNIST dataset must first be converted into a form that is understandable to the computer."
                 "The computer can only read numbers, so we transform the image into a number representation. All white color pixels get the value 255 and all black color pixels get the value 0."
                "When using the demonstrator later you will perform such a transformation.")

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.empty()
with col2:
    st.image("pictures/MNIST4.png")
with col3:
    st.empty()

explain = st.container()
explain.subheader('What is federated learning?')

col1, col2 = st.columns(2)
with col1:
    st.image("pictures/FL_2.png")
with col2:
    video_file = open('pictures/fl_google.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes, start_time=22)

explain.markdown("Traditional machine learning (ML) algorithms are typically trained on data collected from various endpoints such as cell phones, laptops, etc. and aggregated on a central server. The ML- algorithms are then trained on this data to make a prediction on new data. Traditional ML methods also send sensitive user data to the servers, such as chat messages or images. To prevent this, the concept of federated learning (FL) was introduced in 2016. With **federated learning**, ML algorithms can be trained on local data without exchanging data with a central instance. This is a huge advantage in terms of data protection.")


explain.markdown("A Federated Learning System (FLS) consists of multiple clients and a server. The clients are usually understood to be end devices such as cell phones that generate the local data. The server orchestrates the learning. It specifies how many rounds are trained, how many clients should participate in the training, which hyperparameters should be used, and so on. Federated training then always follows the same steps.")

explain.markdown("1. First, all clients download the current ML model (such as Logistic Regression, CNN, FFNN) from the server.")
explain.markdown("2. the clients train the model with their local data. This causes the parameters (usually the weights of a neural network) to be adjusted. Since each client has different data, the models of all clients will also have different parameters after training.")
explain.markdown("3. Once the training of the clients on the local data is completed, they send back only the adjusted parameters to the server.")
explain.markdown("4. The server aggregates all received parameters into a new fitted model.")
explain.markdown("Steps 1-4 are executed several times. Thus, it is possible to train a global ML model without exchanging local data with a central instance. "
                 "(vgl. McMahan et al., 2017)")

setup = st.container()
setup.subheader('Design of the demonstrator')

setup.markdown("The demonstrator is designed to give you a practical example of how federated learning works. The computer you are currently using acts as a client. On this client data can be generated (here numbers between 0-9). Our goal is to create an AI model that can later classify the numbers 0-9 based on our generated data. However, we don't want any central entity to be able to see our generated numbers and since one client can only generate very few numbers it would be helpful to be able to use the numbers generated by other clients for our model without being able to look at the numbers of the other client. Federated training can do just that."
    ""
    ""
    "When you think you have generated enough numbers, you can click on the **Start Training** button. Then your client will connect to the federated learning server. The demonstrator will then show you in real time what it is doing and what intermediate results have been generated. Have fun using it")

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.empty()
with col2:
    st.image("pictures/Aufbau.png")
with col3:
    st.empty()
