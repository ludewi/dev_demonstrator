# import benötiger Pakete
import streamlit as st
import flwr as fl
import sys
import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
import plotly.express as px

# streamlit config
st.set_page_config(
     page_title="Federated training",
     page_icon="🚀",
     layout="centered",
     initial_sidebar_state="expanded"
    )

### Buttons ###
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #F0F2F6;
    color:#000000;
}
div.stButton > button:hover {
    background-color: #CF3A0B;
    color:#000000;
    }
</style>""", unsafe_allow_html=True)

# set variables
round_counter = 1
if "initial_weights" not in st.session_state:
    st.session_state["initial_weights"] = []

if "fedround_weights" not in st.session_state:
    st.session_state["fedround_weights"] = []

st.title("2. Federated training")

all_numbers = [st.session_state["counter_0"], st.session_state["counter_1"], st.session_state["counter_2"],
               st.session_state["counter_3"], st.session_state["counter_4"], st.session_state["counter_5"],
               st.session_state["counter_6"], st.session_state["counter_7"], st.session_state["counter_8"],
               st.session_state["counter_9"]]

train_button = st.button("------------------------------- Click me to participate in the Federated Training ----------------------------------")
################################################## start_training ###################################################
if train_button:
    # reset score lists
    local_val_score = []
    local_train_acc = []
    fed_train_acc = []
    fed_val_score = []
    ################################################## Prepare Data ###################################################

    # Load data
    np_xtrain = np.array(st.session_state["image"])
    np_ytrain = np.array(st.session_state["y_train"])

    # scaling data
    norm_xtrain = []
    for i in range(len(np_xtrain)):
        norm_xtrain.append(np_xtrain[i] / 255)

    # reshaping data
    x_train = np.array(norm_xtrain).reshape(-1, 28, 28, 1)

    if len(st.session_state["y_train"]) > 10:  # da fehler meldung wenn noch keine daten erzeugt wurden
        X = x_train
        y = np_ytrain

        # test train split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    else:
        st.error("Training cannot be started yet, because too little data has been generated.")

    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train, x_test, y_test = x_train, y_train, x_test, y_test

    # reset round counter
    round_counter = 1

    # For display status
    with st.expander("Click here for detailed information about the training data."):
        st.markdown("### Splitting of training and test data")
        st.write(f"x_train shape: {x_train.shape}")
        st.write(f"{x_train.shape[0]} train samples")
        st.write(f"{x_test.shape[0]} test samples")
        st.write(f"The training data are {sys.getsizeof(x_train)} bytes large.")

        # plot local data
        st.subheader("Sample of local generated training data")
        # define number of images to show
        num_row = 2
        num_col = 8
        num = num_row * num_col
        # get images
        images = x_train[0:num]
        labels = y_train[0:num]
        # plot images
        fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
        for i in range(num):
            ax = axes[i // num_col, i % num_col]
            ax.imshow(images[i], cmap='gray_r')
            ax.set_title('Label: {}'.format(labels[i]))
        st.pyplot(fig)

    global check_flag
    check_flag = st.empty()
    check_flag.info("Server checks if required number of clients are connected to the server to start the training...")

    # Define Flower client
    class Client(fl.client.NumPyClient):

        def get_parameters(self):
            """Get parameters of the local model."""
            raise Exception("Not implemented (server-side parameter initialization)")

        def fit(self, parameters, config):
            global round_counter
            check_flag.empty()
            if round_counter == 1:
                with st.spinner('Connecting to server...'):
                    connect_server = st.empty()
                    connect_server.image("pictures/connectServer.png")
                    time.sleep(2)
                    connect_server.success("Successful to Server connected")
                    connect_server.empty()

                st.info('The required number of clients have connected to the server!')

            #########################################################################################################
                with st.spinner("Model is loaded from the server..."):
                    time.sleep(2)
                    st.success('Model successfully loaded from server!')
                # Load new model architecture as dic
                model_arch_dic = config["model"]
                # Convert dic to str object
                model_arch_str = json.dumps(model_arch_dic)

                # Return a json object
                json_obj = json.loads(model_arch_str)

                # Parses a JSON model configuration string and returns a model instance.
                # create a new model from the JSON specification
                loaded_model = tf.keras.models.model_from_json(json_obj)

                # Compiler wurde nicht übergeben, daher muss das Model compiliert werden
                loaded_model.compile(optimizer=config["optimizer"], loss=config["loss"],
                                     metrics=[config["metrics"]])
                self.model = loaded_model

                # übergabe von Hyperparameter für lokales training
                global batch_size
                batch_size = config["local_epochs"]
                global epochs
                epochs = config["batch_size"]
            #########################################################################################################

            st.header(f"Round {round_counter} of federated Training")
            if round_counter == 1:
                with st.spinner("Receiving initial parameters from Server"):
                    rec_initial = st.empty()
                    rec_initial.image("pictures/rec_initial.gif")
                    time.sleep(3)
                    rec_initial.empty()
                    st.session_state["initial_weights"] = self.model.get_weights()
                    received_weights = st.session_state["initial_weights"]
                    st.success('Initial parameters loaded successfully!')

            if round_counter > 1:
                # resetting output
                global waiting_server
                waiting_server.empty()

                # receiving model
                rec_global = st.empty()
                rec_global.image("pictures/rec_global.gif")
                time.sleep(3)
                rec_global.empty()
                self.model.set_weights(parameters)
                received_weights = self.model.get_weights()
                st.success("Successfully received global model parameter!")

            # train model
            with st.spinner(f"Global model is being trained on local data"):
                training = st.empty()
                training.image("pictures/training.gif")
                r = self.model.fit(x_train, y_train, epochs=config["local_epochs"], batch_size=config["batch_size"])
                fed_score = self.model.evaluate(x_test, y_test, verbose=0)
                time.sleep(5)
                training.empty()
                st.success(f'Training  successfully completed!')

            # send back
            send_model = st.empty()
            send_model.image("pictures/send_model.gif")
            time.sleep(3)
            send_model.empty()
            st.success("Successfully send updated model to server!")

            # show accuracy
            hist = r.history
            fed_train_acc.append(hist["accuracy"][-1])
            fed_val_score.append(fed_score[1])
            st.session_state["fedround_weights"] = self.model.get_weights()
            st.write(f"An accuracy of {np.round(fed_score[1], 3)} on unknown data was achieved in this round.")

            # show weights
            with st.expander(f"Summary of exchanged weights from round {round_counter} of federated training"):
                col5, col6 = st.columns(2)
                with col5:
                    st.write("Received weights")
                    st.write(f"The weights received are {sys.getsizeof(received_weights)} bytes large.")
                    st.write(received_weights)
                with col6:
                    st.write("Calculated weights")
                    st.write(f"The weights calculated are{sys.getsizeof(self.model.get_weights())} bytes large.")
                    st.write(self.model.get_weights())

            waiting_server = st.empty()
            waiting_server.info("Waiting for updated weights from server ...")
            round_counter += 1

            return self.model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            self.model.set_weights(parameters)
            self.model.save("fit_global_model")
            loss, accuracy = self.model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("localhost:8080", client=Client())

    # set train local model
    model_local = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model_local.compile(optimizer="adam",
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"])

    # resetting output
    global waiting_server
    waiting_server.empty()

    # training only on local data
    with st.spinner("For comparison, the training is now executed on the local data..."):
        for _ in range(round_counter-1):
            local_trained_model = model_local.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

            local_score = model_local.evaluate(x_test, y_test, verbose=0)
            local_val_score.append(local_score[1])

            fit_hist = local_trained_model.history
            local_train_acc.append(fit_hist["accuracy"][-1])

    # results from local training
    df_val_temp = pd.DataFrame(data=local_val_score, columns=["local (val)"])
    df_fit_temp = pd.DataFrame(data=local_train_acc, columns=["local (fit)"])

    # results from federated training
    df_fed_fit_temp = pd.DataFrame(data=fed_train_acc, columns=["federated (fit)"])
    df_fed_val_temp = pd.DataFrame(data=fed_val_score, columns=["federated (val)"])

    # show result in one DataFrame
    result = pd.concat([df_val_temp, df_fed_val_temp, df_fit_temp, df_fed_fit_temp], axis=1)

    # Appending result from current round to dataframe where all results are stored
    st.session_state["result"] = result
    st.session_state["result"].index += 1

    # show final results
    st.header("Result of Federated learning vs. training with local data only")
    st.metric(label="Accuracy on unknown data for federated training", value=np.round(result["federated (val)"].iloc[-1], 3))
    st.metric(label="Accuracy on unknown data for training with local data only", value=np.round(result["local (val)"].iloc[-1], 3))
