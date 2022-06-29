# import benötiger Pakete
import streamlit as st
import flwr as fl
import sys
import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
import plotly.express as px
# for caputring stdout
import contextlib
import io

# streamlit config
st.set_page_config(
     page_title="Data input",
     page_icon="🚀",
     layout="wide",
     initial_sidebar_state="expanded"
    )

# global variables
round_counter = 1
initial_weights = []
perround_weights = []
fed_weights = []

st.title("2. Federated training")

all_numbers = [st.session_state["counter_0"], st.session_state["counter_1"], st.session_state["counter_2"],
               st.session_state["counter_3"], st.session_state["counter_4"], st.session_state["counter_5"],
               st.session_state["counter_6"], st.session_state["counter_7"], st.session_state["counter_8"],
               st.session_state["counter_9"]]

train_button = st.button("------------------------------------------------------------------------------------------------ Click me to participate in the Federated Training ----------------------------------------------------------------------------------------------------")
################################################## start_training ###################################################
if train_button:
    # reset score lists
    local_val_score = []
    local_train_acc = []
    fed_train_acc = []
    fed_val_score = []
    ################################################## Prepare Data ###################################################
    # Load dataset
    np_x_train = np.array(st.session_state["image"])
    np_y_train = np.array(st.session_state["y_train"])

    # Skalieren der Daten
    x_train_norm = []
    for i in range(len(np_x_train)):
        x_train_norm.append(np_x_train[i] / 255)

    # reshaping the Data
    x_train = np.array(x_train_norm).reshape(-1, 28, 28, 1)

    # shuffle  data
    if len(st.session_state["y_train"]) > 10:  # da fehler meldung wenn noch keine daten erzeugt wurden
        X = x_train
        y = np_y_train

        X, y = shuffle(X, y, random_state=0)

        # test train split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    else:
        st.error("Training cannot be started yet, because too little data has been generated.")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #x_train, y_train, x_test, y_test = x_train, y_train, x_test, y_test

    # reset round counter
    #global round_counter
    round_counter = 1

    # For display status
    with st.expander("Click here for detailed information about the training data."):
        st.markdown("### Splitting of training and test data")
        st.write(f"x_train shape: {x_train.shape}")
        st.write(f"{x_train.shape[0]} train samples")
        st.write(f"{x_test.shape[0]} test samples")
        st.write(f"The training data are {sys.getsizeof(x_train)} bytes large.")

    global check_flag
    check_flag = st.empty()
    check_flag.write(
        "Server checks if required number of clients are connected to the server to start the training...")

    # Define Flower client
    class Client(fl.client.NumPyClient):

        def get_parameters(self):
            """Get parameters of the local model."""
            raise Exception("Not implemented (server-side parameter initialization)")

        def fit(self, parameters, config):
            global round_counter
            check_flag.empty()
            if round_counter == 1:
                with st.spinner('Connect to server...'):
                    time.sleep(2)
                    st.success('Successfully connected to server!')

                st.success('The required number of clients have connected to the server!')
                with st.spinner('Training is started...'):
                    time.sleep(1)
                    st.success(f'Training on the {sum(all_numbers)} generated numbers successfully launched!')

            #########################################################################################################
                with st.spinner("Current model is loaded from the server..."):
                    time.sleep(2)
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
                    st.success('Latest model successfully loaded from server!')

            #########################################################################################################
            if round_counter == 1:
                with st.spinner("Initial parameters are loaded from the..."): # vom Server?
                    time.sleep(2)
                    global initial_weights
                    initial_weights = self.model.get_weights()
                    st.success('Initial parameters loaded successfully!') # vom Server?

            if round_counter > 2:
                with st.expander(f"Weights received from server"):
                    self.model.set_weights(parameters)
                    #self.model.save("fit_global_model") ### model speichern
                    st.write(f"The weights received are {sys.getsizeof(self.model.get_weights())} bytes large.")
                    st.write(self.model.get_weights())

            with st.spinner(f"We are currently in round {round_counter} of federated training..."):
                self.model.set_weights(parameters)
                r = self.model.fit(x_train, y_train, epochs=config["local_epochs"], batch_size=config["batch_size"])
                fed_score = self.model.evaluate(x_test, y_test, verbose=0)

                st.success(f'Training of round {round_counter} successfully completed and updated model with adjusted weights successfully sent back to server!')

            hist = r.history
            train_acc = hist["accuracy"][-1]
            fed_train_acc.append(hist["accuracy"][-1])
            fed_val_score.append(fed_score[1])
            global perround_weights
            perround_weights = self.model.get_weights()

            st.write(f"An accuracy of {train_acc} on the training data was achieved in this round.")

            with st.expander(f"Calculated weights of the round {round_counter}"):
                st.write(f"The weights calculated are{sys.getsizeof(self.model.get_weights())} bytes large.")
                st.write(self.model.get_weights())

            st.info("Waiting for updated weights from server ...")
            round_counter += 1

            return self.model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            self.model.set_weights(parameters)
            self.model.save("fit_global_model")
            loss, accuracy = self.model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    captured_output_fed = io.StringIO()
    with contextlib.redirect_stdout(captured_output_fed):
        fl.client.start_numpy_client("localhost:8080", client=Client())
    st.session_state["fed_log"] = captured_output_fed.getvalue()

    ###### train local #####
    model_local = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model_local.compile(optimizer="adam",
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"])

    with st.expander("Comparison of weights from beginning to end"):
        col5, col6 = st.columns(2)
        with col5:
            st.write("Representation of the third layer of initial weights received from the server")
            fig = px.imshow(initial_weights[2])
            st.plotly_chart(fig)
            st.write(initial_weights)

        with col6:
            st.write("Representation of the third layer of the weights after laster round of federated learning")
            fig = px.imshow(perround_weights[2])
            st.plotly_chart(fig)
            st.write(perround_weights)

    with st.spinner("For comparison, the training is now executed on the local data..."):
        for _ in range(round_counter-1):
            captured_output_local = io.StringIO()
            with contextlib.redirect_stdout(captured_output_local):
                local_trained_model = model_local.fit(x_train, y_train, epochs=2, batch_size=32)
            st.session_state["local_log"] = captured_output_local.getvalue()

            local_score = model_local.evaluate(x_test, y_test, verbose=0)
            local_val_score.append(local_score[1])

            fit_hist = local_trained_model.history
            local_train_acc.append(fit_hist["accuracy"][-1])

        st.success("Local training finished")


    # results from local training
    df_val_temp = pd.DataFrame(data=local_val_score, columns=["Lokal (val)"])
    df_fit_temp = pd.DataFrame(data=local_train_acc, columns=["Lokal (fit)"])

    # results from federated training
    df_fed_fit_temp = pd.DataFrame(data=fed_train_acc, columns=["Föderiert (fit)"])
    df_fed_val_temp = pd.DataFrame(data=fed_val_score, columns=["Föderiert (val)"])

    # show result in one DataFrame
    result = pd.concat([df_val_temp, df_fed_val_temp, df_fit_temp, df_fed_fit_temp], axis=1)

    # Appending result from current round to dataframe where all results are stored
    #st.session_state["result"] = pd.concat([st.session_state["result"], result], ignore_index=True, axis=0)
    st.session_state["result"] = result
    st.session_state["result"].index += 1

    st.subheader("Federated learning performs compared to local training as shown below")
    st.metric(label="Accuracy Federated", value=result["Föderiert (val)"].iloc[-1], delta="1.0%")
    st.metric(label="Accuracy Local", value=result["Lokal (val)"].iloc[-1], delta="1.0%")