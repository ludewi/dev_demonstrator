"""Version 3: Modell wird von Server empfangen
    abd result shown directly"""

# import benötiger Pakete
import streamlit as st
import flwr as fl
import os
from os.path import exists
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from keras_sequential_ascii import keras2ascii
import time
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from streamlit_drawable_canvas import st_canvas
import cv2
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






################################################## data_input ###################################################
st.header("Data input")

st.subheader("Control Center")
col11, col22 = st.columns(2)
with col11:
    st.write("If you don't want to draw a lot of numbers yourself, you can also load a prebuilt dataset.")
    load_data_button = st.button("Import data")
with col22:
    st.write("hier kommt noch was hin.")
    reset_button = st.button("Reset data")


# um Platz zwischen die Elemente zubekommen.
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")

# set session_state for number counter
if "counter_0" not in st.session_state:
    st.session_state["counter_0"] = 1

if "counter_1" not in st.session_state:
    st.session_state["counter_1"] = 0

if "counter_2" not in st.session_state:
    st.session_state["counter_2"] = 0

if "counter_3" not in st.session_state:
    st.session_state["counter_3"] = 0

if "counter_4" not in st.session_state:
    st.session_state["counter_4"] = 0

if "counter_5" not in st.session_state:
    st.session_state["counter_5"] = 0

if "counter_6" not in st.session_state:
    st.session_state["counter_6"] = 0

if "counter_7" not in st.session_state:
    st.session_state["counter_7"] = 0

if "counter_8" not in st.session_state:
    st.session_state["counter_8"] = 0

if "counter_9" not in st.session_state:
    st.session_state["counter_9"] = 0

if "counter" not in st.session_state:
    st.session_state["counter"] = 0

if "number" not in st.session_state:
    st.session_state["number"] = 0

if "image" not in st.session_state:
    st.session_state["image"] = []

if "y_train" not in st.session_state:
    st.session_state["y_train"] = []

# session state for result over all FL rounds
if "result" not in st.session_state:
    st.session_state["result"] = pd.DataFrame()

if "local_log" not in st.session_state:
    st.session_state["local_log"] = ""

if "fed_log" not in st.session_state:
    st.session_state["fed_log"] = ""


all_numbers = [st.session_state["counter_0"], st.session_state["counter_1"], st.session_state["counter_2"],
               st.session_state["counter_3"], st.session_state["counter_4"], st.session_state["counter_5"],
               st.session_state["counter_6"], st.session_state["counter_7"], st.session_state["counter_8"],
               st.session_state["counter_9"]]

# Tabel of the number of numbers already drawn
d = {'0er': st.session_state["counter_0"], '1er': st.session_state["counter_1"], '2er': st.session_state["counter_2"],
     '3er': st.session_state["counter_3"], '4er': st.session_state["counter_4"], '5er': st.session_state["counter_5"],
     '6er': st.session_state["counter_6"], '7er': st.session_state["counter_7"], '8er': st.session_state["counter_8"],
     '9er': st.session_state["counter_9"], "Gesamt": sum(all_numbers)}

col3, col4 = st.columns([1, 3])
with col3:
    st.markdown("**Now it's your turn! Generate data for our demonstrator.**")
    st.info("Important!!! First draw the number then press save button and then the trash can icon!")
    number_to_draw = st.session_state["number"]
    st.markdown(f"### Draw the number: {number_to_draw}")

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=20,
        stroke_color="#ffffff",
        background_color="#000000",
        update_streamlit=True,
        height=300,
        width=300,
        display_toolbar=True,
        key="MNIST_input")

    save_button = st.button("Save image")
    
    
    st.image("pictures/how_to_draw_192.gif")

    

with col4:
    st.markdown(
        "This is how the computer sees the numbers. All 0-values stand for the color black and all values around 255 for the color white.")
    st.markdown("In this form the numbers are passed to the neural network!")
    if canvas_result.image_data is not None:
        image_temp = canvas_result.image_data
        image2 = image_temp.copy()
        image2 = image2.astype('uint8')
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        image2 = cv2.resize(image2, (28, 28))
        df_1 = pd.DataFrame(data=image2)
        st.dataframe(df_1, height=700)

st.write("")
st.write("")
st.write("")

st.markdown("**Listing already drawn numbers**")
df = pd.DataFrame(data=d, index=["Anzahl"])
st.dataframe(df)

# Für die Daten
images = st.session_state["image"]
y_train = st.session_state["y_train"]

if load_data_button:
    # check if .npz file exists
    if exists("image_data.npz"):
        # load saved numpy array with image data (28x28) and target data
        arr = np.load("image_data.npz")

        # save image data in variable and convert from numpy array into python array
        # only then appending works with different dimension arrays eg. (5,28,28).append(1,28,28)
        np_images = arr["x"]
        st.session_state["image"] = np_images.tolist()
        np_y_train = arr["y"]
        st.session_state["y_train"] = np_y_train.tolist()
        st.write("Data was loaded successfully.")
        st.write(np.shape(st.session_state["image"]))
        st.write(np.shape(st.session_state["y_train"]))

        # Update already drawn numbers
        st.session_state["counter_0"] = np.count_nonzero(np_y_train == 0)#np_y_train.count("0")
        st.session_state["counter_1"] = np.count_nonzero(np_y_train == 1)#np_y_train.count("1")
        st.session_state["counter_2"] = np.count_nonzero(np_y_train == 2)#np_y_train.count("2")
        st.session_state["counter_3"] = np.count_nonzero(np_y_train == 3)#np_y_train.count("3")
        st.session_state["counter_4"] = np.count_nonzero(np_y_train == 4)#np_y_train.count("4")
        st.session_state["counter_5"] = np.count_nonzero(np_y_train == 5)#np_y_train.count("5")
        st.session_state["counter_6"] = np.count_nonzero(np_y_train == 6)#np_y_train.count("6")
        st.session_state["counter_7"] = np.count_nonzero(np_y_train == 7)#np_y_train.count("7")
        st.session_state["counter_8"] = np.count_nonzero(np_y_train == 8)#np_y_train.count("8")
        st.session_state["counter_9"] = np.count_nonzero(np_y_train == 9)#np_y_train.count("9")

        # when drawing first number file does not exists
    else:
        st.error("No data has been saved yet!")


# when pressing save_button save all steamlit_canvas data into variable
if save_button:
    # Anzeige welche Zahlen gezeichnet werden müssen
    number = np.random.randint(0, high=10, size=None, dtype=int)
    st.session_state["number"] = number
    if canvas_result.image_data is not None:
        image = canvas_result.image_data
        image1 = image.copy()
        image1 = image1.astype('uint8')
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image1 = cv2.resize(image1, (28, 28))
        images.append(image1)
        y_train.append(number)

        # save numpy array persistence into .npz file
        np.savez("image_data.npz", x=images, y=y_train)

        # Counter to check how many numbers were drawn
        if number == 0:
            st.session_state["counter_0"] += 1
        if number == 1:
            st.session_state["counter_1"] += 1
        if number == 2:
            st.session_state["counter_2"] += 1
        if number == 3:
            st.session_state["counter_3"] += 1
        if number == 4:
            st.session_state["counter_4"] += 1
        if number == 5:
            st.session_state["counter_5"] += 1
        if number == 6:
            st.session_state["counter_6"] += 1
        if number == 7:
            st.session_state["counter_7"] += 1
        if number == 8:
            st.session_state["counter_8"] += 1
        if number == 9:
            st.session_state["counter_9"] += 1

# Output of drawn numbers
st.write("Last drawn numbers:")
try:
    col0, col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(10)
    col0.image(images[-1])
    col1.image(images[-2])
    col2.image(images[-3])
    col3.image(images[-4])
    col4.image(images[-5])
    col5.image(images[-6])
    col6.image(images[-7])
    col7.image(images[-8])
    col8.image(images[-9])
    col9.image(images[-10])
except:
    pass

################################################## start_training ###################################################
train_button = st.button("Participate in the Federated Training")

if train_button:
    # reset score lists
    local_val_score = []
    local_train_acc = []
    fed_train_acc = []
    fed_val_score = []
    ################################################## Prepare Data ###################################################
    # Load dataset
    np_x_train = np.array(images)
    np_y_train = np.array(y_train)

    # Skalieren der Daten
    x_train_norm = []
    for i in range(len(np_x_train)):
        x_train_norm.append(np_x_train[i] / 255)

    # reshaping the Data
    x_train = np.array(x_train_norm).reshape(-1, 28, 28, 1)

    # shuffle  data
    if len(y_train) > 10:  # da fehler meldung wenn noch keine daten erzeugt wurden
        X = x_train
        y = np_y_train

        X, y = shuffle(X, y, random_state=0)

        # test train split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    else:
        st.error("Training cannot be started yet, because too little data has been generated.")

    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train, x_test, y_test = x_train, y_train, x_test, y_test

    # reset round counter
    #global round_counter
    round_counter = 1

    # For display status
    #check_flag = st.empty()
    

    with st.spinner('Connect to server...'):
        time.sleep(2)
        st.success('Successfully connected to server!')

    with st.expander("Click here for detailed information about the training data."):
        st.markdown("### Splitting of training and test data")
        st.write(f"x_train shape: {x_train.shape}")
        st.write(f"{x_train.shape[0]} train samples")
        st.write(f"{x_test.shape[0]} test samples")
        st.write(f"The training data are {sys.getsizeof(x_train)} Bytes large.")

    global check_flag
    check_flag = st.empty()
    check_flag.write(
        "Server checks if required number of clients are connected to the server to start the training...")

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):

        def get_parameters(self):
            """Get parameters of the local model."""
            raise Exception("Not implemented (server-side parameter initialization)")


        def fit(self, parameters, config):
            global round_counter
            check_flag.empty()
            if round_counter == 1:
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
                    self.model.save("fit_global_model") ### model speichern 
                    st.write(f"The weights received are {sys.getsizeof(self.model.get_weights())} bytes large.")
                    st.write(self.model.get_weights())
                    fig = px.imshow(self.model.get_weights()[2])
                    st.plotly_chart(fig)

            with st.spinner(f"We are currently in round {round_counter} of federated training..."):
                self.model.set_weights(parameters)
                r = self.model.fit(x_train, y_train, epochs=2, batch_size=32)
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
    with st.spinner("Wait until the training has been finished!"):
        with contextlib.redirect_stdout(captured_output_fed):
            fl.client.start_numpy_client("localhost:8080", client=CifarClient())
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

    with st.expander("Vergleich empfangen Modell vor Training vs. gesendet Modell mit angepassten Gewichten"):
        st.write("Das sind die Daten die ein Client vom Server erhält (siehe links) und die Daten die der Client zurück an den Server sendet. (siehe rechts)")
        st.write("Wie zu erkennen ist haben diese Zahlen nichts mehr mit den Zahlen zu tun die erzeugt werden, wenn eine Zahl gezeichnet wird.")

        col5, col6 = st.columns(2)
        with col5:
            st.write("Inital erhaltende Gewichte vom Server")
            st.write(initial_weights)
        with col6:
            st.write("Angepasste Gewichte des Modells (Training der letzten Runde auf lokalen Daten)")
            st.write(perround_weights)

    with st.spinner("For comparison, the training is now executed on the local data..."):
        for _ in range(5):
            captured_output_local = io.StringIO()
            with contextlib.redirect_stdout(captured_output_local):
                local_trained_model = model_local.fit(x_train, y_train, epochs=2, batch_size=32)
            st.session_state["local_log"] = captured_output_local.getvalue()

            local_score = model_local.evaluate(x_test, y_test, verbose=0)
            local_val_score.append(local_score[1])

            fit_hist = local_trained_model.history
            local_train_acc.append(fit_hist["accuracy"][-1])

        st.success("Local training finished")

    with st.expander("Here you can see a summary of the results of the last round."):
        # results from local training
        df_val_temp = pd.DataFrame(data=local_val_score, columns=["Lokal (val)"])
        df_fit_temp = pd.DataFrame(data=local_train_acc, columns=["Lokal (fit)"])

        # results from federated training
        df_fed_fit_temp = pd.DataFrame(data=fed_train_acc, columns=["Föderiert (fit)"])
        df_fed_val_temp = pd.DataFrame(data=fed_val_score, columns=["Föderiert (val)"])

        # show result in one DataFrame
        result = pd.concat([df_val_temp, df_fed_val_temp, df_fit_temp, df_fed_fit_temp], axis=1)

        # Appending result from current round to dataframe where all results are stored
        st.session_state["result"] = pd.concat([st.session_state["result"], result], ignore_index=True, axis=0)
        st.session_state["result"].index += 1
        result.index += 1

        col7, col8 = st.columns(2)
        with col7:
            st.write("Vergleich der Genauigkeit auf den Trainingsdaten pro FL-Runde.")

            # show results from current FL Durchgang
            st.dataframe(result[["Lokal (fit)", "Föderiert (fit)"]], width=2000)

            # plot result
            fig_fit = px.line(result[["Lokal (fit)", "Föderiert (fit)"]])
            fig_fit.update_layout(
                title="Genauigkeit Modelle per Runde",
                xaxis_title="FL-Runden",
                yaxis_title="Genauigkeit",
                legend_title="Modelle")
            fig_fit.update_yaxes(range=(0.0, 1.0))
            st.plotly_chart(fig_fit)

        with col8:
            st.write("Vergleich der Genauigkeit auf den Validierungsdaten pro FL-Runde.")

            # show results from current FL Durchgang
            st.dataframe(result[["Lokal (val)", "Föderiert (val)"]], width=2000)

            # plot result
            fig_val = px.line(result[["Lokal (val)", "Föderiert (val)"]])
            fig_val.update_layout(
                title="Genauigkeit Modelle per Runde",
                xaxis_title="FL-Runden",
                yaxis_title="Genauigkeit",
                legend_title="Modelle")
            fig_val.update_yaxes(range=(0.0, 1.0))
            st.plotly_chart(fig_val)

