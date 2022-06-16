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

# for caputring stdout
import contextlib
import io

#import p3_result_V1 as st_ru

# streamlit config
st.set_page_config(layout='wide') #centered

# global variables
round_counter = 0
local_weights = []
fed_weights = []
initial_weights = []

local_hist = []
fit_hist = []
fed_hist = []
fed_eval_hist = []


# Sidebar
with st.sidebar:
    st.subheader("Hier ist die Kommandzentrale")
    train_button = st.button("Am Föderierten Training teilnehmen")
    load_data_button = st.button("Daten einlesen.")
    reset_button = st.button("Daten zurücksetzen")

################################################## data_input ###################################################
st.header("Dateneingabe")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Jetzt bist du gefragt! Generiere Daten für unseren Demonstrator.**")
    st.markdown("Wie in der rechten Abbildung gezeigt!")
with col2:
    st.image("pictures/how_to_draw_192.gif")

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
    number_to_draw = st.session_state["number"]
    st.markdown(f"### Zeichne die Nummer: {number_to_draw}")

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

    save_button = st.button("Speichern")

with col4:
    st.markdown(
        "So sieht der Computer die Zahlen. Alle 0-Werte stehen für die Farbe schwarz und alle Werte um die 255 für die Farbe weiß.")
    st.markdown("In dieser Form werden die Zahlen dem neuronalen Netz übergeben!")
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

st.markdown("**Auflistung der Anzahl schon gezeichneter Zahlen**")
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
        st.write("Es wurden noch keine Daten gespeichert!")


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
st.write("Zuletzt gezeichnete Zahlen:")
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
if train_button:

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
        st.write("Training kann noch nicht gestartet werden, da zu wenig Daten erzeugt wurden.")

    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train, x_test, y_test = x_train, y_train, x_test, y_test

    # reset round counter
    #global round_counter
    round_counter = 0

    # For display status
    check_flag = st.empty()

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):

        def get_parameters(self):

            with st.spinner('Mit Server verbinden...'):
                time.sleep(2)
                st.success('Erfolgreich mit Server Verbunden!')

            with st.expander("Klicke hier für detailierte Information zu den Trainingsdaten."):
                st.markdown("### Aufteilung von Training- und Testdaten")
                st.write(f"x_train shape: {x_train.shape}")
                st.write(f"{x_train.shape[0]} train samples")
                st.write(f"{x_test.shape[0]} test samples")
                st.write(f"Die Trainingsdaten sind {sys.getsizeof(x_train)} Bytes groß.")

            global check_flag
            check_flag = st.empty()
            check_flag.write("Server prüft ob benötige Anzahl von Clients mit dem Server verbunden sind, um das Training starten zu können...")


        def fit(self, parameters, config):
            global round_counter
            check_flag.empty()
            if round_counter == 0:
                st.success('Die benötige Anzahl an Clients haben sich mit dem Server verbunden!')
                with st.spinner('Training wird gestartet...'):
                    time.sleep(1)
                    st.success(f'Training auf den {sum(all_numbers)} erzeugten Zahlen erfolgreich gestartet!')

            #########################################################################################################
                with st.spinner("Aktuelles Modell wird vom Server geladen..."):
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
                    st.success('Aktuelles Modell erfolgreich vom Server geladen!')

            #########################################################################################################
            if round_counter == 0:
                with st.spinner("Initale Parameter werden vom geladen..."): # vom Server?
                    time.sleep(2)
                    global initial_weights
                    initial_weights = self.model.get_weights()
                    st.success('Initale Parameter erfolgreich geladen!') # vom Server?

            if round_counter > 1:
                with st.expander(f"Empfangene Gewichte vom Server"):
                    self.model.set_weights(parameters)
                    st.write(f"Die empfangen Gewichte sind {sys.getsizeof(self.model.get_weights())} Bytes groß.")
                    st.write(self.model.get_weights())

            with st.spinner(f"Wir befinden uns gerade in Runde {round_counter} des föderrierten Trainings... "):
                self.model.set_weights(parameters)
                r = self.model.fit(x_train, y_train, epochs=2, batch_size=32)
                self.model.save("fit_global_model")
                st.success(f'Training der Runde {round_counter} erfolgreich beendet und aktualisiertes Modell mit angepassten Gewichten wurde erfolgreich an Server zurück geschickt!')

            hist = r.history
            train_acc = hist["accuracy"][-1]
            fed_hist.append(hist["accuracy"][-1])
            global local_weights
            local_weights = self.model.get_weights()

            st.write(f"Es wurde in dieser Runde eine Genauigkeit von {train_acc} auf den Trainingsdaten erreicht.")

            with st.expander(f"Berechnete Gewichte der Runde {round_counter}"):
                st.write(f"Die berechneten Gewichte sind {sys.getsizeof(self.model.get_weights())} Bytes groß.")
                st.write(self.model.get_weights())

            st.info("Warte auf aktualisierte Gewichte von Server ...")
            round_counter += 1

            return self.model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            self.model.set_weights(parameters)
            loss, accuracy = self.model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    captured_output_fed = io.StringIO()
    with contextlib.redirect_stdout(captured_output_fed):
        fl.client.start_numpy_client("localhost:8080", client=CifarClient())
    fed_train_log = captured_output_fed.getvalue()

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
            st.write(local_weights)

    with st.spinner("Zum Vergleich wird jetzt noch das Training auf den lokalen Daten  ausgeführt..."):
        for _ in range(5):
            captured_output_local = io.StringIO()
            with contextlib.redirect_stdout(captured_output_local):
                model_local.fit(x_train, y_train, epochs=2, batch_size=32)
            local_train_log = captured_output_local.getvalue()

            score = model_local.evaluate(x_test, y_test, verbose=0)
            local_hist.append(score[1])
        st.success("Lokales Training abgeschlossen")

    if len(local_hist) != 0:
        dfeval_temp = pd.DataFrame(data=local_hist, columns=["Local Accuracy per Round (eval)"])
        dffit_temp = pd.DataFrame(data=fit_hist, columns=["Local Accuracy per Round (fit)"])
        # st.dataframe(df1_temp)

    if len(fed_hist) != 0:

        df_fed_fit_temp = pd.DataFrame(data=fed_hist, columns=["Federated Accuracy per Round (fit)"])  # columns=["Ep1", "Ep2", "Ep3", "Ep4", "Ep5", "Ep6", "Ep7", "Ep8", "Ep9", "Ep10"]
        df_fed_val_temp = pd.DataFrame(data=fed_eval_hist, columns=["Federated Accuracy per Round (eval)"])

        # st.dataframe(df2_temp)
        # st.line_chart(fed_hist)

        result = pd.concat([dfeval_temp, dffit_temp, df_fed_fit_temp, df_fed_val_temp], axis=1)
        st.dataframe(result)
        st.line_chart(result)

    else:
        st.write("Es wurde noch kein Training durchgeführt!")


