# import benötiger Pakete
import streamlit as st
import flwr as fl
import os
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

# for caputring stdout
import contextlib
import io



# streamlit config
st.set_page_config(layout='wide') #centered

# global variables
round_counter = 0
local_weights = []
fed_weights = []
initial_weights = []

local_hist = []
fed_hist = []
local_train_log = ""
fed_train_log = ""


#### input ####

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

# Anzeige welche Zahlen gezeichnet werden müssen
if save_button:
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

################################################## Prepare Data ###################################################
# Load dataset
(X, y_train) = (images, y_train)

# Skalieren der Daten
x_train = []
for image in X:
    x_train.append(image / 255.0)

# reshaping the Data
x_train = np.array(x_train).reshape(-1, 28, 28, 1)
y_train = np.array(y_train)

# shuffle  data
if len(y_train) > 10:  # da fehler meldung wenn noch keine daten erzeugt wurden
    X = x_train
    y = y_train

    X, y = shuffle(X, y, random_state=0)

    # test train split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

else:
    st.write("Training kann noch nicht gestartet werden, da zu wenig Daten erzeugt wurden.")

################################################## start_training ###################################################

# Train Lokal or Fed
with st.sidebar:
    st.subheader("Hier ist die Kommandzentrale")
    train_button = st.button("Am Föderierten Training teilnehmen")
    reset_button = st.button("Daten zurücksetzen")


if train_button:
    # Make TensorFlow log less verbose
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    with st.spinner('Mit Server verbinden...'):
        time.sleep(2)
        st.success('Erfolgreich mit Server Verbunden!')
    # Load model and data (MobileNetV2, CIFAR-10)
    with st.spinner("Aktuelles Modell wird vom Server geladen..."):
        time.sleep(4)
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        st.success('Aktuelles Modell erfolgreich vom Server geladen!')
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        #x_train, y_train, x_test, y_test = x_train, y_train, x_test, y_test


        check_flag = st.empty()

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):

        def get_parameters(self):
            with st.spinner("Aktuelle Parameter werden vom Server geladen..."):
                time.sleep(2)
                global initial_weights
                initial_weights = model.get_weights()
                st.success('Aktuelle Parameter erfolgreich vom Server geladen!')
            with st.expander("Klicke hier für detailierte Information zur Modellarchitektur und Trainingsdaten."):
                col8, col9 = st.columns(2)
                with col8:
                    st.markdown("### Geladene Modell Architektur")
                    captured_output_model = io.StringIO()
                    with contextlib.redirect_stdout(captured_output_model):
                        keras2ascii(model)
                    captured_model = captured_output_model.getvalue()
                    st.write(captured_model)
                with col9:
                    st.markdown("### Aufteilung von Training- und Testdaten")
                    st.write(f"x_train shape: {x_train.shape}")
                    st.write(f"{x_train.shape[0]} train samples")
                    st.write(f"{x_test.shape[0]} test samples")

            global check_flag
            check_flag = st.empty()
            check_flag.write("Server prüft ob benötige Anzahl von Clients mit dem Server verbunden sind, um das Training starten zu können...")
            return model.get_weights()

        def fit(self, parameters, config):
            global round_counter
            check_flag.empty()
            if round_counter == 0:
                st.success('Die benötige Anzahl an Clients haben sich mit dem Server verbunden!')
                with st.spinner('Training wird gestartet...'):
                    time.sleep(1)
                    st.success(f'Training auf den {sum(all_numbers)} erzeugten Zahlen erfolgreich gestartet!')

            round_counter += 1

            if round_counter > 1:
                with st.expander(f"Empfangene Gewichte vom Server"):
                    model.set_weights(parameters)
                    st.write(model.get_weights())

            with st.spinner(f"Wir befinden uns gerade in Runde {round_counter} des föderrierten Trainings... "):
                model.set_weights(parameters)
                r = model.fit(x_train, y_train, epochs=2, batch_size=32)
                model.save("fit_global_model")
                st.success(f'Training der Runde {round_counter} erfolgreich beendet und aktualisiertes Modell mit angepassten Gewichten wurde erfolgreich an Server zurück geschickt!')

            with st.expander(f"Berechnete Gewichte der Runde {round_counter}"):
                st.write(model.get_weights())

            hist = r.history
            fed_hist.append(hist["accuracy"][-1])
            global local_weights
            local_weights = model.get_weights()
            st.info("Warte auf aktualisierte Gewichte von Server ...")
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
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

            #calculate whoch % of initial weights got changed
            #weight_diff = np.subtract(initial_weights, local_weights)
            #weight_diff_r = weight_diff[np.where(weight_diff != 0)]
            #st.write(len(weight_diff_r)/len(weight_diff)*100)

    with st.spinner("Zum Vergleich wird jetzt noch das Training auf den lokalen Daten  ausgeführt..."):
        for _ in range(5):
            captured_output_local = io.StringIO()
            with contextlib.redirect_stdout(captured_output_local):
                model_local.fit(x_train, y_train, epochs=2, batch_size=32)
            local_train_log = captured_output_local.getvalue()

            score = model_local.evaluate(x_test, y_test, verbose=0)
            local_hist.append(score[1])
        st.success("Lokales Training abgeschlossen")

show_result = st.button("Ergebnisse anschauen")

if show_result:

    st.subheader("Ergebnisseite")

    if len(local_hist) != 0:
        df1_temp = pd.DataFrame(data=local_hist, columns=["Local Accuracy per Round"])
        # st.dataframe(df1_temp)

    if len(fed_hist) != 0:
        df2_temp = pd.DataFrame(data=fed_hist, columns=[
            "Federated Accuracy per Round"])  # columns=["Ep1", "Ep2", "Ep3", "Ep4", "Ep5", "Ep6", "Ep7", "Ep8", "Ep9", "Ep10"]
        # st.dataframe(df2_temp)
        # st.line_chart(fed_hist)

        result = pd.concat([df1_temp, df2_temp], axis=1)
        st.dataframe(result)
        st.line_chart(result)

    else:
        st.write("Es wurde noch kein Training durchgeführt!")

    st.subheader("Klassifizierung")
    st.write("Nun prüfen wir ob unser föderriertes Modell auch eine gute Vorhersage treffen kann.")

    # load_model for classification
    model = load_model('fit_global_model')

    col3, col4 = st.columns(2)

    with col3:
        SIZE = 192
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=20,
            stroke_color="#000000",
            background_color="#ffffff",
            update_streamlit=True,
            height=300,
            width=300,
            display_toolbar=True,
            key="MNIST_predict")

    with col4:
        if canvas_result.image_data is not None:
            img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
            rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            st.write('Model Input')
            st.image(rescaled)

        if st.button('Predict'):
            test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            val = model.predict(test_x.reshape(1, 28, 28))
            st.write(f'result: {np.argmax(val[0])}')
            st.bar_chart(val[0])

    with st.expander("Hier findest du die Log Daten des lokalen Trainings"):
        st.write(local_train_log)

    with st.expander("Hier findest du die Log Daten des Föderierten Trainings"):
        st.write(fed_train_log)
