# import benötiger Pakete
import streamlit as st
import flwr as fl
from os.path import exists
import sys
import tensorflow as tf
from tensorflow import keras

import time
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from streamlit_drawable_canvas import st_canvas
from tensorflow.python.keras.models import load_model
import cv2
import json
import plotly.express as px

# for caputring stdout
import contextlib
import io

st.set_page_config(
    page_title="Federated Learning Demonstrator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
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

st.title("Information")
st.markdown("**Here you can find basic information about our demonstrator.**")

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

explain.markdown("Herkömmliche Maschine Learning (ML) Algorithmen werden in der Regel auf Daten trainiert, die von "
                 "verschiedenen Endgeräten wie Mobiltelefonen, Laptops usw. gesammelt und auf einem zentralen Server "
                 "zusammengeführt werden. Die ML- Algorithmen werden dann auf diesen Daten trainiert, um so eine "
                 "Prognose auf neuen Daten zu erstellen. Bei den herkömmlichen ML-Methoden werden auch sensible "
                 "Nutzerdaten an die Server gesendet, wie Chat-Nachrichten oder Bilder. Um dies zu verhindern wurde "
                 "2016 das Konzept des Föderierten Lernens (FL) (engl. federated learning) eingeführt. "
                 "Beim **Föderierten Lernen** können ML-Algorithmen auf lokalen Daten trainiert werden, ohne dass "
                 "Daten mit einer zentralen Instanz ausgetauscht werden. Dies ist ein enormer Vorteil in Sachen Datenschutz.")
explain.markdown("Ein „Federated Learning System“ (FLS) besteht aus mehreren Clients und einem Server. Unter den Clients "
                 "werden meist Endgeräte wie z.B. Mobiltelefone verstanden, welche die lokalen Daten erzeugen. Der "
                 "Server orchestriert das Lernen. Er gibt vor, wie viele Runden trainiert werden, wie viele Clients "
                 "am Training teilnehmen sollen, welche Hyperparameter verwendet werden sollen usw.. Das Föderierte "
                 "Training findet dann immer nach den gleichen Schritten statt.")
explain.markdown("1.	Zuerst laden alle Clients das aktuelle ML-Modell (wie z.B. Logistische Regression, CNN, FFNN) "
                 "vom Server herunter.")
explain.markdown( "2.	Die Clients trainieren das Modell mit ihren lokalen Daten. Dies führt dazu, dass die Parameter"
                 "(meist die Gewichte eines neuronalen Netztes) angepasst werden. Da jeder Client unterschiedliche Daten"
                  "hat, werden auch die Modelle aller Clients nach dem Training unterschiedliche Parameter haben. ")
explain.markdown("3.	Ist das Training der Clients auf den lokalen Daten abgeschossen, senden sie nur die "
                 "angepassten Parameter an den Server zurück.")
explain.markdown("4.	Der Server aggregiert alle erhaltenen Parameter zu einem neuen angepassten Modell.")
explain.markdown("Die Schritte 1-4 werden mehrmals ausgeführt. So ist es möglich ein globales ML-Modell zu trainieren,"
                 "ohne dass die lokalen Daten mit einer zentralen Instanz ausgetauscht werden müssen. "
                 "(vgl. McMahan et al., 2017)")

setup = st.container()
setup.subheader('Aufbau des Demonstrators')

setup.markdown(
    "Der Demonstrator soll dir die funktionsweise des föderrierten Lernen an einem praktischen Beispiel näher bringen."
    "Den Computer den du gerade benutzt fungiert als ein Client. Auf diesem Clients können Daten erzeugt werden "
    "(hier Zahlen zwischen 0-9). Unser Ziel ist ein KI-Modell zu erzeugen, das aufgrundlage unserer erzeugten Daten später "
    "die Zahlen 0-9 klassifizieren kann. Wir wollen aber das keine zentrale Instanz unsere erzeugten Zahlen einsehen kann und "
    "weil ein Client nur sehr wenige Zahlen erzeugen kann wäre es hilfreich die Zahlen die andere Clients erzeugen für unser Modell "
    "verwenden zu können ohne das wir die Zahlen des anderen Clients anschauen können."
    "Das föderrierte Training kann genau das leisten."
    ""
    ""
    "Wenn du der Meinung bist genügend Zahlen erzeugt zu haben, kannst du auf den Knopf **Training starten** klicken."
    "Dann Verbindet sich dein Client mit dem Server für föderriertes Lernen"
    "Der Demonstrator zeigt die dann ich echtzeit was er gerade macht und welche zwischen Ergebnisse erzeugt wurden. "
    "Viel Spass bei verwenden")

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.empty()
with col2:
    st.image("pictures/Aufbau.png")
with col3:
    st.empty()


# global variables
round_counter = 1
initial_weights = []
perround_weights = []
fed_weights = []


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
        st.write("Daten wurden erfolgreich geladen.")
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
        st.write("Training kann noch nicht gestartet werden, da zu wenig Daten erzeugt wurden.")

    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train, x_test, y_test = x_train, y_train, x_test, y_test

    # reset round counter
    #global round_counter
    round_counter = 1

    # For display status
    #check_flag = st.empty()
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
    check_flag.write(
        "Server prüft ob benötige Anzahl von Clients mit dem Server verbunden sind, um das Training starten zu können...")

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):

        def get_parameters(self):

            """with st.spinner('Mit Server verbinden...'):
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
            check_flag.write("Server prüft ob benötige Anzahl von Clients mit dem Server verbunden sind, um das Training starten zu können...")"""


        def fit(self, parameters, config):
            global round_counter
            check_flag.empty()
            if round_counter == 1:
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
            if round_counter == 1:
                with st.spinner("Initale Parameter werden vom geladen..."): # vom Server?
                    time.sleep(2)
                    global initial_weights
                    initial_weights = self.model.get_weights()
                    st.success('Initale Parameter erfolgreich geladen!') # vom Server?

            if round_counter > 2:
                with st.expander(f"Empfangene Gewichte vom Server"):
                    self.model.set_weights(parameters)
                    st.write(f"Die empfangen Gewichte sind {sys.getsizeof(self.model.get_weights())} Bytes groß.")
                    st.write(self.model.get_weights())
                    fig = px.imshow(self.model.get_weights()[2])
                    st.plotly_chart(fig)

            with st.spinner(f"Wir befinden uns gerade in Runde {round_counter} des föderrierten Trainings... "):
                self.model.set_weights(parameters)
                r = self.model.fit(x_train, y_train, epochs=2, batch_size=32)
                fed_score = self.model.evaluate(x_test, y_test, verbose=0)
                self.model.save("fit_global_model")
                st.success(f'Training der Runde {round_counter} erfolgreich beendet und aktualisiertes Modell mit angepassten Gewichten wurde erfolgreich an Server zurück geschickt!')

            hist = r.history
            train_acc = hist["accuracy"][-1]
            fed_train_acc.append(hist["accuracy"][-1])
            fed_val_score.append(fed_score[1])
            global perround_weights
            perround_weights = self.model.get_weights()

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

    with st.spinner("Zum Vergleich wird jetzt noch das Training auf den lokalen Daten  ausgeführt..."):
        for _ in range(5):
            captured_output_local = io.StringIO()
            with contextlib.redirect_stdout(captured_output_local):
                local_trained_model = model_local.fit(x_train, y_train, epochs=2, batch_size=32)
            st.session_state["local_log"] = captured_output_local.getvalue()

            local_score = model_local.evaluate(x_test, y_test, verbose=0)
            local_val_score.append(local_score[1])

            fit_hist = local_trained_model.history
            local_train_acc.append(fit_hist["accuracy"][-1])

        st.success("Lokales Training abgeschlossen")

    with st.expander("Hier kannst du die Ergebnisse der letzten Runde zusammengefasst anschauen."):
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

st.title("Ergebnisseite")
st.sidebar.markdown("# Ergebnisseite")
result = st.session_state["result"]


try:
    col7, col8 = st.columns(2)
    with col7:
        st.write("Vergleich der Genauigkeit auf den Trainingsdaten pro FL-Runde.")

        # show results from current FL Durchgang
        st.dataframe(result[["Lokal (fit)", "Föderiert (fit)"]])

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
        st.dataframe(result[["Lokal (val)", "Föderiert (val)"]])

        # plot result
        fig_val = px.line(result[["Lokal (val)", "Föderiert (val)"]])
        fig_val.update_layout(
            title="Genauigkeit Modelle per Runde",
            xaxis_title="FL-Runden",
            yaxis_title="Genauigkeit",
            legend_title="Modelle")
        fig_val.update_yaxes(range=(0.0, 1.0))
        st.plotly_chart(fig_val)


except:
    st.write("Es wurde noch kein Training durchgeführt!")
    pass

st.subheader("Klassifizierung")
st.write("Nun prüfen wir ob unser föderriertes Modell auch eine gute Vorhersage treffen kann.")

#load_model for classification
model = load_model('fit_global_model')

col3, col4 = st.columns(2)

with col3:
    SIZE = 192
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


st.subheader("Log-Files")
with st.expander("Hier findest du die Log Daten des lokalen Trainings"):
        st.write(st.session_state["local_log"])

with st.expander("Hier findest du die Log Daten des Föderierten Trainings"):
        st.write(st.session_state["fed_log"])
