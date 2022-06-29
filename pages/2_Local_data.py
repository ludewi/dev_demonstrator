"""Version 3: Modell wird von Server empfangen
    abd result shown directly"""

# import benötiger Pakete
import streamlit as st
from os.path import exists
import time
import numpy as np
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import cv2


# streamlit config
st.set_page_config(
     page_title="Data input",
     page_icon="🚀",
     layout="wide",
     initial_sidebar_state="expanded"
    )


################################################## data_input ###################################################
st.title("Local Data")

col11, col22 = st.columns(2)
with col11:
    st.write("If you don't want to draw a lot of numbers yourself, you can also load a prebuilt dataset.")
    load_data_button = st.button("Import data")
    load_data1 = st.empty()
    load_data2 = st.empty()
    load_data3 = st.empty()
with col22:
    st.write("For reseting all drawn numbers click here!")
    reset_button = st.button("Reset data")
    reset_data1 = st.empty()
    reset_data2 = st.empty()
    reset_data3 = st.empty()

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

if "bg" not in st.session_state:
    st.session_state["bg"] = "#000000"


all_numbers = [st.session_state["counter_0"], st.session_state["counter_1"], st.session_state["counter_2"],
               st.session_state["counter_3"], st.session_state["counter_4"], st.session_state["counter_5"],
               st.session_state["counter_6"], st.session_state["counter_7"], st.session_state["counter_8"],
               st.session_state["counter_9"]]

# Tabel of the number of numbers already drawn
d = {'0er': st.session_state["counter_0"], '1er': st.session_state["counter_1"], '2er': st.session_state["counter_2"],
     '3er': st.session_state["counter_3"], '4er': st.session_state["counter_4"], '5er': st.session_state["counter_5"],
     '6er': st.session_state["counter_6"], '7er': st.session_state["counter_7"], '8er': st.session_state["counter_8"],
     '9er': st.session_state["counter_9"], "Gesamt": sum(all_numbers)}


st.subheader("1. Data input")
st.markdown("**Now it's your turn! First generate data for our demonstrator. Than take part in a federated training**")
col3, col4 = st.columns([1, 3])
with col3:
    st.info("Important!!! First draw the number then press save image button and then the trash can icon!")

    number_to_draw = st.session_state["number"]
    st.markdown(f"### Draw the number: {number_to_draw}")

    # Create a canvas component

    #print(st.session_state["bg"])
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=20,
        stroke_color="#ffffff",
        background_color=st.session_state["bg"],
        update_streamlit=True,
        height=300,
        width=300,
        display_toolbar=True,
        key="MNIST_input")

    #print(st.session_state["bg"])
    if st.session_state["bg"] == "#000010":
        #print("hier")
        st.session_state["bg"] = "#000000"
        st.experimental_rerun()

    save_button = st.button("save image")
    st.image("pictures/how_to_draw_192.gif")

with col4:
    st.markdown(
        "This is a representation of the local data on this client in a computer readable format. This data stays on this device and will not be shared with the server.")
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

# reminder for user to start training
if sum(all_numbers) % 10 == 0:
    st.info(
        "Enough numbers drawn? Join the federated training by going to the federated training tab and clicking the join training button")

st.markdown("**Listing already drawn numbers**")
df = pd.DataFrame(data=d, index=["Anzahl"])
st.dataframe(df)


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

        # text ausgabe gui
        load_data1.write("Data was loaded successfully.")
        load_data2.write(np.shape(st.session_state["image"]))
        load_data3.write(np.shape(st.session_state["y_train"]))


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
        time.sleep(2)

        st.experimental_rerun()
    # when drawing first number file does not exists
    else:
        st.error("No data has been saved yet!")

if reset_button:
    st.session_state["image"] = []
    st.session_state["y_train"] = []

    # text ausgabe gui
    reset_data1.write("Data was reseted successfully.")
    reset_data2.write(np.shape(st.session_state["image"]))
    reset_data3.write(np.shape(st.session_state["y_train"]))

    # Update already drawn numbers
    st.session_state["number"] = 1
    st.session_state["counter_0"] = 0
    st.session_state["counter_1"] = 1
    st.session_state["counter_2"] = 0
    st.session_state["counter_3"] = 0
    st.session_state["counter_4"] = 0
    st.session_state["counter_5"] = 0
    st.session_state["counter_6"] = 0
    st.session_state["counter_7"] = 0
    st.session_state["counter_8"] = 0
    st.session_state["counter_9"] = 0
    time.sleep(2)

    st.experimental_rerun()

# when pressing save_button save all steamlit_canvas data into variable
if save_button:
    st.session_state["bg"] = "#000010"
    # Anzeige welche Zahlen gezeichnet werden müssen
    number = np.random.randint(0, high=10, size=None, dtype=int)
    st.session_state["number"] = number
    if canvas_result.image_data is not None:
        image = canvas_result.image_data
        image1 = image.copy()
        image1 = image1.astype('uint8')
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image1 = cv2.resize(image1, (28, 28))
        st.session_state["image"].append(image1)
        st.session_state["y_train"].append(number)

        # save numpy array persistence into .npz file
        np.savez("image_data.npz", x=st.session_state["image"], y=st.session_state["y_train"])

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

    st.experimental_rerun()


# Output of drawn numbers
st.write("Last drawn numbers:")
try:
    col0, col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(10)
    col0.image(st.session_state["image"][-1])
    col1.image(st.session_state["image"][-2])
    col2.image(st.session_state["image"][-3])
    col3.image(st.session_state["image"][-4])
    col4.image(st.session_state["image"][-5])
    col5.image(st.session_state["image"][-6])
    col6.image(st.session_state["image"][-7])
    col7.image(st.session_state["image"][-8])
    col8.image(st.session_state["image"][-9])
    col9.image(st.session_state["image"][-10])
except:
    pass

