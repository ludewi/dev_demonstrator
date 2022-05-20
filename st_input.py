"""Input data and start training"""
from audioop import mul
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import time
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import train_local as tl
import st_result as res
import dataframe_image as dfi

import client1 as cl1

local_hist = []

fed_train_ouput = ""
local_weights = []
global_weights = []
model_vis =[]
model_vis2 =[]

test = 1

para = False
fit = False
eval = False

def app():
    ################################################## data_input ###################################################
    st.subheader("Dateneingabe")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Jetzt bist du gefragt! Generiere Daten für unseren Demonstrator.**")
        st.markdown("Wie in der rechten Abbildung gezeigt!") 
    with col2:
        st.image("pictures/how_to_draw_192.gif")

    # um Platz ziwschen die Elemente zubekommen.
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

    
    
    all_numbers = [st.session_state["counter_0"], st.session_state["counter_1"], st.session_state["counter_2"], st.session_state["counter_3"],
     st.session_state["counter_4"], st.session_state["counter_5"], st.session_state["counter_6"], st.session_state["counter_7"], st.session_state["counter_8"], st.session_state["counter_9"]]
     # Listing of the number of numbers already drawn
    d = {'0er': st.session_state["counter_0"], '1er': st.session_state["counter_1"], '2er': st.session_state["counter_2"], '3er': st.session_state["counter_3"],
     '4er': st.session_state["counter_4"], '5er': st.session_state["counter_5"], '6er': st.session_state["counter_6"], '7er': st.session_state["counter_7"],
      '8er': st.session_state["counter_8"], '9er': st.session_state["counter_9"], "Gesamt": sum(all_numbers)}

    col3, col4 = st.columns([1, 3])
    with col3:
        number_display = st.session_state["number"]
        st.markdown(f"### Zeichne die Nummer: {number_display}")
        
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
        st.markdown("So sieht der Computer die Zahlen. Alle 0-Werte stehen für die Farbe schwarz und alle Werte um die 255 für die Farbe weiß.")
        st.markdown("In dieser Form werden die Zahlen dem neuronalen Netz übergeben!")
        if canvas_result.image_data is not None:
            image_temp = canvas_result.image_data
            image2 = image_temp.copy()
            image2 = image2.astype('uint8')
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            image2 = cv2.resize(image2, (28, 28))
            df_1 = pd.DataFrame(data=image2)
            df_1.style.applymap(lambda x: "background-color: red" if x>150 else "background-color: white")
            #dfi.export(df_1, "pictures/number_as_DT.png")
            #st.write(df_1)
            #st.image("pictures/number_as_DT.png")
            st.dataframe(df_1,  height=700)

        
    
    
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
            # x_train = np.array(image1).reshape(-1, 28, 28, 1)
            images.append(image1)
            #sql.insert_into_table(image1)

            #df_1 = pd.DataFrame(data=image1)
            #df_1.style.applymap(number_background)#style.applymap(lambda x: "background-color: red" if x>240 else "background-color: white")
            

            
            
            
        
    
            # st.session_state["image"] = images
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

    #Skalieren der Daten
    x_train = []
    for image in X:
        x_train.append(image/255.0)

    # reshaping the Data
    x_train = np.array(x_train).reshape(-1, 28, 28, 1)
    y_train = np.array(y_train)

    # shuffle  data
    if len(y_train) > 10: #da fehler meldung wenn noch keine daten erzeugt wurden
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
        train_button=st.button("Am Föderierten Training teilnehmen")
        start_training = st.radio("Möchtest du Am Föderierten Training teilnehmen?",("Ja, teilnehmen", "Nein, nicht teilnehmen"), index=1)
        reset_button = st.button("Daten zurücksetzen")

    if start_training == "Ja, teilnehmen":
        tl.train(x_train, x_test, y_train, y_test)
    
    def start_train():
        x_train_1, x_test_1, y_train_1, y_test_1 = x_train, x_test, y_train, y_test
        cl1.train(x_train_1, x_test_1, y_train_1, y_test_1)
        
    def flag_check():
        return para, fit, eval 

    if train_button:
        
        #import threading 
        #t = threading.Thread(target=cl1.train(x_train, x_test, y_train, y_test))
        import multiprocessing 
        

        p1 = multiprocessing.Process(target=start_train)
        


        
        with st.spinner('Mit Server verbinden...'):
            while True:    
                time.sleep(2)
                
                st.success('Erfolgreich mit Server Verbunden!')
                break

        
        with st.spinner('Server prüft ob benötige Anzahl von Clients mit dem Server verbunden sind, um das Training starten zu können...'):
            while True:
                
                p1.start()
            #cl1.train(x_train, x_test, y_train, y_test)
            

                st.success('Die benötige Anzahl an Clients haben sich mit dem Server verbunden!')
                break
        
        
        #st.write(para_flag) 
        with st.spinner('Aktuelles Modell wird vom Server geladen...'):
            para_flag = ""  
            while True:
                try:
                    with open('parameter.txt') as f:
                        para_flag = f.read()
                except:
                    pass
                #temp = flag_check()
                #para_flag = temp[0]        
                #st.write(model_vis2)
                #st.image(model_vis)
                if para_flag == "parameter":
                    st.success('Aktuelles Modell erfolgreich vom Server geladen!')
                    break
                else:
                    time.sleep(3)
                    st.write(para_flag)
                    
                    
        
        st.image("pictures/ml_model.png", width=300)
        
          
            
        with st.spinner('Training wird gestartet...'):
            fit_flag = ""  
            while True:
                try:
                    with open('fit.txt') as f:
                        fit_flag = f.read()
                except:
                    pass
                
                if fit_flag == "fit":  
                    
                    st.success(f'Training auf den {sum(all_numbers)} erzeugten Zahlen erfolgreich gestartet!')
                    break
        
        
            
        with st.spinner('Es wird trainiert...'):
            while True:
            
                # schreibt in var von st_result
                for x in range(5):    
                    hist_temp = tl.train(x_train, x_test, y_train, y_test)
                    local_hist.append(hist_temp)
                res.local_hist = local_hist
                st.success('Training erfolgreich beendet!')
                break
            
        
        with st.spinner('Aktualisiertes Modell mit angepassten Gewichten an Server senden...'):
            while True:    
                time.sleep(2)
                st.success('Aktualisiertes Modell mit angepassten Gewichten wurde erfolgreich an Server zurück geschickt!')
                break

            

            

            

            with st.expander("Vergleich empfangen Modell vor Training vs. gesnedet Modell mit angepassten Gewichten"):
                st.write("Das sind die Daten die ein Client vom Server erhält (siehe links) und die Daten die der Client zurück an den Server sendet. (siehe rechts)")
                st.write("Wie zu erkennen ist haben diese Zahlen nichts mehr mit den Zahlen zu tun die erzeugt werden, wenn eine Zahl gezeichnet wird.")
                col5, col6 = st.columns(2)
                with col5:
                    st.write("Erhaltende Gewichte vom Server (Modells des gloablen Trainings)")
                    st.write(global_weights)
                with col6:
                    st.write("Angepasste Gewichte des Modells (Training auf lokalen Daten)")
                    st.write(local_weights)

    if reset_button:
        images = []
        st.session_state["number"] = []


