"""Result page V1. Results from Fed training and local Traing displayed on this page
    and classifiction """
import numpy as np
import pandas as pd
import cv2
from tensorflow.python.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from pages.p2_demonstrator_V3 import local_val_score, local_train_acc, fed_train_acc, fed_val_score, local_train_log, fed_train_log

st.set_page_config(
     page_title="Results",
     page_icon="🚀",
     layout="wide",
     initial_sidebar_state="expanded"
    )

st.title("Ergebnisseite")
st.sidebar.markdown("# Ergebnisseite")

if len(local_val_score) != 0:
    df_val_temp = pd.DataFrame(data=local_val_score, columns=["Local Accuracy per Round (eval)"])
    df_fit_temp = pd.DataFrame(data=local_train_acc, columns=["Local Accuracy per Round (fit)"])

if len(fed_train_acc) != 0:
    df_fed_fit_temp = pd.DataFrame(data=fed_train_acc, columns=["Federated Accuracy per Round (fit)"]) #columns=["Ep1", "Ep2", "Ep3", "Ep4", "Ep5", "Ep6", "Ep7", "Ep8", "Ep9", "Ep10"]
    df_fed_val_temp = pd.DataFrame(data=fed_val_score, columns=["Federated Accuracy per Round (eval)"])

    result = pd.concat([df_val_temp, df_fit_temp, df_fed_fit_temp, df_fed_val_temp], axis=1)
    st.dataframe(result)
    st.line_chart(result)

else:
    st.write("Es wurde noch kein Training durchgeführt!")

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

with st.expander("Hier findest du die Log Daten des lokalen Trainings"):
        st.write(local_train_log)

with st.expander("Hier findest du die Log Daten des Föderierten Trainings"):
        st.write(fed_train_log)
