"""Result page V1. Results from Fed training and local Traing displayed on this page
    and classifiction """
import numpy as np
import pandas as pd
import cv2
from tensorflow.python.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from pages.p2_demonstrator_V3 import local_train_log, fed_train_log

st.set_page_config(
     page_title="Results",
     page_icon="🚀",
     layout="wide",
     initial_sidebar_state="expanded"
    )

st.title("Ergebnisseite")
st.sidebar.markdown("# Ergebnisseite")

if st.session_state["result"] is not st.session_state["result"].empty:
    st.dataframe(st.session_state["result"])
    st.line_chart(st.session_state["result"])

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
