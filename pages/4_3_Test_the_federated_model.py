
import numpy as np
import pandas as pd
import cv2
from tensorflow.python.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import plotly.express as px
st.subheader("Classification")
st.write("Now we check if our federated model can also make a good prediction.")


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

    # load_model for classification
    model = load_model('fit_global_model')

    if st.button('Predict'):
        x_test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_test = x_test/255
        val = model.predict(x_test.reshape(-1, 28, 28 ,1))
        st.write(f'result: {np.argmax(val[0])}')
        st.bar_chart(val[0])
