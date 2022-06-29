"""Result page V1. Results from Fed training and local Traing displayed on this page
    and classifiction """
import numpy as np
import pandas as pd
import cv2
from tensorflow.python.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import plotly.express as px

st.set_page_config(
     page_title="Results",
     page_icon="🚀",
     layout="wide",
     initial_sidebar_state="expanded"
    )

if "result" not in st.session_state:
    st.session_state["result"] = pd.DataFrame()

if "local_log" not in st.session_state:
    st.session_state["local_log"] = ""

if "fed_log" not in st.session_state:
    st.session_state["fed_log"] = ""

st.title("Results")
result = st.session_state["result"]


try:
    col7, col8 = st.columns(2)
    with col7:
        st.write("Comparison of accuracy on training data per FL round.")

        # show results from current FL Durchgang
        st.dataframe(result[["Lokal (fit)", "Föderiert (fit)"]])

        # plot result
        fig_fit = px.line(result[["Lokal (fit)", "Föderiert (fit)"]])
        fig_fit.update_layout(
            title="Accuracy per round",
            xaxis_title="FL-rounds",
            yaxis_title="Accuracy",
            legend_title="Models")

        fig_fit.update_yaxes(range=(0.0, 1.0))
        st.plotly_chart(fig_fit)

    with col8:
        st.write("Comparison of accuracy on validation data per FL round.")

        # show results from current FL Durchgang
        st.dataframe(result[["Lokal (val)", "Föderiert (val)"]])

        # plot result
        fig_val = px.line(result[["Lokal (val)", "Föderiert (val)"]])
        fig_val.update_layout(
            title="Accuracy per round",
            xaxis_title="FL-rounds",
            yaxis_title="Accuracy",
            legend_title="Models")
        fig_val.update_yaxes(range=(0.0, 1.0))
        st.plotly_chart(fig_val)
except:
    st.error("No training has been performed yet!")
    pass


st.subheader("Classification")
st.write("Now we check if our federated model can also make a good prediction.")

col3, col4 = st.columns(2)

with col3:
    SIZE = 192
    # Create a canvas component
    canvas_result_1 = st_canvas(
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
    if canvas_result_1.image_data is not None:
        img = cv2.resize(canvas_result_1.image_data.astype('uint8'), (28, 28))
        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write('Model Input')
        st.image(rescaled)
    
    #load_model for classification
    model = load_model('fit_global_model')

    if st.button('Predict'):
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val = model.predict(test_x.reshape(-1, 28, 28,1))
        st.write(f'result: {np.argmax(val[0])}')
        st.bar_chart(val[0])

st.subheader("Log-Files")
with st.expander("Here you can find the log data of the local training"):
        st.write(st.session_state["local_log"])

with st.expander("Here you can find the log data of the federated training"):
        st.write(st.session_state["fed_log"])
