"""Result page. Results from Fed training and local Traing displayed on this page. """

import pandas as pd
import streamlit as st
import plotly.express as px

# general page settings
st.set_page_config(
     page_title="Results",
     page_icon="🚀",
     layout="wide",
     initial_sidebar_state="expanded"
    )

with st.sidebar:
    st.subheader("Client ID: 001")

st.title("4. Results")

# set session state variables
if "initial_weights" not in st.session_state:
    st.session_state["initial_weights"] = []

if "fedround_weights" not in st.session_state:
    st.session_state["fedround_weights"] = []

if "result" not in st.session_state:
    st.session_state["result"] = pd.DataFrame()

result = st.session_state["result"]

# show results if available
try:
    col7, col8 = st.columns(2)
    with col7:
        st.write("Comparison of accuracy on training data per FL round.")

        # show results from latest FL round
        st.dataframe(result[["local (fit)", "federated (fit)"]])

        # plot result
        fig_fit = px.line(result[["local (fit)", "federated (fit)"]])
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
        st.dataframe(result[["local (val)", "federated (val)"]])

        # plot result
        fig_val = px.line(result[["local (val)", "federated (val)"]])
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

# graphical representation
if st.session_state["initial_weights"] != []:
    with st.expander("Comparison of weights from beginning to end"):
        col5, col6 = st.columns(2)
        with col5:
            st.write("Representation of the third layer of initial weights received from the server")
            fig = px.imshow(st.session_state["initial_weights"][2])
            st.plotly_chart(fig)
            st.write(st.session_state["initial_weights"])

        with col6:
            st.write("Representation of the third layer of the weights after last round of federated learning")
            fig = px.imshow(st.session_state["perround_weights"][2])
            st.plotly_chart(fig)
            st.write(st.session_state["perround_weights"])
