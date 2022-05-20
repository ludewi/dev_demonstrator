import numpy as np
import pandas as pd
import cv2
from tensorflow.python.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas


local_hist = []
fed_hist = []
local_train_ouput = ""
fed_train_ouput = ""

def app():
    st.subheader("Ergebnisseite")
    
    
    if len(local_hist) != 0:
       
        df1_temp = pd.DataFrame(data=local_hist, columns=["Local Accuracy per Round"])
        #st.dataframe(df1_temp)
       
            
    

    if len(fed_hist) != 0:
        df2_temp = pd.DataFrame(data=fed_hist,columns=["Federated Accuracy per Round"]) #columns=["Ep1", "Ep2", "Ep3", "Ep4", "Ep5", "Ep6", "Ep7", "Ep8", "Ep9", "Ep10"]
        #st.dataframe(df2_temp)
        #st.line_chart(fed_hist)
        
        
        result = pd.concat([df1_temp, df2_temp], axis=1)
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
            st.write(local_train_ouput)
    
    with st.expander("Hier findest du die Log Daten des Föderierten Trainings"):
            st.write(fed_train_ouput)