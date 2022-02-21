import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Predicción - tipo de flor de lirio
""")

st.sidebar.header('Parámetros de entrada del usuario')

def user_input_features():
    sepal_length = st.sidebar.slider('Longitud de los segmentos', 4.8, 8.6, 5.4)
    sepal_width = st.sidebar.slider('Ancho de los segmentos', 2.6, 4.4, 3.4)
    petal_length = st.sidebar.slider('Longitud de los pétalos', 1.2, 6.9, 1.8)
    petal_width = st.sidebar.slider('Ancho de los pétalos', 0.6, 2.5, 0.2)
    data = {'segmento longitud': sepal_length,
            'segmento ancho': sepal_width,
            'petalo longitud': petal_length,
            'petalo ancho': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

dfs = pd.read_csv("IRIS.csv")

st.subheader('Dataset flor de lirio')
st.write(dfs)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Clase de flor y su número de índice')
st.write(iris.target_names)

st.subheader('Predicción - Especie')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Probabilidad de predicción')
st.write(prediction_proba)