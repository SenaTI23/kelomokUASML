import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load data iris dan model langsung di script
@st.cache_data
def load_data():
    return sns.load_dataset('iris')

@st.cache_resource
def train_model():
    iris = load_data()
    X = iris.drop(columns=['species'])
    y = iris['species']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

model = train_model()

st.title("Aplikasi Klasifikasi Iris dengan Random Forest")

st.sidebar.header("Input Data")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.0, 8.0, 5.1)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider('Petal length', 1.0, 7.0, 1.4)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('Data Input')
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Hasil Prediksi')
st.write(f"Jenis Iris: **{prediction[0]}**")

st.subheader('Probabilitas Prediksi')
proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
st.write(proba_df)

# Visualisasi dataset asli
if st.checkbox('Tampilkan Data Iris Asli'):
    iris = load_data()
    st.write(iris)
    st.subheader("Visualisasi Dataset Iris")
    fig = sns.pairplot(iris, hue='species')
    st.pyplot(fig)
