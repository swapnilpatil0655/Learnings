import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_name = load_data()

# Train model
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

# Sidebar inputs
st.sidebar.title("Iris Classification")

sepal_length = st.sidebar.slider(
    "Sepal length",
    float(df['sepal length (cm)'].min()),
    float(df['sepal length (cm)'].max()),
    float(df['sepal length (cm)'].mean())
)

sepal_width = st.sidebar.slider(
    "Sepal width",
    float(df['sepal width (cm)'].min()),
    float(df['sepal width (cm)'].max()),
    float(df['sepal width (cm)'].mean())
)

petal_length = st.sidebar.slider(
    "Petal length",
    float(df['petal length (cm)'].min()),
    float(df['petal length (cm)'].max()),
    float(df['petal length (cm)'].mean())
)

petal_width = st.sidebar.slider(
    "Petal width",
    float(df['petal width (cm)'].min()),
    float(df['petal width (cm)'].max()),
    float(df['petal width (cm)'].mean())
)

# Prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

prediction = model.predict(input_data)
prediction_species = target_name[prediction[0]]

st.write("Predictiion:")
st.write("Your Prediction:", prediction_species)