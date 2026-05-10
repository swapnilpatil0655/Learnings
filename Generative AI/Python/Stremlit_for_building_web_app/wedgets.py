import streamlit as st
import pandas as pd

st.title("Streamlit Widgets")

name = st.text_input("Enter your name:")




if name:
    st.write(f"Hello,{name}")


# Create a slider for age selection
age = st.slider("Select your age:", 0, 100, 25)
st.write(f"Your age is: {age}")


# Create a selectbox for fevorite language selection
options = ["Python", "JavaScript", "Java", "C++"]
choice = st.selectbox("Select your favorite programming language:", options)
st.write(f"Your favorite programming language is: {choice}")


# Dataframe for multiselect
data = {
    "name": ["Alice", "Bob", "Charlie", "David"],
    "age": [25, 30, 35, 40],
    "city": ["New York", "Los Angeles", "Chicago", "Houston"]

}

df = pd.DataFrame(data)
df.to_csv("data.csv")

st.write(df)


## upload a file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)