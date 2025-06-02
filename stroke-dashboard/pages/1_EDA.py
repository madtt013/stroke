import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Exploratory Data Analysis (EDA)")

# Load data
df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
st.write("### Dataset", df.head())

# Info data
st.write("### Info Data")
st.write(df.describe())

# Visualisasi
st.write("### Distribusi Umur")
fig, ax = plt.subplots()
sns.histplot(df['age'], kde=True, ax=ax)
st.pyplot(fig)

st.write("### Korelasi Fitur")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
