import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

st.title("Model Training")

# Load data
df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
df = df.dropna()
df = pd.get_dummies(df.drop(columns=["id"]), drop_first=True)

X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model/stroke_model.pkl")
