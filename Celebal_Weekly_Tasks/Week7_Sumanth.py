#!/usr/bin/env python
# coding: utf-8

# In[12]:


# train_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)


# In[13]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌸 Iris Flower Classifier")
st.write("Enter flower measurements to predict the species.")

# User Inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
predicted_class = model.predict(input_data)[0]
predicted_proba = model.predict_proba(input_data)

# Display Prediction
target_names = load_iris().target_names
st.subheader("Prediction")
st.write(f"**Predicted Species:** {target_names[predicted_class]}")

# Probability Chart
st.subheader(" Prediction Probabilities")
proba_df = pd.DataFrame(predicted_proba, columns=target_names)

st.bar_chart(proba_df.T)

# Visualize Feature Importance
st.subheader(" Feature Importance")
feature_importance = model.feature_importances_
features = load_iris().feature_names

fig, ax = plt.subplots()
sns.barplot(x=feature_importance, y=features, ax=ax)
st.pyplot(fig)


# In[ ]:




