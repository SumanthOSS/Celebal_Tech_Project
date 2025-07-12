#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[3]:


# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[4]:


#Define and Train Multiple Models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
    print()

for name, model in models.items():
    evaluate_model(name, model)


# In[5]:


#Hyperparameter Tuning with GridSearchCV
param_grid_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=3, scoring='f1_macro')
grid_search.fit(X_train, y_train)

print("Best Parameters (GridSearchCV):", grid_search.best_params_)

y_pred_grid = grid_search.predict(X_test)
print("F1 Score after GridSearchCV:", f1_score(y_test, y_pred_grid, average='macro'))


# In[7]:


#Hyperparameter Tuning with RandomizedSearchCV
from scipy.stats import randint

param_dist_rf = {
    'n_estimators': randint(10, 200),
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(RandomForestClassifier(), param_dist_rf, n_iter=10, cv=3, scoring='f1_macro', random_state=42)
random_search.fit(X_train, y_train)

print("Best Parameters (RandomizedSearchCV):", random_search.best_params_)

y_pred_rand = random_search.predict(X_test)
print("F1 Score after RandomizedSearchCV:", f1_score(y_test, y_pred_rand, average='macro'))


# In[9]:


#Compare and Select the Best Model
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1': []
}


# In[10]:


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results['Model'].append(name)
    results['Accuracy'].append(accuracy_score(y_test, y_pred))
    results['Precision'].append(precision_score(y_test, y_pred, average='macro'))
    results['Recall'].append(recall_score(y_test, y_pred, average='macro'))
    results['F1'].append(f1_score(y_test, y_pred, average='macro'))


# In[11]:


df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='F1', ascending=False)
print(df_results)


# In[ ]:




