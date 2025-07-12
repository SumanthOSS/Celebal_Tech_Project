#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


try:
    df = pd.read_csv('titanic.csv')
except FileNotFoundError:
    # Fallback for local execution; replace with actual path if needed
    df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')


# In[3]:


# 1. Overview of the dataset
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())


# In[4]:


# 2. Check for missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])


# In[5]:


# Handling missing values
# Fill 'Age' with median
df['Age'].fillna(df['Age'].median(), inplace=True)
# Fill 'Embarked' with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# Drop 'Cabin' due to high missing values
df.drop(columns=['Cabin'], inplace=True)


# In[6]:


# 3. Statistical summary
print("\nStatistical Summary:")
print(df.describe(include='all'))


# In[7]:


# 4. Data Distributions - Numerical Features
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()


# In[8]:


# 5. Detect Outliers - Box Plots for Numerical Features
plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()


# In[9]:


# 6. Categorical Features Analysis
categorical_cols = ['Survived', 'Pclass', 'Sex', 'Embarked']
plt.figure(figsize=(12, 8))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=col, data=df)
    plt.title(f'Count Plot of {col}')
plt.tight_layout()
plt.show()


# In[10]:


# 7. Relationships Between Variables - Correlation Heatmap
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()


# In[11]:


# 8. Survival Analysis by Categorical Features
plt.figure(figsize=(12, 4))
# Survival by Sex
plt.subplot(1, 3, 1)
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Sex')
# Survival by Pclass
plt.subplot(1, 3, 2)
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Pclass')
# Survival by Embarked
plt.subplot(1, 3, 3)
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title('Survival by Embarked')
plt.tight_layout()
plt.show()


# In[12]:


# 9. Age Distribution by Survival
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[13]:


# 10. Fare Distribution by Pclass
plt.figure(figsize=(8, 6))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare Distribution by Pclass')
plt.show()


# In[14]:


# 11. Additional Insights - Pairplot for Numerical Features
sns.pairplot(df[numerical_cols + ['Survived']], hue='Survived')
plt.show()


# In[15]:


print("\nEDA Summary:")
print("- Missing values handled: 'Age' filled with median, 'Embarked' with mode, 'Cabin' dropped.")
print("- Numerical features analyzed for distributions and outliers.")
print("- Categorical features show survival patterns (e.g., females and higher-class passengers had higher survival rates).")
print("- Correlations indicate relationships like Pclass and Fare, and Pclass and Survived.")
print("- Visualizations highlight key patterns in survival, age, fare, and class.")


# In[ ]:




