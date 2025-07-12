#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


plt.style.use('seaborn')


# In[3]:


data = pd.read_csv('/Users/sumanthoruganti/Desktop/MIT MANIPAL ACADEMICS/datasets1/titanic.csv')


# In[4]:


# --- Preprocessing ---
# Check for missing values
print("Missing values before preprocessing:")
print(data.isnull().sum())

# Handle missing values
# Fill missing Age with median age
data['Age'] = data['Age'].fillna(data['Age'].median())

# Drop Cabin column (too many missing values, not needed for our visualizations)
data = data.drop(columns=['Cabin'])

# Drop rows with missing Embarked (only a few)
data = data.dropna(subset=['Embarked'])

# Convert Pclass to categorical for clarity
data['Pclass'] = data['Pclass'].astype('category')

# Verify preprocessing
print("\nMissing values after preprocessing:")
print(data.isnull().sum())


# In[5]:


# --- Visualization 1: Survival Rates by Passenger Class (Bar Chart) ---
# Calculate survival rate for each passenger class
survival_by_class = data.groupby('Pclass')['Survived'].mean() * 100  # Convert to percentage
classes = ['1st Class', '2nd Class', '3rd Class']
survival_rates = survival_by_class.values

# Create the bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(classes, survival_rates, color=['#4CAF50', '#2196F3', '#F44336'], 
               edgecolor=['#388E3C', '#1976D2', '#D32F2F'])
plt.title('Survival Rates by Passenger Class', fontsize=14, pad=15)
plt.xlabel('Passenger Class', fontsize=12)
plt.ylabel('Survival Rate (%)', fontsize=12)
plt.ylim(0, 100)  # Set y-axis limit to 0-100%

# Add percentage labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 2, f'{height:.1f}%', 
             ha='center', fontsize=10)

plt.tight_layout()
plt.show()


# In[7]:


# Age Distribution of Survivors vs. Non-Survivors (Histogram) ---
# Create histograms for survivors and non-survivors
plt.figure(figsize=(10, 6))
plt.hist(data[data['Survived'] == 1]['Age'], bins=20, alpha=0.5, color='#4CAF50', label='Survived')
plt.hist(data[data['Survived'] == 0]['Age'], bins=20, alpha=0.5, color='#F44336', label='Did Not Survive')
plt.title('Age Distribution of Survivors vs. Non-Survivors', fontsize=14, pad=15)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()


# In[11]:


#Visualization 2: Gender Breakdown of Survivors (Pie Chart) ---
# Filter for survivors only and count by gender
survivors = data[data['Survived'] == 1]
gender_counts = survivors['Sex'].value_counts()
labels = ['Female', 'Male']
sizes = [gender_counts.get('female', 0), gender_counts.get('male', 0)]
percentages = [x / sum(sizes) * 100 for x in sizes]

# Create the pie chart
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=['#FF6384', '#36A2EB'], autopct='%1.1f%%', 
       startangle=90, wedgeprops={'edgecolor': '#000000'})  # Single black edge color
plt.title('Gender Distribution of Titanic Survivors', fontsize=14, pad=15)
plt.axis('equal')  # Makes the pie chart circular
plt.tight_layout()
plt.show()


# In[ ]:




