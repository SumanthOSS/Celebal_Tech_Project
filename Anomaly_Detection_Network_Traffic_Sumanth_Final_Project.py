#!/usr/bin/env python
# coding: utf-8

# # What the Project is About
# 
# ### This project focuses on Anomaly Detection in Network Traffic using unsupervised learning methods.
# 
# ### We're working with network connection records from the KDD Cup 1999 dataset.
# 
# ### The goal is to detect anomalies (attacks) in network traffic without using labels to train the model.

# ## Step 1: 📦 Import Libraries

# In[26]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
from sklearn.model_selection import train_test_split


# ## Step 2: 📥 Load Dataset

# In[27]:


import os


folder_path = '/Users/sumanthoruganti/Desktop/archive (4)'

# List all files to see the exact filename
files = os.listdir(folder_path)
for f in files:
    print(f)


# In[30]:


import os

folder_path = '/Users/sumanthoruganti/Desktop/archive (4)'
files = os.listdir(folder_path)

for f in files:
    if "corrected" in f.lower():  # safer match
        print("Found:", repr(f))


# In[31]:


import pandas as pd

file_path = '/Users/sumanthoruganti/Desktop/archive (4)/kddcup.data_10_percent_corrected'

# Try loading with default encoding
try:
    df = pd.read_csv(file_path, header=None)
except:
    # If encoding error occurs, fall back to latin1
    df = pd.read_csv(file_path, header=None, encoding='latin1')

print("✅ Dataset loaded successfully!")
df.head()


# ## Step 3: Assign Proper Column Names

# In[32]:


column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

df.columns = column_names
df.head()


# ## Step 4: Encode Categorical Features and Scale the Data

# In[33]:


from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label'].copy()

# Convert 'normal' to 0 and others to 1 (anomalies)
y_binary = y.apply(lambda x: 0 if x == 'normal.' else 1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ## Step 5: Isolation Forest

# In[34]:


from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Train model
iso_forest = IsolationForest(contamination=0.1, random_state=42)
y_pred_if = iso_forest.fit_predict(X_scaled)

# Convert output: 1 → 0 (normal), -1 → 1 (anomaly)
y_pred_if = [0 if x == 1 else 1 for x in y_pred_if]

# Evaluation
print("🔍 Isolation Forest Results:\n")
print(classification_report(y_binary, y_pred_if, target_names=["Normal", "Anomaly"]))

# Confusion matrix
cm_if = confusion_matrix(y_binary, y_pred_if)
print("Confusion Matrix:\n", cm_if)


# ## Step 6: Visualize Results (Fit Graph)

# In[35]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.heatmap(cm_if, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Pred: Normal", "Pred: Anomaly"],
            yticklabels=["True: Normal", "True: Anomaly"])
plt.title("Isolation Forest Confusion Matrix")
plt.show()


# ## Step 7: Autoencoder (Deep Learning)

# In[36]:


from sklearn.model_selection import train_test_split

# Train only on normal data (label 0)
X_train = X_scaled[y_binary == 0]

# Split into train-validation sets
X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)


# In[37]:


from keras.models import Model
from keras.layers import Input, Dense

input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')


# ## Step 8: Train Autoencoder

# In[39]:


history = autoencoder.fit(X_train, X_train,
                          epochs=50,
                          batch_size=64,
                          shuffle=True,
                          validation_data=(X_val, X_val))


# ## Step 9: Plot Training & Validation Loss

# In[40]:


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# ## Step 10: Detect Anomalies using Autoencoder

# In[41]:


# Predict reconstruction loss on entire data
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)

# Set threshold (based on training normal data reconstruction error)
threshold = np.percentile(mse[y_binary == 0], 95)
print("Reconstruction error threshold:", threshold)

# Predict anomalies
y_pred_ae = [1 if e > threshold else 0 for e in mse]


# ## Step 11: Evaluate Autoencoder

# In[42]:


print("🔍 Autoencoder Results:\n")
print(classification_report(y_binary, y_pred_ae, target_names=["Normal", "Anomaly"]))

# Confusion Matrix
cm_ae = confusion_matrix(y_binary, y_pred_ae)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_ae, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Pred: Normal", "Pred: Anomaly"],
            yticklabels=["True: Normal", "True: Anomaly"])
plt.title("Autoencoder Confusion Matrix")
plt.show()


# In[ ]:




