#!/usr/bin/env python
# coding: utf-8

# ##  1. Load Data

# In[32]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[33]:


train  = pd.read_csv("/Users/sumanthoruganti/Desktop/house-prices-advanced-regression-techniques/train.csv")
test   = pd.read_csv("/Users/sumanthoruganti/Desktop/house-prices-advanced-regression-techniques/test.csv")
test_ids = test['Id']
y = np.log1p(train['SalePrice'])
X = train.drop(['SalePrice', 'Id'], axis=1)
test = test.drop('Id', axis=1)
all_data = pd.concat([X, test], axis=0, ignore_index=True)


# In[34]:


# Missing values
cols_fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
cols_fill_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
cols_fill_mode = ['MasVnrType', 'Electrical', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']
for col in cols_fill_none:
    all_data[col] = all_data[col].fillna('None')
for col in cols_fill_zero:
    all_data[col] = all_data[col].fillna(0)
for col in cols_fill_mode:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# In[35]:


# Feature engineering
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['HouseAge'] = all_data['HouseAge'].clip(lower=0)
all_data['RemodAge'] = all_data['RemodAge'].clip(lower=0)
all_data['HasPool'] = all_data['PoolQC'].apply(lambda x: 1 if x != 'None' else 0)


# In[36]:


# Encoding
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
for col in ordinal_cols:
    all_data[col] = all_data[col].map(quality_map)
all_data = pd.get_dummies(all_data)


# In[37]:


# Scaling
scaler = StandardScaler()
numerical_cols = all_data.select_dtypes(include=['int64', 'float64']).columns
all_data[numerical_cols] = scaler.fit_transform(all_data[numerical_cols])


# In[38]:


# Split
X = all_data.iloc[:len(train), :]
test = all_data.iloc[len(train):, :]


# In[39]:


# Train model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print('RMSE:', np.sqrt(mean_squared_error(y_val, y_pred)))


# In[40]:


# Predict
predictions = np.expm1(model.predict(test))
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})
submission.to_csv('submission.csv', index=False)


# In[ ]:




