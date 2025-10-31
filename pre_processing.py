# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression as mutual_info
from sklearn.feature_selection import SequentialFeatureSelector

import warnings
warnings.filterwarnings('ignore')

# Load the data
X_train = pd.read_csv('data/data_labeled/X_train.csv')
X_test = pd.read_csv('data/data_labeled/X_test.csv')

# y CSV files don't have headers, so we need to specify header=None and provide column names
y_train = pd.read_csv('data/data_labeled/y_train.csv', header=None, names=['heart_failure_risk'])
y_test = pd.read_csv('data/data_labeled/y_test.csv', header=None, names=['heart_failure_risk'])

# Explore the data

# print(X_train.info())
"""
RangeIndex: 1000 entries, 0 to 999
Data columns (total 14 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   age                1000 non-null   int64  
 1   blood pressure     1000 non-null   float64
 2   calcium            1000 non-null   float64
 3   cholesterol        1000 non-null   float64
 4   hemoglobin         1000 non-null   float64
 5   height             1000 non-null   float64
 6   potassium          1000 non-null   float64
 7   profession         1000 non-null   object 
 8   sarsaparilla       1000 non-null   object 
 9   smurfberry liquor  1000 non-null   object 
 10  smurfin donuts     1000 non-null   object 
 11  vitamin D          1000 non-null   float64
 12  weight             1000 non-null   float64
 13  img_filename       1000 non-null   object 
dtypes: float64(8), int64(1), object(5)
memory usage: 109.5+ KB
"""
# y_train.info()
"""
RangeIndex: 999 entries, 0 to 998
Data columns (total 1 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   0.06    999 non-null    float64
dtypes: float64(1)
memory usage: 7.9 KB
"""
# X_test.info()
# y_test.info()

# delete missing values
X_train = X_train.dropna()

# print length of X_train
#print(len(X_train))
"""
1000
""" # There doesn't seem to be any missing values in the data

# delete missing values in y_train
y_train = y_train.dropna()

# print length of y_train
# print(len(y_train))
"""
1000
""" # There doesn't seem to be any missing values in the data

# We can drop the img_filename column
X_train = X_train.drop(columns=['img_filename'])
X_test = X_test.drop(columns=['img_filename'])

# print all the possible values for the object columns
# print(X_train['profession'].unique())
"""
['food production' 'services' 'manufacturing' 'craftsmanship'
 'administration and governance' 'resource extraction']
"""
print(X_train['sarsaparilla'].unique())
"""
['High' 'Moderate' 'Very low' 'Low' 'Very high']
"""
print(X_train['smurfberry liquor'].unique())
"""
['High' 'Moderate' 'Very high' 'Low' 'Very low']
"""
print(X_train['smurfin donuts'].unique())
"""
['Very high' 'Very low' 'High' 'Low' 'Moderate']
"""

# Transform profession column into a single column for each different profession with binary encoding
X_train = pd.get_dummies(X_train, columns=['profession'])
X_test = pd.get_dummies(X_test, columns=['profession'])

# print the shape of X_train and X_test
# print(X_train.info())
# print(X_test.info())

# Map the columns sarsaparilla, smurfberry liquor, and smurfin donuts to values 0, 1, 2, 3, 4
X_train['sarsaparilla'] = X_train['sarsaparilla'].map({'Very high': 4, 'High': 3, 'Moderate': 2, 'Low': 1, 'Very low': 0})
X_test['sarsaparilla'] = X_test['sarsaparilla'].map({'Very high': 4, 'High': 3, 'Moderate': 2, 'Low': 1, 'Very low': 0})
X_train['smurfberry liquor'] = X_train['smurfberry liquor'].map({'Very high': 4, 'High': 3, 'Moderate': 2, 'Low': 1, 'Very low': 0})
X_test['smurfberry liquor'] = X_test['smurfberry liquor'].map({'Very high': 4, 'High': 3, 'Moderate': 2, 'Low': 1, 'Very low': 0})
X_train['smurfin donuts'] = X_train['smurfin donuts'].map({'Very high': 4, 'High': 3, 'Moderate': 2, 'Low': 1, 'Very low': 0})
X_test['smurfin donuts'] = X_test['smurfin donuts'].map({'Very high': 4, 'High': 3, 'Moderate': 2, 'Low': 1, 'Very low': 0})

# print the shape of X_train and X_test
# print(X_train.shape)
# print(X_test.shape)
