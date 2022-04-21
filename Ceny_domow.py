import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#df_train = pd.read_csv("train.csv")

#brakujace dane
# data1 = df_train.copy(deep = True)
# #print(df_train.info())

# for col in data1:
#     if data1[col].isnull().sum() != 0 :
#         print(f'{col}', data1[col].isnull().sum())
        
# print("-"*10)

# # 
# df_train.describe(include = 'all')


#%% notanik 1 Comprehensive data exploration with Python

df_train = pd.read_csv("train.csv")

df_train['SalePrice'].describe()