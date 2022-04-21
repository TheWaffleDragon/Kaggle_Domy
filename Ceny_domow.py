import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv("train.csv")

# #%% wtępna analiza 

# df_train['SalePrice'].describe()
# sns.distplot(df_train['SalePrice'])

# #skewness and kurtosis
# print("Skewness: %f" % df_train['SalePrice'].skew())
# print("Kurtosis: %f" % df_train['SalePrice'].kurt())

# #scatter plot grlivarea/saleprice
# var = 'GrLivArea'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# #scatter plot totalbsmtsf/saleprice
# var = 'TotalBsmtSF'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# #box plot overallqual/saleprice
# var = 'OverallQual'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000)

# var = 'YearBuilt'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# f, ax = plt.subplots(figsize=(30, 8))
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000);
# plt.xticks(rotation=90)

#%% korelacje
#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix
k = 11 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show()

#%%
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)*100
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

'''
              Total    Percent
PoolQC         1453  99.520548 --> drop - covered by PoolArea
MiscFeature    1406  96.301370 --> drop - no strong corr
Alley          1369  93.767123 --> drop - no strong corr
Fence          1179  80.753425 --> drop - no strong corr
FireplaceQu     690  47.260274 --> drop - no strong corr
LotFrontage     259  17.739726 --> drop - no strong corr


GarageType       81   5.547945 -->> missing values = houses without a garage --> drop
GarageYrBlt      81   5.547945
GarageQual       81   5.547945
GarageCond       81   5.547945
GarageFinish     81   5.547945

BsmtFinType2     38   2.602740 -->>no basement --> drop
BsmtExposure     38   2.602740
BsmtCond         37   2.534247
BsmtFinType1     37   2.534247
BsmtQual         37   2.534247

MasVnrArea        8   0.547945 -->> drop - no strong corr
MasVnrType        8   0.547945


Electrical        1   0.068493 --> drop record

df_train_a = df_train
df_train_a['Alley'][df_train_a['Alley']=='Pave'] = 1
df_train_a['Alley'][df_train_a['Alley']=='Grvl'] = 2
df_train_a['Alley'] = df_train_a['Alley'].fillna(0)
var = 'Alley'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

'''

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
print(df_train.isnull().sum().max())

#%%


