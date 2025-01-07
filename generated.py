from sklearn.datasets import load_wine

# Load dataset
wine_data = load_wine()

# Feature datas and labels
X = wine_data.data  # Features (13 chemical features)
y = wine_data.target  # Labels (3 kinds of wine)

# Description of data
# print(wine_data.DESCR)
len(X), len(y)

import pandas as pd

wine = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine['wine_type'] = wine_data.target
wine.head()

wine.shape

# Missing
wine.isna().sum(axis=0)

# Duplication
wine.duplicated().sum()

TRAIN_RATE = 0.6
VAL_RATE = 0.8

idx_train = int(len(wine) * TRAIN_RATE)
idx_val = int(len(wine)* VAL_RATE)
idx_train, idx_val

train = wine.iloc[:idx_train, :]
val = wine.iloc[idx_train:idx_val, :]
test = wine.iloc[idx_val:, :]

train.shape, val.shape, test.shape

X_train = train.drop('wine_type', axis=1)
y_train = train['wine_type']

X_val = val.drop('wine_type', axis=1)
y_val = val['wine_type']

y_train.value_counts(), y_val.value_counts()

u = X_train.mean()
std = X_train.std()

X_train_s = (X_train - u)/std
X_train_s.head()

X_val_s = (X_val - u)/std
X_val_s.head()

ss_dic = {'mean':u, 'std':std}
ss_dic

y_unique = set(y_val.unique()).union(set(y_train.unique()))
label_dict = {specie:i  for i, specie in enumerate(y_unique)}
# label_dict = {specie:i  for i, specie in enumerate(y_val.unique())}

label_dict

y_train_e = y_train.map(label_dict)
y_val_e = y_val.map(label_dict)

y_train_e.shape, y_val_e.shape

u = X_train.mean()
std = X_train.std()

X_train_s = (X_train - u)/std
X_train_s.head()

y_train_e = y_train.map(label_dict)
y_val_e = y_val.map(label_dict)

y_train_e.shape, y_val_e.shape

X_train_s = X_train_s.to_numpy()
y_train_e = y_train_e.to_numpy()

X_val_s = X_val_s.to_numpy()
y_val_e = y_val_e.to_numpy()

print(X_train_s.shape, y_train_e.shape)
print(X_val_s.shape, y_val_e.shape)
print(type(X_train_s), type(y_train_e))
print(type(X_val_s), type(y_val_e))

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_s, y_train_e)

y_pred = clf.predict(X_val_s)
y_pred

from sklearn.metrics import accuracy_score

scores = []
for k in range(3, 30):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train_s, y_train_e)
    y_pred = clf.predict(X_val_s)
    acc = accuracy_score(y_val_e, y_pred)
    scores.append(acc)

import matplotlib.pyplot as plt

plt.plot(scores)

X_test = test.drop('wine_type', axis=1)
y_test = test['wine_type']

X_test_s = (X_test - ss_dic['mean'])/ss_dic['std']
y_test_e = y_test.map(label_dict)

X_test_s = X_test_s.to_numpy()
y_test_e = y_test_e.to_numpy()

y_pred = clf.predict(X_test_s)

(y_test_e == y_pred).sum()/len(y_test_e)

from sklearn.metrics import confusion_matrix

cfm = confusion_matrix(y_test_e, y_pred)
cfm

import seaborn as sns

s = sns.heatmap(cfm, annot=True, cmap='Blues', fmt='d', cbar=False)
s.set(xlabel='Prediction', ylabel='Actual')
plt.show()

from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score

print('accuracy:', accuracy_score(y_test_e, y_pred))
print('recall:', recall_score(y_test_e, y_pred, average='macro'))
print('precision:', precision_score(y_test_e, y_pred, average='macro'))
print('f1 :', f1_score(y_test_e, y_pred, average='macro'))