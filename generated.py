import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42

# !wget https://raw.githubusercontent.com/devdio/flyai_datasets/main/citrus.csv

citrus_link = 'https://raw.githubusercontent.com/devdio/flyai_datasets/main/citrus.csv'
citrus = pd.read_csv(citrus_link)
# citrus = pd.read_csv('citrus.csv')
citrus.shape

citrus.head()

df = citrus.copy()
df.info()

df.describe().T

# Count the number of missing (NaN) values in each column of the DataFrame.
df.isna().sum(axis=0)

# Count the total number of duplicate rows in the DataFrame.
df.duplicated().sum()

# Shuffle the rows of the DataFrame randomly,
# using the specified seed for reproducibility,
# and display the first 5 rows of the shuffled DataFrame."
df = df.sample(frac=1, random_state=SEED)
df.head()

idx_train = int(len(df) * 0.6)
idx_val = int(len(df)* 0.8)

idx_train, idx_val

# Split the DataFrame into train, validation, and test sets and display their shapes.
train = df.iloc[:idx_train, :]
val = df.iloc[idx_train:idx_val, :]
test = df.iloc[idx_val:, :]

train.shape, val.shape, test.shape

X_train = train.drop('name', axis=1)
y_train = train['name']

X_val = val.drop('name', axis=1)
y_val = val['name']

y_train.value_counts(), y_val.value_counts()

u = X_train.mean()
std = X_train.std()

u, std

X_train_s = (X_train - u)/std
X_train_s.head()

X_val_s = (X_val - u)/std
X_val_s.head()

ss_dic = {'mean':u, 'std':std}
ss_dic

label_dict = {'grapefruit':0, 'orange':1}

y_train_e = y_train.map(label_dict)
y_val_e = y_val.map(label_dict)

y_train_e, y_val_e

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

y_val_e

(y_pred == y_val_e).sum()/len(y_val_e)

from sklearn.metrics import accuracy_score

scores = []
for k in range(3, 30):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train_s, y_train_e)
    y_pred = clf.predict(X_val_s)
    acc = accuracy_score(y_val_e, y_pred)
    scores.append(acc)

plt.plot(scores)

test.head()

X_test = test.drop('name', axis=1)
y_test = test['name']

X_test_s = (X_test - ss_dic['mean'])/ss_dic['std']
y_test_e = y_test.map(label_dict)

X_test_s = X_test_s.to_numpy()
y_test_e = y_test_e.to_numpy()

y_pred = clf.predict(X_test_s)

(y_test_e == y_pred).sum()/len(y_test_e)

from sklearn.metrics import confusion_matrix

cfm = confusion_matrix(y_test_e, y_pred)
cfm

s = sns.heatmap(cfm, annot=True, cmap='Blues', fmt='d', cbar=False)
s.set(xlabel='Prediction', ylabel='Actual')
plt.show()

from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score

print('accuracy:', accuracy_score(y_test_e, y_pred))
print('recall:', recall_score(y_test_e, y_pred))
print('precision:', precision_score(y_test_e, y_pred))
print('f1 :', f1_score(y_test_e, y_pred))
