#%%
from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time
from matplotlib.colors import ListedColormap
begin = time.time()
dataset = pd.read_csv('starbacs_food.csv')
X = dataset.iloc[:, 2:13].values
y = dataset.iloc[:, 14].values
X_train,X_test,y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=0)
print(X_train.shape)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
plt.close()
time.sleep(1)
end = time.time()
print(f"Total runtime of the program is {end - begin}")




# %%
