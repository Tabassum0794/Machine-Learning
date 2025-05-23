#%%
# from tkinter import Y
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
import timeit 
from sklearn import metrics
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
begin = time.time()
df = pd.read_csv("starbacs_food.csv")
df.head()
plt.scatter(df["Calories"],df["Sugars(g)"],df["Total Carbohydrate(g)"])
plt.show()
km=KMeans(n_clusters=2,random_state=1)
y_predicted = km.fit_predict(df[['Calories','Sugars(g)','Total Carbohydrate(g)']])
y_predicted
df['cluster'] = y_predicted
df.head()
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
# df3 = df[df.cluster==2]
# df4 = df[df.cluster==3]
plt.scatter(df1.Calories,df1['Sugars(g)'],df1['Total Carbohydrate(g)'],color='green')
plt.scatter(df2.Calories,df2['Sugars(g)'],df2["Total Carbohydrate(g)"],color='red')
# plt.scatter(df3.Calories,df3['Sugars(g)'],color='black')
# plt.scatter(df4.Calories,df4['Sugars(g)'],color='yellow')
plt.xlabel('Calories')
plt.ylabel('Sugars(g)')
plt.show()
# plt.legend()
print(df[['Name','Calories', 'Sugars(g)','cluster']])
score = silhouette_score(df[['Calories','Sugars(g)','Total Carbohydrate(g)']], km.labels_, metric='euclidean')
print(score)
time.sleep(1)
end = time.time()
print(f"Total runtime of the program is {end - begin}")
 





# %%

# %%

# %%
