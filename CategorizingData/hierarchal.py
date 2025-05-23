#%%
from cProfile import label
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import time
begin = time.time()
df = pd.read_csv("starbacs_food.csv")
df1 = df.loc[:,["Cholesterol(mg)","Sodium(mg)","Total Fat(g)"]]
print(df.head())
plt.scatter(df1['Cholesterol(mg)'],df1['Sodium(mg)'],df1['Total Fat(g)'])
plt.figure(figsize=(10,7))
dendrogram =sch.dendrogram(sch.linkage(df1,method="ward"))
plt.title('dendogram')
plt.xlabel('food item')
plt.ylabel('euclidean distance')
plt.show()
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage='ward')
cluster.fit_predict(df1)
c1= cluster.fit_predict(df1)
sil=silhouette_score(df1,c1)
x=df1.values
plt.figure(figsize=(10,7))
plt.scatter(x[c1==0,0],x[c1==0,1], s=100, c='red', label='cluster1')
plt.scatter(x[c1==1,0],x[c1==1,1], s=100, c='blue', label='cluster2')
plt.scatter(x[c1==2,0],x[c1==2,1], s=100, c='green', label='cluster 3')
plt.xlabel('Cholesterol(mg)')
plt.ylabel('Sodium(mg)')
plt.show()
print(sil)
time.sleep(1)
end = time.time()
print(f"Total runtime of the program is {end - begin}")

 # %%
