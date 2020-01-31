#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=d7NJGLevmwA


"""
@author: Sreenivas Bhattiprolu
"""

import pandas as pd
from matplotlib import pyplot as plt

df=pd.read_excel('other_files/K_Means.xlsx')
print(df.head())

import seaborn as sns
sns.regplot(x=df['X'], y=df['Y'], fit_reg=False)


from sklearn.cluster import KMeans

#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

model = kmeans.fit(df)

predicted_values = kmeans.predict(df)


plt.scatter(df['X'], df['Y'], c=predicted_values, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', alpha=0.5)
plt.show()
