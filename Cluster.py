# Importing essential libraries
import numpy as np
import pandas as pd
import pickle
import xlrd
import seaborn
import matplotlib.pyplot as plt

# Loading the dataset
df = pd.read_csv('Data/42674_74935_bundle_archive.xls')

# Replacing the 0 values  by NaN
df_copy = df.copy(deep=True)
df_copy[['CustomerID','Gender','Age','Annual income','Spending score']] = df_copy[['CustomerID','Gender','Age','Annual income','Spending score']].replace(0,np.NaN)


#Dropping CustomerID , as it will not help in prediction
df_copy.drop('CustomerID',axis=1)

#selecting 'Annual income' and 'spending score' as the feature for clustering
X=df_copy.iloc[:,[2,3]]


#calculating WCSS values for 1 to 10 clusters

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
  kmeans_model = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans_model.fit(X)
  wcss.append(kmeans_model.inertia_)

# Plotting the WCSS values
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Training the KMeans model with n_clusters=5
kmeans_model = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans_model.fit_predict(X)


















