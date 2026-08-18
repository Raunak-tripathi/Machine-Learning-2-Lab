import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

iris = load_iris(as_frame=True)
X=iris.data
print("Dataset shape:",X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k=3
kmean= KMeans(n_clusters=k, random_state=42,n_init=10)
cluster = kmean.fit_predict(X_scaled)

X_clustered = X.copy()
X_clustered['Cluster'] = cluster

print("Sample Clustered Dataset\n",X_clustered.head())

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=cluster,cmap='viridis',s=50)
plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],c='red',s=200, label="centroids")
plt.title('K-Means Clustering Iris')
plt.xlabel("Featur 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.legend()
plt.show()
