import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def optimalK(data, max_k):
    wcss = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='-')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.show()


features_df = pd.DataFrame(pd.read_pickle('saved_features\save_train_dense_k25_D1_train.pkl')['features'])[["features_RGB"]]
features = np.squeeze(features_df.values, axis=1)

middle_frame_index = features[0].shape[0] // 2 + 1
selected_frames = []
for i in features:
    selected_frames.append(i[middle_frame_index, :])

data = np.array(selected_frames)

# optimalK(data, 30) # 8
# exit()

kmeans = KMeans(n_clusters=8, init='k-means++', random_state=42)
kmeans.fit(data)
# features_df['Cluster'] = kmeans.labels_

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('K-means Clustering (2D PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()