from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np


def principal_component_analysis(feature_list, n_component, feature_name, sub_idx, date):  # PCA
    feature_list = StandardScaler().fit_transform(feature_list)  # standardization

    pca = PCA(n_components=n_component)
    pComponents = pca.fit_transform(feature_list)
    print("Explaned variance ratio by principal components :", pca.explained_variance_ratio_, "\n Overall ratio: ",
          sum(pca.explained_variance_ratio_))


    return pComponents, pca.explained_variance_ratio_


def kneighbors(distance_matrix, length, k):
    neighbors = np.zeros((length, k), dtype=float)

    neigh = NearestNeighbors(n_neighbors=k, p=1)
    for i, vector in enumerate(distance_matrix):
        v = np.expand_dims(vector, axis=1)
        neigh.fit(v)
        neighbor = neigh.kneighbors([[vector[i]]], return_distance=False)
        neighbors[i] = neighbor.squeeze()
    neighbors = neighbors.astype(dtype='int32')
    return neighbors