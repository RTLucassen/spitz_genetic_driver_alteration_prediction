"""
Cluster stain information.
"""

import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.spatial import distance

import matplotlib.pyplot as plt


path = Path(r"")

N_clusters = 5


if __name__ == '__main__':

    with open(path) as f:
        data = json.loads(f.read())

    # get features
    features = {}
    for k, v in data.items():
        features[k] = v['stain_matrix_target'][0]+v['stain_matrix_target'][1]#+v['maxC_target'][0]
    X = np.array(list(features.values()))

    if False:
        model = GaussianMixture(n_components=N_clusters)
        model.fit(X)

        for j in range(5):
            print(np.array([[model.means_[j][:3], model.means_[j][3:6]]]))
            plt.imshow(np.array([[model.means_[j][:3], model.means_[j][3:6]]]).astype(np.uint8))
            plt.show()

    if True:
        model = KMeans(n_clusters=N_clusters, random_state=0)
        model.fit(X)

    if False:
        for j in range(5):
            print(sum([1 if i == j else 0 for i in model.labels_.tolist()]))
            print(np.array([[model.cluster_centers_[j][:3], model.cluster_centers_[j][3:6]]]))
            plt.imshow(np.array([[model.cluster_centers_[j][:3], model.cluster_centers_[j][3:6]]]).astype(np.uint8))
            plt.show()
    
    for i in range(N_clusters):
        smallest_distance = None
        selected_key = None
        for key, feature in features.items():
            dst = distance.euclidean(model.cluster_centers_[i], feature)
            if smallest_distance is None or dst < smallest_distance:
                smallest_distance = dst
                selected_key = key
        
        print(smallest_distance, selected_key)