import os 
import cv2, numpy as np
from sklearn.cluster import KMeans

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname,'data/')
image_data = os.listdir(filename)

def visualize_colors(cluster, centroids):
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    red, green, blue = colors[0][1][0], colors[0][1][1], colors[0][1][2]
    return red, green, blue

for path in image_data:
    image = cv2.imread('data/' + path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))
    cluster = KMeans(n_clusters=1).fit(reshape)
    print(visualize_colors(cluster, cluster.cluster_centers_))