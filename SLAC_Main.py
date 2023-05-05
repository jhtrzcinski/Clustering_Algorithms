import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Distances():
    def __init__(self, data):
        self.data = data
        self.len_data = len(data)

    def get_distance_array(self):
        distance_array = np.zeros([self.len_data, self.len_data])
        for i in range(self.len_data):
            for j in range(self.len_data):
                distance_array[i][j] = self._get_euclidean_distance(i, j)
        return distance_array

    def _get_euclidean_distance(self, i, j):
        return np.sqrt(np.sum((self.data[i,:] - self.data[j,:])**2))

class SLAC():
    def __init__(self, data, distances_array, num_clusters=3):
        self.data = data
        self.distances_array = distances_array
        self.num_clusters = num_clusters
        self.clusters = []

    def run(self):
        clusters = [[i] for i in range(len(self.data))]
        while len(clusters) > self.num_clusters:
            i_min, j_min, min_distance = self._find_min_distance(clusters)

            merged_cluster = clusters[i_min] + clusters[j_min]
            clusters[i_min] = merged_cluster
            clusters.pop(j_min)

        self.clusters = clusters
        return self.clusters

    def _find_min_distance(self, clusters):
        i_min = 0
        j_min = 1
        min_distance = self._cluster_distance(clusters[i_min], clusters[j_min])

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = self._cluster_distance(clusters[i], clusters[j])
                if distance < min_distance:
                    i_min, j_min, min_distance = i, j, distance

        return i_min, j_min, min_distance

    def _cluster_distance(self, cluster1, cluster2):
        min_distance = float('inf')
        for i in cluster1:
            for j in cluster2:
                distance = self.distances_array[i][j]
                if distance < min_distance:
                    min_distance = distance
        return min_distance
    
    def _map_from_cluster(self, cluster):
        points = np.empty(shape=[0, self.data.shape[1]])
        for i in cluster:
            temp = self.data[i, :]
            points = np.vstack((points, temp))
        return points

    def _plot_2d(self, clusters, axes, name, filename):
        colors = ['red', 'green', 'blue']
        for i, cluster in enumerate(clusters):
            plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i])
        plt.title(name)
        plt.xlabel(axes[0])
        plt.ylabel(axes[1])
        plt.savefig(filename)
        plt.show()
        return

    def Visualize(self, clusters):
        mapped_clusters = [self._map_from_cluster(cluster) for cluster in clusters]

        feature1_clusters = [cluster[:, :2] for cluster in mapped_clusters]
        feature2_clusters = [cluster[:, 2:] for cluster in mapped_clusters]

        self._plot_2d(clusters=feature1_clusters, axes=["Feature 1", "Feature 2"], name="SLAC Clustering (Features 1 & 2)", filename="SLAC_Features_1_2.png")
        self._plot_2d(clusters=feature2_clusters, axes=["Feature 3", "Feature 4"], name="SLAC Clustering (Features 3 & 4)", filename="SLAC_Features_3_4.png")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = ['red', 'green', 'blue']
        for i, cluster in enumerate(mapped_clusters):
            ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], color=colors[i])

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')

        plt.savefig("SLAC_3D_clusters.png")
        plt.show()

if __name__ == '__main__':
    df = pd.read_csv("iris.data",
                     names=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"],
                     dtype={'SepalLength': float, 'SepalWidth': float, 'PetalLength': float, 'PetalWidth': float, 'Class': str}
                     )
    df = np.asarray(df)
    df = df[:, :4]

    distances = Distances(data=df)
    da = distances.get_distance_array()

    slac = SLAC(data=df, distances_array=da)
    clusters = slac.run()

    slac.Visualize(clusters=clusters)