import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

class DBSCAN():
    def __init__(self, eps=1, min_points=5, num_clusters=3, data=[], distances_array=[]):
        self.eps = eps
        self.min_points = min_points
        self.data = data
        self.num_clusters = num_clusters
        self.clusters = []
        self.distances_array = distances_array
        self.visited_samples = []
        self.neighbors_array = [[] for i in range(len(data))]

    def _get_neighbors(self, sample_i):
        neighbors = []
        for j in range(len(self.data)):
            if self.distances_array[sample_i][j] < self.eps:
                neighbors.append(j)
        return neighbors

    def _get_point_type(self, neighbors):
        if len(neighbors) >= self.min_points:
            return "Core"
        elif len(neighbors) < self.min_points and len(neighbors) != 0:
            return "Border"
        else:
            return "Noise"

    def _expand_cluster(self, sample_i, neighbors):
        cluster = [sample_i]
        for neighbor_i in neighbors:
            if not neighbor_i in self.visited_samples:
                self.visited_samples.append(neighbor_i)
                self.neighbors_array[neighbor_i] = self._get_neighbors(neighbor_i)
                if len(self.neighbors_array[neighbor_i]) >= self.min_points:
                    expanded_cluster = self._expand_cluster(neighbor_i, self.neighbors_array[neighbor_i])
                    cluster = cluster + expanded_cluster
                else:
                    cluster.append(neighbor_i)
        return cluster

    def _get_cluster_labels(self):
        labels = np.full(shape=self.data.shape[0], fill_value=len(self.clusters))
        for cluster_i, cluster in enumerate(self.clusters):
            for sample_i in cluster:
                labels[sample_i] = cluster_i
        return labels

    def run(self):
        n_samples = np.shape(self.data)[0]
        for sample_i in range(n_samples):
            if sample_i in range(n_samples):
                if sample_i in self.visited_samples:
                    continue
                self.neighbors_array[sample_i] = self._get_neighbors(sample_i)
                if len(self.neighbors_array[sample_i]) >= self.min_points:
                    self.visited_samples.append(sample_i)
                    new_cluster = self._expand_cluster(sample_i, neighbors=self.neighbors_array[sample_i])
                    self.clusters.append(new_cluster)

        return self.clusters  # Return self.clusters instead of cluster_labels

    def _map_from_cluster(self, cluster):
        points = np.empty(shape=[0, self.data.shape[1]])
        for i in cluster:
            temp = self.data[i, :]
            points = np.vstack((points, temp))
        return points

    def _plot_2d(self, clusters, axes, name):
        filename = name.strip()
        plt.scatter(clusters[0][:, 0], clusters[0][:, 1], color='red')
        plt.scatter(clusters[1][:, 0], clusters[1][:, 1], color='green')
        plt.scatter(clusters[2][:, 0], clusters[2][:, 1], color='blue')
        plt.title(name)
        plt.xlabel(axes[0])
        plt.ylabel(axes[1])
        plt.savefig(fname=filename)
        plt.show()
        return

    def Visualize(self, clusters):
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

        mapped_clusters = [self._map_from_cluster(cluster=cluster) for cluster in clusters]
        sepal_clusters = [cluster[:, :2] for cluster in mapped_clusters]
        petal_clusters = [cluster[:, 2:] for cluster in mapped_clusters]
        threeD_clusters = mapped_clusters

        for cluster, color in zip(sepal_clusters, colors[:len(clusters)]):
            plt.scatter(cluster[:, 0], cluster[:, 1], color=color)
        plt.title("Sepal length vs Sepal width")
        plt.xlabel("sepal length (in cm)")
        plt.ylabel("sepal width (in cm)")
        plt.savefig(fname="DBSCAN_Features_1_2")
        plt.show()

        for cluster, color in zip(petal_clusters, colors[:len(clusters)]):
            plt.scatter(cluster[:, 0], cluster[:, 1], color=color)
        plt.title("Petal length vs Petal width")
        plt.xlabel("petal length (in cm)")
        plt.ylabel("petal width (in cm)")
        plt.savefig(fname="DBSCAN_Features_3_4")
        plt.show()

        threeDfig = plt.figure()
        ax = threeDfig.add_subplot(111, projection='3d')

        for cluster, color in zip(threeD_clusters, colors[:len(clusters)]):
            ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], color=color)
        ax.set_xlabel('Sepal length (in cm)')
        ax.set_ylabel('Sepal width (in cm)')
        ax.set_zlabel('Petal length (in cm)')

        threeDfig.savefig(fname='DBSCAN_3D_clusters')

        return

if __name__ == '__main__':
    df = pd.read_csv("iris.data",
                     names=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"],
                     dtype={'SepalLength': float, 'SepalWidth': float, 'PetalLength': float, 'PetalWidth': float, 'Class': str}
                     )
    df = np.asarray(df)
    df = df[:, :4]

    distances = Distances(data=df)
    da = distances.get_distance_array()

    eps = 1
    min_points = 5

    algo = DBSCAN(eps=eps, min_points=min_points, data=df, distances_array=da)
    clusters = algo.run()
    algo.Visualize(clusters=clusters)