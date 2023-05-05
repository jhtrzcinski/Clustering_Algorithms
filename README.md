# Cluster Algorithms: DBSCAN and SLAC

Author: Jacob Trzcinski

This is an implementation of the DBSCAN (Density-Based Spatial Clustering of Applications with Noise), and SLAC (Single Linkage Agglomerative Clustering) algorithms in Python. The implementation is designed to work with the Iris dataset, which contains information about iris flowers' sepal and petal dimensions. The goal is to cluster the iris flowers into groups based on their similarities.

## Requirements

To run this code, you need to have the following Python libraries installed:

- numpy
- pandas
- matplotlib

You also need the Iris dataset in a CSV file format, which can be downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris). The file should be named `iris.data`.

These algorithms assume that the labels column is included when the data is imported!!

## Running the DBSCAN Algorithm

To run the DBSCAN algorithm, follow these steps:

1. Download the `iris.data` file from the link provided above and place it in the same directory as the Python script.
2. Open a terminal or command prompt, navigate to the directory containing the Python script, and run the following command: `python DBSCAN_Main.py`.

The script will run the DBSCAN algorithm on the Iris dataset and generate three plots:

1. Sepal length vs Sepal width: A 2D scatter plot showing the clusters based on sepal length and width.
2. Petal length vs Petal width: A 2D scatter plot showing the clusters based on petal length and width.
3. 3D plot: A 3D scatter plot showing the clusters based on sepal length, sepal width, and petal length.

The plots will be saved as image files in the same directory as the Python script.

## Customizing DBSCAN Hyperparameters

To change the hyperparameters of the DBSCAN algorithm (i.e., `eps` and `min_points`), you can modify the following line in the Python script:

```python
algo = DBSCAN(eps=1, min_points=5, data=df, distances_array=da)
```

## Analysis of the DBSCAN Algorithm

To determine the complexity of this DBSCAN implementation, we must analyze each section of the code.

1. Distance class: This portion of code calculates the distance matrix. in the `get_distance_array` function, we have two nested loops, each running n times, where n is the number of observations/data points. This means that the complexity of this portion is O(n^2).
2. DBSCAN class:
   
   a. `_get_neighbors` function: This function has a loop that iterates over n data points. So, the time complexity is O(n).

   b. `_expand_cluster` function: Worst case scenario, this method is called once for each data point. Inside this function, we have a loop iterating over the neighbors, which, at most, can be n. So, the time complexity is O(n^2).

   c. `run` function: This function has a loop that iterates over n data points. Inside the loop, we call `_get_neighbors` (O(n)) and possibly `_expand_cluster` (O(n^2)). So, the time complexity is O(n^3).

Overall, the time complexity of the DBSCAN algorithm is O(n^3) in the worst case. Which isn't too bad for a DIY program.

One of the most glaring issues with the DIY DBSCAN implementation is that it only discovers two clusters. This is not reality since, we know from the `iris.names` file, that there are three types of irises in the database. We suspect that due to a large default epsilon, the clusters have merged during expansion. 

To combat this, we can decrease epsilon manually and see if more clusters appear (at the cost of our sanity), we can create a silhouette method to autonomously optimize our hyperparameters (at the cost of our time writing that), or we can manually set that we expect three clusters (at the cost of generality).

## Running the SLAC Algorithm

To run the SLAC algorithm, follow these steps:

1. Download the `iris.data` file from the link provided above and place it in the same directory as the Python script.
2. Open a terminal or command prompt, navigate to the directory containing the Python script, and run the following command: `python SLAC_Main.py`.

The script will run the SLAC algorithm on the Iris dataset and generate three plots:

1. Sepal length vs Sepal width: A 2D scatter plot showing the clusters based on sepal length and width.
2. Petal length vs Petal width: A 2D scatter plot showing the clusters based on petal length and width.
3. 3D plot: A 3D scatter plot showing the clusters based on sepal length, sepal width, and petal length.

The plots will be saved as image files in the same directory as the Python script.

## Customizing SLAC Hyperparameters

To change the hyperparameters of the SLAC algorithm (i.e., `num_clusters`), you can modify the following line in the Python script:

```python
slac = SLAC(data=df, distances_array=da, num_clusters=X)
```

## Analysis of the SLAC Algorithm

To determine the complexity of this SLAC implementation, we must evaluate each portion of the code.

1. Distance class: Same as DBSCAN (I copied it for obvious reasons). We have two nested loops each running n times, where n is the number of observations/data points. This means that the complexity of this portion is O(n^2).
2. SLAC class: Similarly to DBSCAN, in the worst case, each iteration of the algorithm to cluster requires comparing every cluster to every other cluster. Since the maximum number of clusters can be each lone point/observation, the time complexity of this class is O(n^3).

Overall, the time complexity of this SLAC implementation is O(n^3) in the worst case. Again, not terrible for a DIY project. 

While this algorithm is better than our DBSCAN implementation because it has three clusters, we are far from it being correct. From the default hyperparameters, we have that the third cluster (in blue), is just two observations on the outskirts of the data. 

This behavior shows a lack of maturity of the hyperparameters, since three clusters is hardcoded as the default. Perhaps adjusting the use of variables like `min_distance` would yield better results.