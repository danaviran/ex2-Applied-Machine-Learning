import numpy as np
import faiss


class KNNClassifier:
    def __init__(self, k, distance_metric='l2'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.Y_train = None

    def fit(self, X_train, Y_train):
        """
        Update the kNN classifier with the provided training data.
        Parameters:
        - X_train (numpy array) of size (N, d): Training feature vectors.
        - Y_train (numpy array) of size (N,): Corresponding class labels.
        """
        self.X_train = X_train.astype(np.float32)
        self.Y_train = Y_train
        d = self.X_train.shape[1]
        if self.distance_metric == 'l2':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L2)
        elif self.distance_metric == 'l1':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L1)
        else:
            raise NotImplementedError
        pass
        self.index.add(self.X_train)


    def predict(self, X):
        """
        Predict the class labels for the given data.
        Parameters: X (numpy array) of size (M, d): Feature vectors.
        Returns: (numpy array) of size (M,): Predicted class labels.
        Returns updated: array of k-nearest distances, array of k-nearest neighbors indexes,
            array of predictions - of each data point's final predicted label.
        """
        X = X.astype(np.float32)
        # Initialize an array to store predicted labels
        predictions = np.zeros(X.shape[0], dtype=self.Y_train.dtype)
        # Use knn_distance function to find k-nearest-neighbors for each data point in X
        distances, indexes = self.knn_distance(X)
        # Iterate through each data point in X
        for i in range(X.shape[0]):
            # Get indexes of k-nearest neighbors for the current data point
            cur_neighbors_indexes = indexes[i]
            # Get class labels of k-nearest neighbors
            cur_neighbors_labels = self.Y_train[cur_neighbors_indexes]
            # Count occurrences of each unique class label
            unique_labels, counts = np.unique(cur_neighbors_labels, return_counts=True)
            # Find the label with the majority count
            predicted_label = unique_labels[np.argmax(counts)]
            # Assign the predicted label to the predictions array
            predictions[i] = predicted_label
        # return array of k-nearest distances, array of k-nearest neighbors indexes, and an
        # array of predictions - of each data point's final predicted label
        return distances, indexes, predictions

    def knn_distance(self, X):
        """
        Calculate kNN distances for the given data.
        You must use the faiss library to compute the distances.
        See lecture slides and
        https://github.com/facebookresearch/faiss/wiki/Getting-started#in-python-2
        for more information.
        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.
        Returns:
        - (numpy array) of size (M, k): kNN distances.
        - (numpy array) of size (M, k): Indices of kNNs.
        """
        X = X.astype(np.float32)
        # Perform kNN search using the Faiss index
        distances, indexes = self.index.search(X, self.k)
        return distances, indexes
