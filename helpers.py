import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from knn import KNNClassifier


def decision_tree_demo():
    # Create random data
    np.random.seed(42)
    X = np.random.rand(100, 2)  # Feature matrix with 100 samples and 2 features
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary labels based on a simple condition
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize Decision Tree classifier
    tree_classifier = DecisionTreeClassifier(random_state=42)
    # Train the Decision Tree on the training data
    tree_classifier.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = tree_classifier.predict(X_test)
    # Compute the accuracy of the predictions
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")


def loading_random_forest():
    model = RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=4)


def loading_xgboost():
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=4)


def plot_decision_boundaries(model, X, y, title='Decision Boundaries'):
    """
    Plots decision boundaries of a classifier and colors the space by the prediction of each point.

    Parameters:
    - model: The trained classifier (sklearn model).
    - X: Numpy Feature matrix.
    - y: Numpy array of Labels.
    - title: Title for the plot.
    """
    # h = .02  # Step size in the mesh

    # enumerate y
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    enum_y = np.array([y_map[v] for v in y]).astype(int)

    h_x = (np.max(X[:, 0]) - np.min(X[:, 0])) / 200
    h_y = (np.max(X[:, 1]) - np.min(X[:, 1])) / 200

    # Plot the decision boundary.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Make predictions on the meshgrid points.
    # add 'distances, indexes' when using kNN model
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([y_map[v] for v in Z])
    Z = Z.reshape(xx.shape)
    vmin = np.min([np.min(enum_y), np.min(Z)])
    vmax = np.min([np.max(enum_y), np.max(Z)])

    # Plot the decision boundary.
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, vmin=vmin, vmax=vmax)

    # Scatter plot of the data points with matching colors.
    plt.scatter(X[:, 0], X[:, 1], c=enum_y, cmap=plt.cm.Paired, edgecolors='k', s=40, alpha=0.7,
                vmin=vmin, vmax=vmax)

    plt.title("Decision Boundaries")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.show()


def read_data_demo(filename):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """
    # the data in pandas dataframe format
    df = pd.read_csv(filename)
    # Extract feature columns (assuming the last column is the label)
    X_train = df.iloc[:, :-1].values
    # Extract label column
    Y_train = df.iloc[:, -1].values
    return X_train, Y_train


def create_kNN_model(X_train, Y_train, X_test, Y_test, k, distance_metric):
    """
    Notice the similarity to the decision tree demo above.
    This is the sklearn standard format for models.
    """
    # Initialize the KNNClassifier
    knn_classifier = KNNClassifier(k=k, distance_metric=distance_metric)
    # Train the classifier
    knn_classifier.fit(X_train, Y_train)
    # Predict the labels for the test set
    distances, indexes, y_pred = knn_classifier.predict(X_test)
    # Calculate the accuracy of the classifier
    accuracy = np.mean(y_pred == Y_test)
    return knn_classifier, y_pred, accuracy


def kNN_scenario():
    x_train, y_train = read_data_demo('train.csv')
    x_test, y_test = read_data_demo('test.csv')
    models = []
    predictions = []
    accuracies = []
    # Iterate over distance metrics
    for distance_metric in ['l1', 'l2']:
        cur_models = []
        cur_preds = []
        cur_accuracies = []
        # Iterate over values of k
        for k in [1, 10, 100, 1000, 3000]:
            model, y_pred, accuracy = create_kNN_model(x_train, y_train, x_test, y_test, k,
                                                       distance_metric)
            # Append model information and accuracy to lists
            cur_models.append(model)
            cur_preds.append(y_pred)
            cur_accuracies.append(accuracy)
        models.append(cur_models)
        predictions.append(cur_preds)
        accuracies.append(cur_accuracies)
    create_kNN_result_table(accuracies)
    # Plot decision boundaries for L2, k_max
    plot_decision_boundaries(models[0][0], x_test, y_test, 'L1, k=1')


def create_kNN_result_table(accuracies):
    # Round to 4 decimal places
    accuracies = np.round(np.array(accuracies), 4)
    data = np.column_stack(([1, 10, 100, 1000, 3000], accuracies[0], accuracies[1]))
    # Create a table using Matplotlib
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=data, colLabels=['K', 'L1 distance metric', 'L2 distance metric'], loc='center')
    plt.show()


def anomaly_detection_scenario():
    x_train, y_train = read_data_demo('train.csv')
    df = pd.read_csv('AD_test.csv')
    X_test = df.iloc[:, :].values
    # Initialize the KNNClassifier with k=5 and L2 distance metric
    knn_classifier = KNNClassifier(k=5, distance_metric='l2')
    knn_classifier.fit(x_train, y_train)
    # Predict the labels for the test set
    distances, indexes, y_pred = knn_classifier.predict(X_test)
    distance_sums = np.sum(distances, axis=1)
    top_50_indexes = np.argsort(distance_sums)[-50:]
    top_50_coordinates = X_test[top_50_indexes]
    # Plot test points
    plt.scatter(X_test[:, 0], X_test[:, 1], color='blue', label='test data',alpha=0.5, s=3)
    # Plot top 50 largest distances in red
    plt.scatter(top_50_coordinates[:, 0], top_50_coordinates[:, 1], color='red',
                label='top 50 anomalies in test data', alpha=1.0, s=3)
    # Plot train points
    plt.scatter(x_train[:, 0], x_train[:, 1], color='black', alpha=0.01, label='train data', s=3)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Scatter Plot of AD Test and Train Data')
    plt.legend()
    plt.show()


def decision_tree_scenario():
    x_train, y_train = read_data_demo('train.csv')
    x_validation, y_validation = read_data_demo('validation.csv')
    x_test, y_test = read_data_demo('test.csv')

    models_table = []
    max_depths_table = []
    max_leaf_nodes_table = []
    train_accuracy_table = []
    validation_accuracy_table = []
    test_accuracy_table = []

    for max_leaf_nodes in [50, 100, 1000]:
        cur_models = []
        cur_max_depths = []
        cur_max_leaf_nodes = []
        cur_train_accuracy = []
        cur_validation_accuracy = []
        cur_test_accuracy = []
        for max_depth in [1, 2, 4, 6, 10, 20, 50, 100]:
            # Initialize Decision Tree classifier
            tree_classifier = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes,
                                                     max_depth=max_depth, random_state=42)
            tree_classifier.fit(x_train, y_train)
            # Save model and hyper-parameters
            cur_models.append(tree_classifier)
            cur_max_leaf_nodes.append(max_leaf_nodes)
            cur_max_depths.append(max_depth)

            # Make predictions on the train data
            train_y_pred = tree_classifier.predict(x_train)
            train_accuracy = np.mean(train_y_pred == y_train)
            cur_train_accuracy.append(train_accuracy)

            # Make predictions on the test data
            test_y_pred = tree_classifier.predict(x_test)
            test_accuracy = np.mean(test_y_pred == y_test)
            cur_test_accuracy.append(test_accuracy)

            # Make predictions on the validation data
            validation_y_pred = tree_classifier.predict(x_validation)
            validation_accuracy = np.mean(validation_y_pred == y_validation)
            cur_validation_accuracy.append(validation_accuracy)

            """
            # Visualize the decision tree
            plt.figure(figsize=(10, 6))
            if isinstance(x_train, pd.DataFrame):
                plot_tree(tree_classifier, filled=True, feature_names=list(x_train.columns),
                          class_names=True, rounded=True)
            else:
                plot_tree(tree_classifier, filled=True, class_names=True, rounded=True)
            plt.title(f"Decision Tree - Max Leaf Nodes: {max_leaf_nodes}, Max Depth: {max_depth}")
            plt.show()
            """
        # append
        models_table.append(cur_models)
        max_depths_table.append(cur_max_depths)
        max_leaf_nodes_table.append(cur_max_leaf_nodes)
        train_accuracy_table.append(cur_train_accuracy)
        validation_accuracy_table.append(cur_validation_accuracy)
        test_accuracy_table.append(cur_test_accuracy)
    # visualize:
    # create_tree_accuracy_table(train_accuracy_table)
    # create_tree_accuracy_table(validation_accuracy_table)
    # create_tree_accuracy_table(test_accuracy_table)
    # Best Tree on Validation Data (Max depth of 20, max leaves of 1000)
    plot_decision_boundaries(models_table[2][5], x_train, tree_classifier.predict(x_train),
                             'Decision Tree Model\nMax Leaves: 1000, Max Depth: 20\nTrain data')
    plot_decision_boundaries(models_table[2][5], x_validation, tree_classifier.predict(x_validation),
                             'Decision Tree Model\nMax Leaves: 1000, Max Depth: 20\nValidation data')
    plot_decision_boundaries(models_table[2][5], x_test, tree_classifier.predict(x_test),
                             'Decision Tree Model\nMax Leaves: 1000, Max Depth: 20\nTest data')
    # Best Tree on Validation Data of Maximum 50 leaves (max depth of 20)
    plot_decision_boundaries(models_table[0][5], x_train, tree_classifier.predict(x_train),
                             'Decision Tree Model\nMax Leaves: 50, Max Depth: 20\nTrain data')
    plot_decision_boundaries(models_table[0][5], x_validation, tree_classifier.predict(x_validation),
                             'Decision Tree Model\nMax Leaves: 50, Max Depth: 20\nValidation data')
    plot_decision_boundaries(models_table[0][5], x_test, tree_classifier.predict(x_test),
                             'Decision Tree Model\nMax Leaves: 50, Max Depth: 20\nTest data')
    # Best Tree of Maximum 6 depth (max 50 leaves)
    plot_decision_boundaries(models_table[0][3], x_train, tree_classifier.predict(x_train),
                             'Decision Tree Model\nMax Leaves: 50, Max Depth: 6\nValidation data')
    plot_decision_boundaries(models_table[0][3], x_validation, tree_classifier.predict(x_validation),
                             'Decision Tree Model\nMax Leaves: 50, Max Depth: 6\nTrain data')
    plot_decision_boundaries(models_table[0][3], x_test, tree_classifier.predict(x_test),
                             'Decision Tree Model\nMax Leaves: 50, Max Depth: 6\nTest data')


def create_tree_accuracy_table(accuracy_table):
    accuracy_table = np.round(accuracy_table, 4)
    max_leaves_values = [50, 100, 1000]
    max_depth_values = [1, 2, 4, 6, 10, 20, 50, 100]
    # Create a 3x8 table
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.text(0.5, 0.63, 'Max Depth', ha='center', va='center', fontsize=10, transform=ax.transAxes)
    ax.text(-0.11, 0.5, 'Max Leaves', ha='center', va='center', rotation='vertical', fontsize=10,
            transform=ax.transAxes)
    columns = [''] + [str(depth) for depth in max_depth_values]
    table = ax.table(cellText=accuracy_table,
                     colLabels=[str(depth) for depth in max_depth_values],
                     rowLabels=max_leaves_values, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1)
    ax.set_title('Test Accuracy Table', y=1.02, pad=-90, fontsize=12, ha='center', va='bottom')
    plt.show()


def random_forest_scenario():
    x_train, y_train = read_data_demo('train.csv')
    x_validation, y_validation = read_data_demo('validation.csv')
    x_test, y_test = read_data_demo('test.csv')
    model = RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=4)
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    validation_pred = model.predict(x_validation)
    test_pred = model.predict(x_test)
    train_accuracy = np.mean(train_pred == y_train)
    validation_accuracy = np.mean(validation_pred == y_validation)
    test_accuracy = np.mean(test_pred == y_test)
    print("train accuracy: ", train_accuracy)
    print("validation accuracy: ", validation_accuracy)
    print("test accuracy: ", test_accuracy)

    # plot
    # plot_decision_boundaries(model, x_train, model.predict(x_train),'Random Forest Decision Boundaries - Train Set')
    # plot_decision_boundaries(model, x_validation, model.predict(x_validation),'Random Forest Decision Boundaries - Validation Set')
    # plot_decision_boundaries(model, x_test, model.predict(x_test), 'Random Forest Decision Boundaries - Test Set')


def XGBoost_scenario():
    x_train, y_train = read_data_demo('train.csv')
    x_validation, y_validation = read_data_demo('validation.csv')
    x_test, y_test = read_data_demo('test.csv')
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=4)
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    validation_pred = model.predict(x_validation)
    test_pred = model.predict(x_test)
    train_accuracy = np.mean(train_pred == y_train)
    validation_accuracy = np.mean(validation_pred == y_validation)
    test_accuracy = np.mean(test_pred == y_test)
    print("train accuracy: ", train_accuracy)
    print("validation accuracy: ", validation_accuracy)
    print("test accuracy: ", test_accuracy)
    # plot_decision_boundaries(model, x_train, train_pred, 'XGBoost Decision Boundaries - Train Set')
    # plot_decision_boundaries(model, x_validation, validation_pred, 'XGBoost Decision Boundaries - Validation Set')
    # plot_decision_boundaries(model, x_test, test_pred, 'XGBoost Decision Boundaries - Test Set')


if __name__ == '__main__':
    np.random.seed(0)
    # kNN_scenario()
    # anomaly_detection_scenario()
    # decision_tree_scenario()
    # random_forest_scenario()
    # XGBoost_scenario()