from sklearn.cluster import KMeans
import utils
import numpy as np
from scipy import stats


def main(path):

    train_images, train_labels = utils.load_training_data(path)
    test_images, test_labels = utils.load_test_data(path)

    n_classes = 10
    model = KMeans(n_clusters=n_classes, n_init=1, max_iter=1)
    model.fit(train_images)

    # which images are assigned to each cluster:
    # 1. check all data points assigned to each cluster
    # 2. check actual labels of the data points assigned to each cluster
    # 3. assign the mode of actual labels to be the label for that cluster

    cluster_label_dict = {}
    for cluster_num in range(n_classes):
        idx = utils.cluster_indices(cluster_num, model.labels_)
        original_labels = np.take(train_labels, idx)
        mode = stats.mode(original_labels)[0][0]
        cluster_label_dict.update({cluster_num: mode})

    # prediction
    predicted_cluster = model.predict(test_images)
    predicted_labels = np.vectorize(cluster_label_dict.get)(predicted_cluster)

    accuracy = utils.classification_accuracy(predicted_labels, test_labels)
    print(" K means clustering accuracy for cifar 10 = {}".format(accuracy))

    # visualise clusters
    cluster_centroids = model.cluster_centers_
    utils.visualize(cluster_centroids)


if __name__ == '__main__':
    data_path = "cifar-10-batches-py"
    main(data_path)