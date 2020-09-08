import pickle
import numpy as np
import os
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return data


def classification_accuracy(prediction, ground_truth):
    ground_truth = ground_truth[:prediction.shape[0]]
    n_images = prediction.shape[0]
    x = prediction - ground_truth
    n_wrong_predictions = np.count_nonzero(x)
    accuracy = (n_images - n_wrong_predictions) / n_images

    return accuracy*100


def load_training_data(dataset_path):
    train_images = np.zeros([50000, 3072])
    train_labels = np.zeros([50000])

    start = 0
    n_images_in_a_file = 10000
    for i in range(1, 6):
        path = os.path.join(dataset_path, "data_batch_{}".format(i))
        data_dict = unpickle(path)
        train_images[start: start + n_images_in_a_file, :] = data_dict["data"]
        train_labels[start: start + n_images_in_a_file] = data_dict["labels"]
        start += n_images_in_a_file

    return np.asarray(train_images, dtype=np.int), np.asarray(train_labels, dtype=np.int)


def load_test_data(dataset_path):

    path = os.path.join(dataset_path, "test_batch")
    datadict = unpickle(path)
    test_images = datadict["data"]
    test_labels = datadict["labels"]
    return np.asarray(test_images, dtype=np.int), np.asarray(test_labels, dtype=np.int)


def cluster_indices(clust_num, labels_array):
    return np.where(labels_array == clust_num)[0]


def visualize(cluster_centroid):
    cluster_centroid = np.reshape(cluster_centroid, [10, 3, 32, 32]).transpose([0, 2, 3, 1])
    for i in range(10):
        centroid = cluster_centroid[i, :, :, :]
        centroid = centroid / centroid.max()
        plt.subplot(2, 5, i + 1)
        plt.axis('off')
        plt.title("cluster {} mean image".format(i + 1))
        plt.imshow(centroid)
    plt.show()


