from Utils.Reshape import flatten
import numpy as np
import random


def preprocessing(feature_list, sample_count_list, n_samples, n_labels_by_trials, n_nodes, n_features,
                  seed):  # flattening and simultaneous random shuffling
    labeled_data_identifier = np.zeros(n_samples)
    start = 0
    end = 0
    random.seed(seed)
    for num in sample_count_list:
        end += num
        idcs = random.sample(range(start, end - 1), n_labels_by_trials)
        labeled_data_identifier[idcs] = 1
        start += num

    flattened_feature_list = flatten(feature_list, n_nodes * n_features)

    return flattened_feature_list, labeled_data_identifier


def other_preprocessing(f1, f2, label_list, n_labels_by_class, n_classes, seed,
                        isdeap=False):  # flattening and simultaneous random shuffling
    shape = f1.shape
    n_samples = shape[0]
    n_features = shape[1]
    n_nodes = shape[2]

    ff1 = flatten(f1, n_features * n_nodes)
    ff2 = flatten(f2, n_features * n_nodes)

    random.seed(seed)

    if isdeap == True:
        vlc_label = label_list[0]
        ars_label = label_list[1]

        vlc_labeled_data_identifier = np.zeros(n_samples)
        ars_labeled_data_identifier = np.zeros(n_samples)

        for i in range(n_classes):
            indices = np.where(vlc_label == i)[0]
            labeled_idx = random.sample(sorted(indices), n_labels_by_class)
            vlc_labeled_data_identifier[labeled_idx] = 1

        for i in range(n_classes):
            indices = np.where(ars_label == i)[0]
            labeled_idx = random.sample(sorted(indices), n_labels_by_class)
            ars_labeled_data_identifier[labeled_idx] = 1

        return ff1, ff2, vlc_labeled_data_identifier, ars_labeled_data_identifier

    else:
        labeled_data_identifier = np.zeros(n_samples)
        for i in range(n_classes):
            indices = np.where(label_list == i)[0]
            labeled_idx = random.sample(sorted(indices), n_labels_by_class)
            labeled_data_identifier[labeled_idx] = 1

        return ff1, ff2, labeled_data_identifier