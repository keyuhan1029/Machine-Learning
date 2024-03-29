import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# separate the y data and x data
def data_process(line):
    data = line.split()
    # the first data of file is y , the rest of file is feature
    y = float(data[0])
    features = np.zeros(8)
    # make features an array
    for x in data[1:]:
        index, value = x.split(":")
        features[int(index) - 1] = float(value)
    return features, y


def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    features, y = zip(*[data_process(line) for line in data])
    x_arr = np.array(features)
    y_arr = np.array(y)
    return x_arr, y_arr


# split the data
def split(x, y, feature, threshold):
    small = x[:, feature] <= threshold
    large = x[:, feature] > threshold
    return x[small], y[small], x[large], y[large]


def square_error(y):
    err = 0
    mean = np.mean(y)
    for data in y:
        err += (data - mean) ** 2
    return err


def find_best_splits(data):
    best_splits = {}
    sample, features = data.shape
    for index in range(features):
        values = data[:, index]
        unique_values = np.unique(values)
        best_splits[index] = []
        for ind in range(len(unique_values)):
            if ind != 0:
                current_value = unique_values[ind]
                previous_value = unique_values[ind - 1]
                potential_split = (current_value + previous_value) / 2
                best_splits[index].append(potential_split)
    return best_splits


def bootstrap(x, y):
    # the size of samples
    samples = x.shape[0]
    # randomly chose half of size of sample to be the bootstrap dataset
    index = np.random.choice(samples, size=int(0.5 * samples), replace=True)
    return x[index], y[index]


def build_tree(x, y):
    samples, features = x.shape

    best_feature, best_threshold, best_error = None, None, float('inf')
    potential_splits = find_best_splits(x)

    if samples == 1 or len(np.unique(y)) == 1:
        leaf_value = np.mean(y)
        return Node(value=leaf_value)

    for index in potential_splits:
        for threshold in potential_splits[index]:
            _, y_left, _, y_right = split(x, y, index, threshold)
            error = square_error(y_left) + square_error(y_right)

            if error < best_error:
                best_error, best_feature, best_threshold = error, index, threshold

    if best_feature is not None:
        X_left, y_left, X_right, y_right = split(x, y, best_feature, best_threshold)
        left_subtree = build_tree(X_left, y_left)
        right_subtree = build_tree(X_right, y_right)
        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    return Node(value=np.mean(y))


def predict(tree, x):
    if tree.value is not None:
        return tree.value
    feature_val = x[tree.feature]
    # determine the branch
    branch = tree.right if feature_val >= tree.threshold else tree.left
    return predict(branch, x)


def cal_err(data, pre_data):
    err = np.mean((data - pre_data) ** 2)
    return err


def random_forest(X_train, y_train, X_test, y_test):
    trees = []
    E_ins = []
    E_outs = []
    E_outs_forest = []

    for i in range(2000):
        X_sample, y_sample = bootstrap(X_train, y_train)
        tree = build_tree(X_sample, y_sample)
        trees.append(tree)
        # predict the train data and test data
        predictions_train = [predict(tree, x) for x in X_train]
        predictions_test = [predict(tree, x) for x in X_test]
        forest_predictions = [np.mean([predict(t, x) for t in trees]) for x in X_test]
        # calculate the error
        err = cal_err(y_train, predictions_train)
        error = cal_err(y_test, predictions_test)
        E_out = cal_err(y_test, forest_predictions)
        E_ins.append(err)
        E_outs.append(error)
        E_outs_forest.append(E_out)
        print(f"tree {i}")

    return trees, E_ins, E_outs, E_outs_forest


x_train, y_train = read_data("hw6_train.txt")
x_test, y_test = read_data("hw6_test.txt")
trees, E_ins, E_outs, E_outs_forest = random_forest(x_train, y_train, x_test, y_test)
plt.hist(E_outs, bins=20, edgecolor='black')
plt.xlabel('Eout')
plt.ylabel('Frequency')
plt.title('freqency vs Eout')
plt.show()
