import numpy as np

train_data = []
test_data = []

# read train test data
with open("hw6_train.txt", "r") as file:
    for line in file:
        items = line.strip().split()
        label = float(items[0])
        features = {int(item.split(':')[0]): float(item.split(':')[1]) for item in items[1:]}
        train_data.append((label, features))
with open("hw6_test.txt", "r") as file:
    for line in file:
        items = line.strip().split()
        label = float(items[0])
        features = {int(item.split(':')[0]): float(item.split(':')[1]) for item in items[1:]}
        test_data.append((label, features))


def squared_error(data, mean):
    labels = [label for label, _ in data]
    err = 0
    for y in labels:
        err += (y - mean) ** 2
    return err


# spilt the data by threshold
def split(data, feature, threshold):
    small = [item for item in data if item[1].get(feature, 0) <= threshold]
    large = [item for item in data if item[1].get(feature, 0) > threshold]
    return small, large


def square_loss(data, feature, threshold):
    small, large = split(data, feature, threshold)
    if not small or not large:
        return float('inf')
    # calculate the mean for square error small large data
    mean_small = np.mean([label for label, _ in small])
    loss_small = np.sum([(label - mean_small) ** 2 for label, _ in small])
    mean_large = np.mean([label for label, _ in large])
    loss_large = np.sum([(label - mean_large) ** 2 for label, _ in large])

    return loss_small + loss_large


def split_data(data):
    features = len(data[0][1])
    best_feature, best_threshold = None, None
    best_loss = float('inf')

    for feature in range(1, features + 1):
        feature_values = sorted(set(feat.get(feature, 0) for _, feat in data))
        for i in range(len(feature_values) - 1):
            threshold = 0.5 * (feature_values[i] + feature_values[i + 1])
            loss = square_loss(data, feature, threshold)
            # update loss feature and threshold
            if loss < best_loss:
                best_loss = loss
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


def tree(data_list):
    if len(set(x[0] for x in data_list)) == 1:
        return {'leaf': True, 'value': data_list[0][0]}

    best_feature, best_threshold = split_data(data_list)

    if best_feature is None:
        print(x[0] for x in data_list)
        return {'leaf': True, 'value': np.mean([x[0] for x in data_list])}

    left, right = split(data_list, best_feature, best_threshold)

    return {
        'left': tree(left),
        'right': tree(right),
        'leaf': False,
        'feature': best_feature,
        'threshold': best_threshold
    }


def predict(tree, features):
    if tree['leaf']:
        return tree['value']
    else:
        value = (features.get(tree['feature'], 0))
        if value <= tree['threshold']:
            return predict(tree['left'], features)
        else:
            return predict(tree['right'], features)


def error(data, tree):
    predictions = [predict(tree, features) for _, features in data]
    labels = [label for label, _ in data]
    predictions_arr = np.array(predictions)
    labels_arr = np.array(labels)
    return np.mean((predictions_arr - labels_arr) ** 2)


trees = tree(train_data)

train_err = error(train_data, trees)
test_err = error(test_data, trees)

print(f'E_out(g): {test_err}')
