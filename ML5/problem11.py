from libsvm.svmutil import *
import random
import matplotlib.pyplot as plt
import numpy as np

y_train, x_train = svm_read_problem("train.txt")
y_data = [1 if y == 1 else -1 for y in y_train]

C = [0.01, 0.1, 1, 10, 100]

max_acc_list = []
selection_frequencies = []
for i in range:
    np.random.seed(i)
    min_Eout = float('inf')
    # 選出train 跟 validation 的 index
    validation_indices = random.sample(range(len(y_train)), 200)
    training_indices = [i for i in range(len(y_train)) if i not in validation_indices]
    # 把資料分成 train data 跟 validation data
    y_training, x_training = [y_data[i] for i in training_indices], [x_train[i] for i in training_indices]
    y_validation, x_validation = [y_data[i] for i in validation_indices], [x_train[i] for i in validation_indices]
    best_c = None
    for c in C:
        model = svm_train(y_training, x_training, f'-t 2 -c {c} -g 1 -q')
        p_label, p_acc, p_val = svm_predict(y_validation, x_validation, model)
        err = 100 - p_acc[0]
        if err < min_Eout:
            min_Eout = err
            best_c = c
    max_acc_list.append(best_c)
    selection_frequencies = [max_acc_list.count(c) for c in C]
logc = [-2, -1, 0, 1, 2]
# Plot the bar chart
plt.bar(logc, selection_frequencies, align="center")
plt.xlabel('log C Values')
plt.ylabel('Selection Frequency')
plt.title('C Selection Frequency with Random Validation')
plt.show()
