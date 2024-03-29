from liblinear.liblinearutil import *
import numpy as np
import random
import matplotlib.pyplot as plt

# 讀取資料
with open('hw4_train.txt', 'r') as file:
    data = file.read()
    lines = data.strip().split('\n')
x_train = []
y_train = []
# 生成 train data
for line in lines:
    numbers = [float(x) for x in line.split()]
    x_train.append(numbers[:6])
    y_train.append(numbers[6])


def transfer(data):
    feature = [1]
    # 加入x1到x6
    feature.extend(data)
    # 加入 x1*x1 x1*x2 到 x6*x6
    for i in range(6):
        for j in range(i, 6):
            feature.append(data[i] * data[j])
    # 加入 x1*x1*x1 x1*x1*x2 到 x3*x3*x3
    for i in range(6):
        for j in range(i, 6):
            for k in range(j, 6):
                feature.append(data[i] * data[j] * data[k])
    return feature


max_acc_list = []
for a in range(128):
    np.random.seed(a)
    x_data = []
    y_data = []
    x_val = []
    y_val = []
    all = list(range(0, 200))
    numbers = random.sample(range(0, 200), 120)

    for number in all:
        if number in numbers:
            x_data.append(transfer(x_train[number]))
            y_data.append(y_train[number])
        else:
            x_val.append(transfer(x_train[number]))
            y_val.append(y_train[number])

        lam = [0.000001, 0.0001, 0.01, 1, 100]
        lam_list = [-6, -4, -2, 0, 2]
        acc = []
    for i in range(5):
        mod = train(y_data, x_data, f'-s 0 -c {1 / (2 * lam[i])} -e 0.000001 -q')
        p_labels, p_acc, p_vals = predict(y_val, x_val, mod)
        acc.append(float(p_acc[0]))
    max_acc = max(acc)
    max_indices = [i for i, v in enumerate(acc) if v == max_acc]
    max_index = max(max_indices)
    max_acc_list.append(lam_list[max_index])

plt.hist(max_acc_list, bins=20)
plt.xlabel('log10(λ*)')
plt.ylabel('Frequency')
plt.title('Distribution of log10(λ*) from 128 experiments')
plt.show()
