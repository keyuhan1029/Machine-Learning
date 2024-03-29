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


def val(x_val, y_val):
    x_data = []
    y_data = []
    if x_val == xf1:
        x_data = xf2 + xf3 + xf4 + xf5
        y_data = yf2 + yf3 + yf4 + yf5
    elif x_val == xf2:
        x_data = xf1 + xf3 + xf4 + xf5
        y_data = yf1 + yf3 + yf4 + yf5
    elif x_val == xf3:
        x_data = xf1 + xf2 + xf4 + xf5
        y_data = yf1 + yf2 + yf4 + yf5
    elif x_val == xf4:
        x_data = xf1 + xf2 + xf3 + xf5
        y_data = yf1 + yf2 + yf3 + yf5
    elif x_val == xf5:
        x_data = xf1 + xf2 + xf3 + xf4
        y_data = yf1 + yf2 + yf3 + yf4

    lam = [0.000001, 0.0001, 0.01, 1, 100]
    acc = np.array([])

    for i in range(5):
        mod = train(y_data, x_data, f'-s 0 -c {1 / (2 * lam[i])} -e 0.000001 -q')
        p_labels, p_acc, p_vals = predict(y_val, x_val, mod)
        acc = np.append(acc, float(p_acc[0]))
    return acc

max_acc_list = []
for a in range(128):
    np.random.seed(a)
    numbers = np.arange(0, 200)
    np.random.shuffle(numbers)
    subset = np.array_split(numbers, 5)
    set_1, set_2, set_3, set_4, set_5 = subset
    xf1 = []
    yf1 = []
    xf2 = []
    yf2 = []
    xf3 = []
    yf3 = []
    xf4 = []
    yf4 = []
    xf5 = []
    yf5 = []
    for i in set_1:
        xf1.append(transfer(x_train[i]))
        yf1.append(y_train[i])
    for i in set_2:
        xf2.append(transfer(x_train[i]))
        yf2.append(y_train[i])
    for i in set_3:
        xf3.append(transfer(x_train[i]))
        yf3.append(y_train[i])
    for i in set_4:
        xf4.append(transfer(x_train[i]))
        yf4.append(y_train[i])
    for i in set_5:
        xf5.append(transfer(x_train[i]))
        yf5.append(y_train[i])
    acc_total = val(xf1, yf1)+val(xf2, yf2)+val(xf3, yf3)+val(xf4, yf4)+val(xf5, yf5)
    max_acc = max(acc_total)
    max_indices = [i for i, v in enumerate(acc_total) if v == max_acc]
    max_index = max(max_indices)
    lam_list = [-6, -4, -2, 0, 2]
    max_acc_list.append(lam_list[max_index])

plt.hist(max_acc_list, bins=20)
plt.xlabel('log10(λ*)')
plt.ylabel('Frequency')
plt.title('Distribution of log10(λ*) from 128 experiments')
plt.show()
