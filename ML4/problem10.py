from liblinear.liblinearutil import *
import numpy as np
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


x_data = []
for i in range(200):
    x_data.append(transfer(x_train[i]))

# c = 1/2(lamda)

lam = [0.000001, 0.0001, 0.01, 1, 100]

for i in range(5):
    mod = train(y_train, x_data, f'-s 0 -c {1 / (2 * lam[i])} -e 0.000001 -q')
    p_labels, p_acc, p_vals = predict(y_train, x_data, mod)
    # print(p_acc)
