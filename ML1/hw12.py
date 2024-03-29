import numpy as np
import random
import matplotlib.pyplot as plt


# 計算中位數
def medium(num):
    sorted_num = sorted(num)
    size = len(num)
    if size % 2 == 1:  # 奇數
        median = sorted_num[size // 2]
    else:  # 偶數
        middle1 = sorted_num[(size // 2) - 1]
        middle2 = sorted_num[size // 2]
        median = (middle1 + middle2) / 2
    return median


times = []
for i in range(1000):
    random.seed(i)
    # 讀取資料
    with open('train.dat.txt', 'r') as file:
        data = file.read()
        lines = data.strip().split('\n')
    result = []
    for line in lines:
        numbers = [float(x) for x in line.split()]
        result.append(numbers)
    # 將x0 = 1 加入x矩陣
    for ib in range(len(result)):
        result[ib].insert(0, 1)
    # 初始化 w
    w = [0.0] * 13
    w_array = np.array(w)
    # 隨機取點 初始化 x
    random_point = random.choice(result)
    x_array = np.array(random_point)
    dot, update, success = 0, 0, 0
    sus = True
    min_length = 13
    while sus:
        # 判斷錯誤條件 持續更新直到正確
        if dot * random_point[13] <= 0:
            update += 1
            # 更新 w
            for ia in range(min_length):
                w_array[ia] = w_array[ia] + np.multiply(x_array, x_array[13])[ia]
            dot = 0
            for ind in range(min_length):
                dot += w_array[ind] * random_point[ind]
            # 成功次數歸零
            success = 0
        else:
            success += 1
            dot = 0
            random_point = random.choice(result)
            x_array = np.array(random_point)
            for ind in range(min_length):
                dot += w_array[ind] * x_array[ind]
            # 設定停止條件
            if success == 1280:
                sus = False
                times.append(update)
                update = 0
    sus = True
    print(i)
    if i == 999:
        med = medium(times)
        print(times)
        print(med)
        # 繪圖
        plt.figure()
        plt.hist(times, bins=50, edgecolor='k', align='left', alpha=0.75, color='b')
        plt.title('HW1-12')
        plt.xlabel('updates')
        plt.ylabel('times')
        plt.show()
