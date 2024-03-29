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

# 將1000筆更新次數加入列表
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
    # 初始化 w 並轉為陣列
    w = [0.0] * 13
    w_array = np.array(w)
    # 隨機取點 初始化 x
    random_point = random.choice(result)
    x_array = np.array(random_point)*11.26  # 將資料同乘 11.26
    dot, update, success = 0, 0, 0  # 將內積, 更新次數, 成功次數初始化為零
    sus = True
    min_length = 13
    while sus:
        # 判斷錯誤條件
        if dot * random_point[13] <= 0:
            update += 1
            # 更新 w
            for ia in range(min_length):
                w_array[ia] = w_array[ia] + np.multiply(x_array, x_array[13]/11.26)[ia]
            # 內積歸零 並重新選一組資料
            dot = 0
            random_point = random.choice(result)
            x_array = np.array(random_point)*11.26
            # 重新計算內積
            for ind in range(min_length):
                dot += w_array[ind] * random_point[ind]
            success = 0  # 成功次數歸零
        else:
            success += 1  # 成功次數加一
            dot = 0  # 內積歸零
            # 重新選一組
            random_point = random.choice(result)
            x_array = np.array(random_point)*11.26
            for ind in range(min_length):
                dot += w_array[ind] * x_array[ind]
            # 設定停止條件
            if success == 1280:
                sus = False  # 跳出迴圈
                times.append(update)  # 將更新次數加入列表
                update = 0  # 更新次數歸零
    sus = True
    print(i)
    if i == 999:
        med = medium(times)
        print(times)
        print(med)
        # 繪圖
        plt.figure()
        plt.hist(times, bins=50, edgecolor='k', align='left', alpha=0.75, color='b')
        plt.title('HW1-10')
        plt.xlabel('updates')
        plt.ylabel('times')
        plt.show()
