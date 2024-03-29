import numpy as np
import matplotlib.pyplot as plt
# 存2000個 Ein Eout theta epsilon 的 list
EinList = []
EoutList = []
thetaList = []
epsilonList = []
for time in range(2000):
    # 在1到-1中隨機選32個值 並排序
    x_random = np.random.uniform(-1, 1, 8)
    x = np.sort(x_random)
    # 創建theta
    theta = np.mean([x[i:i + 2] for i in range(7)], axis=1)
    theta = np.insert(theta, 0, -1)
    # 定義noise及y
    noise = np.random.choice([1, -1], 8, p=[0.9, 0.1])
    y = np.sign(x) * noise
    # 定義s
    s = [1, -1]
    theta_ram = np.random.choice(theta)
    s_ram = np.random.choice(s)
    h = s_ram*(x - theta_ram)
    hypothesis = np.sign(h)
    wrong = (hypothesis * y).tolist()
    count = len([x for x in wrong if x < 0])
    Ein = count / 8
    EinList.append(Ein)
    thetaList.append(theta_ram)
    Eout = 0.5 - 0.4 * s_ram + 0.4 * s_ram * abs(theta_ram)
    EoutList.append(Eout)
    epsilon = Eout - Ein
    epsilonList.append(epsilon)
# 計算中位數
median = np.median(np.array(epsilonList))
# 繪圖
plt.scatter(EinList, EoutList, alpha=0.5)
plt.xlabel("Ein(g)")
plt.ylabel("Eout(g)")
plt.title("scatter plot")
plt.show()
print(median)
