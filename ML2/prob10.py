import numpy as np
import matplotlib.pyplot as plt
# 存2000個 Ein Eout theta epsilon 的 list
EinList = []
EoutList = []
thetaList = []
epsilonList = []
for time in range(2000):
    # 在1到-1中隨機選32個值 並排序
    x_random = np.random.uniform(-1, 1, 32)
    x = np.sort(x_random)
    # 創建theta
    theta = np.mean([x[i:i + 2] for i in range(31)], axis=1)
    theta = np.insert(theta, 0, -1)
    # 定義noise及y
    noise = np.random.choice([1, -1], 32, p=[0.9, 0.1])
    y = np.sign(x) * noise
    # 定義s
    s = [1, -1]
    # Ein 最小值
    Ein_min = 1000.0
    # Ein 有最小值時的 theta
    Emin_theta = 0
    # 目前的s
    s_now = 0
    s_right = 0
    for i in theta:
        # 定義當 s = +-1 時候的 hypothesis
        h_plus = (x - i)
        h_minus = -1 * (x - i)
        hypothesis_plus = np.sign(h_plus)
        hypothesis_minus = np.sign(h_minus)
        # 判斷 y 是否跟 h 的結果一樣
        wrong_plus = (hypothesis_plus * y).tolist()
        wrong_minus = (hypothesis_minus * y).tolist()
        # 計算當中有幾個錯誤的
        count_plus = len([x for x in wrong_plus if x < 0])
        count_minus = len([x for x in wrong_minus if x < 0])
        # 計算機率
        Ein_plus = count_plus / 32
        Ein_minus = count_minus / 32
        # 比較 Ein 在 s=+-1 時的大小
        if Ein_plus < Ein_minus:
            Ein = Ein_plus
            s_now = 0
        else:
            Ein = Ein_minus
            s_now = 1
        # 找出 Ein 最小值及此時的 theta 並記錄下來
        if Ein < Ein_min:
            Ein_min = Ein
            Emin_theta = i
            s_right = s_now  # 判斷s是1或-1
        elif Ein == Ein_min:
            if s[s_now] * i < s[s_right] * Emin_theta:
                Ein = s[s_now] * i
                Emin_theta = i
                s_right = s_now  # 判斷s是1或-1
            else:
                Ein = s[s_right]*Emin_theta
    # 將記錄到的 Ein_min 及 theta 加入列表
    EinList.append(Ein_min)
    thetaList.append(Emin_theta)
    # 計算Eout
    Eout = 0.5 - 0.4 * s[s_right] + 0.4 * s[s_right] * abs(Emin_theta)
    EoutList.append(Eout)
    epsilon = Eout - Ein_min
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
