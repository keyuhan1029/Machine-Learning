import numpy as np
import matplotlib.pyplot as plt

mean_p = [3, 2]
cov_p = [[0.4, 0], [0, 0.4]]
mean_n = [5, 0]
cov_n = [[0.6, 0], [0, 0.6]]
mean_o = [0, 6]
cov_o = [[0.1, 0], [0, 0.3]]
N_train = 256
N_test = 4096


def generate_testing_data(i, meanp, covp, meann, covn, N):
    xd = []
    yd = []
    np.random.seed(i)
    for n in range(N):
        y = np.random.choice([1, -1])
        if y == 1:
            x = np.hstack([1, np.random.multivariate_normal(meanp, covp)])
        else:
            x = np.hstack([1, np.random.multivariate_normal(meann, covn)])
        xd.append(x)
        yd.append(y)
    xd = np.array(xd)
    yd = np.array(yd)
    return xd, yd


def generate_training_data(i, meanp, covp, meann, covn, N):
    xd = []
    yd = []
    np.random.seed(i)
    for n in range(N):
        y = np.random.choice([1, -1])
        if y == 1:
            x = np.hstack([1, np.random.multivariate_normal(meanp, covp)])
        else:
            x = np.hstack([1, np.random.multivariate_normal(meann, covn)])
        xd.append(x)
        yd.append(y)
    for time in range(16):
        y = 1
        x = np.hstack([1, np.random.multivariate_normal(mean_o, cov_o)])
        xd.append(x)
        yd.append(y)
    xd = np.array(xd)
    yd = np.array(yd)
    return xd, yd


def linear_regression(x, y):
    inverse = np.linalg.inv(x.T @ x)
    w = inverse @ x.T @ y
    return w


def logistic_regression(x, y, learning_rate):
    w = np.zeros(3)
    for i in range(500):
        E_grad = np.zeros(3)
        for j in range(N_train):
            wx = x[j] @ w.T
            sig = 1 / (1 + np.exp(wx * y[j]))
            ele = sig * (-x[j] * y[j])/len(y)
            E_grad += ele
        w -= learning_rate * E_grad
    return w


def zero_one_error(wx, y):
    mistake = np.where(wx != y)[0]
    error = len(mistake) / len(y)
    return error


err_lin = []
err_log = []
for i in range(128):
    x_train, y_train = generate_training_data(i, mean_p, cov_p, mean_n, cov_n, N_train)
    x_test, y_test = generate_testing_data(i, mean_p, cov_p, mean_n, cov_n, N_test)

    w_log = logistic_regression(x_train, y_train, 0.1)
    w_lin = linear_regression(x_train, y_train)

    wx_A = np.sign(x_test @ w_lin)
    wx_B = np.sign(x_test @ w_log)

    err_A = zero_one_error(wx_A, y_test)
    err_B = zero_one_error(wx_B, y_test)

    err_lin.append(err_A)
    err_log.append(err_B)

median_lin = np.median(err_lin)
median_log = np.median(err_log)
print(median_lin, median_log)

plt.xlabel('Eout_lin')
plt.ylabel('Eout_log')
plt.title('Simple Scatter Plot')
plt.scatter(err_lin, err_log, marker='o', s=20, alpha=0.5)
plt.xlabel('(E0/1(wlin))')
plt.ylabel('(E0/1(wlog))')
plt.title('Elin v.s Elog')
plt.show()