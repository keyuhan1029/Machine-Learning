import numpy as np
import matplotlib.pyplot as plt

mean_p = [3, 2]
cov_p = [[0.4, 0], [0, 0.4]]
mean_n = [5, 0]
cov_n = [[0.6, 0], [0, 0.6]]


def generate_data(i, meanp, covp, meann, covn):
    xd = []
    yd = []
    np.random.seed(i)
    for n in range(256):
        y = np.random.choice([1, -1])
        if y == 1:
            x = np.hstack([1, np.random.multivariate_normal(meanp, covp)])
        else:
            x = np.hstack([1, np.random.multivariate_normal(meann, covn)])
        xd.append(x)
        yd.append(y)
    return xd, yd


def linear_regression(x, y):
    inverse = np.linalg.inv(x.T @ x)
    w = inverse @ x.T @ y
    return w


error = []
for i in range(128):
    x_train, y_train = generate_data(i, mean_p, cov_p, mean_n, cov_n)
    xtrain = np.array(x_train)
    ytrain = np.array(y_train)
    w_lin = linear_regression(xtrain, ytrain)
    err = np.mean((xtrain @ w_lin - ytrain) ** 2)
    error.append(err)

plt.hist(error, bins=50)
plt.xlabel("error")
plt.ylabel("times")
plt.title("Distribution of error")
plt.show()

median = np.median(error)
print(median)
