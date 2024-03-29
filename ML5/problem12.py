from libsvm.svmutil import *
import matplotlib.pyplot as plt

C = [0.01, 0.1, 1, 10, 100]
C_list = []
norm_w_list = []

y_train, x_train = svm_read_problem("train.txt")
y_data = [1 if y == 3 else -1 for y in y_train]
for c in C:
    model = svm_train(y_data, x_train, f"-s 0 -t 2 -g 1 -c {c} -q")
    sv_indices = model.get_sv_indices()
    coefficients = model.get_sv_coef()
    w = [0] * 36
    for i in range(len(sv_indices)):
        # 是SV的 index
        sv_index = sv_indices[i]
        coef = coefficients[i][0]
        sv = x_train[sv_index - 1]  # LibSVM indices start from 1
        # Update the weight vector w
        for j in range(1, 37):
            if j in sv:
                w[j-1] += coef * sv[j]

    norm_w = sum(v ** 2 for v in w) ** 0.5
    # Store C and ∥w∥ values
    C_list.append(c)
    norm_w_list.append(norm_w)

plt.plot(C_list, norm_w_list, marker='o', linestyle='-', color='b')
plt.xscale('log')
plt.xlabel('C Values')
plt.ylabel('Norm of the Weight Vector (∥w∥)')
plt.title('C versus ∥w∥ with Gaussian Kernel')
plt.show()


