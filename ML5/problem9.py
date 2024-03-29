from libsvm.svmutil import *

y_train, x_train = svm_read_problem("train.txt")
y_data = [1 if y == 4 else -1 for y in y_train]
y_test, x_test = svm_read_problem("test.txt")

C = [0.1, 1, 10]
Q = [2, 3, 4]
min_support_vectors = float('inf')
best_C = None
best_Q = None

for c in C:
    for q in Q:
        model = svm_train(y_data, x_train, f"-s 0 -t 1 -d {q} -g 1 -r 1 -c {c} -q")
        num = model.get_nr_sv()
        print(f'(C={c}, Q={q}): Number of support vectors = {num}')
        if num < min_support_vectors:
            min_support_vectors = num
            best_C = c
            best_Q = q

print(f'\nBest combination: (C={best_C}, Q={best_Q}), Number of support vectors = {min_support_vectors}')

