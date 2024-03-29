from libsvm.svmutil import *

y_train, x_train = svm_read_problem("train.txt")
y_data = [1 if y == 1 else -1 for y in y_train]
y_test, x_test = svm_read_problem("test.txt")
y_data_test = [1 if y == 1 else -1 for y in y_test]
C = [0.01, 0.1, 1, 10, 100]

min_classification_error = float('inf')
best_C = None

for c in C:
    model = svm_train(y_data, x_train, f"-s 0 -t 2 -g 1 -c {c} -q")
    p_label, p_acc, p_val = svm_predict(y_data_test, x_test, model, )
    err = 100-p_acc[0]
    print(f'C={c}: Eout = {err}%')
    if err < min_classification_error:
        min_classification_error = err
        best_C = c

# Print the best value of C
print(f'\nBest value of C: {best_C}, Minimum Eout = {min_classification_error}%')




