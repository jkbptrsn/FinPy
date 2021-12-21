import numpy as np
from sklearn.svm import SVC


def f_norm(x_train, x):
    # "Normalize" each feature of x based on training data x_train
    x_norm = x.copy()
    for n in range(len(x_train[0, :])):
        x_norm[:, n] = \
            (x_norm[:, n] - x_train[:, n].mean()) / x_train[:, n].std()
    return x_norm


def f_loss01(y, y_true):
    # 0-1 loss
    return sum(np.abs(y + y_true) < 1) / len(y)


# Data sets
y_train = np.loadtxt('y_train1-1.csv', delimiter=',')
y_test = np.loadtxt('y_test1-1.csv', delimiter=',')
x_train = np.loadtxt('X_train_binary.csv', delimiter=',')
x_test = np.loadtxt('X_test_binary.csv', delimiter=',')

# Class frequencies of training set
f_plus = sum(y_train == 1) / len(y_train)
f_minus = sum(y_train == -1) / len(y_train)
print(f_plus, f_minus)

# Normalizing data sets
x_test = f_norm(x_train, x_test)
x_train = f_norm(x_train, x_train)
for n in range(len(x_test[0, :])):
    print(str(n) + ' & ' + str(round(x_test[:, n].mean(), 3))
          + ' & ' + str(round(x_test[:, n].std() ** 2, 3)) + ' \\\\')

# Regularization parameter
c_values = [0.01, 0.1, 1, 10, 100, 1000, 10000]
# Kernel parameter
gamma_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

loss01_min = 1
for c in c_values:
    for g in gamma_values:
        clf = SVC(C=c, gamma=g)
        # 5-fold cross validation
        n_rows = int(len(x_train[:, 0]) / 5)
        loss01 = 0.0
        for n in range(5):
            # Training
            slicing = np.r_[:n * n_rows, (n + 1) * n_rows:len(x_train)]
            x_train_temp = x_train[slicing]
            y_train_temp = y_train[slicing]
            clf.fit(x_train_temp, y_train_temp)
            # Testing
            x_test_temp = x_train[n * n_rows: (n + 1) * n_rows]
            y_test_temp = y_train[n * n_rows: (n + 1) * n_rows]
            y_out = clf.predict(x_test_temp)
            # 0-1 loss
            loss01 += f_loss01(y_out, y_test_temp)
        loss01 /= 5
        if loss01 < loss01_min:
            loss01_min = loss01
            print(c, g, loss01)

clf = SVC(C=0.001, gamma=10)
# Loss on training data set
clf.fit(x_train, y_train)
print(f_loss01(clf.predict(x_train), y_train))
# Loss on test data set
print(f_loss01(clf.predict(x_test), y_test))

# The product y_i * f(x_i) for all support vectors
yifxi = y_train[clf.support_] * clf.decision_function(x_train[clf.support_, :])
print(yifxi)
print("Number of free SVs: ", sum(np.abs(yifxi - 1) < 0.001))
print("Number of bounded SVs: ", sum(np.abs(yifxi - 1) >= 0.001))

print(clf.support_)
# print(clf.support_vectors_[0, :].size)
# print(clf.support_vectors_[0, :])
print(clf.n_support_)
