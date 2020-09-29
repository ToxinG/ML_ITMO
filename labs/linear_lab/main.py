import math
import time
import numpy as np
import matplotlib.pyplot as plt

def gauss(m):
    # eliminate columns
    for col in range(len(m[0])):
        for row in range(col+1, len(m)):
            r = [(rowValue * (-(m[row][col] / m[col][col]))) for rowValue in m[col]]
            m[row] = [sum(pair) for pair in zip(m[row], r)]
    # now backsolve by substitution
    ans = []
    m.reverse() # makes it easier to backsolve
    for sol in range(len(m)):
            if sol == 0:
                ans.append(m[sol][-1] / m[sol][-2])
            else:
                inner = 0
                # substitute in all known coefficients
                for x in range(sol):
                    inner += (ans[x]*m[sol][-2-x])
                # the equation is now reduced to ax + b = c form
                # solve with (c - b) / a
                ans.append((m[sol][-1]-inner)/m[sol][-sol-2])
    ans.reverse()
    return ans


def main():
    learning_rate = 0.001
    batch_size = 64
    tau = 0.1

    X_train = []
    X_test = []

    with open('data/1.txt', 'r') as f:
        num_features = int(f.readline().rstrip())
        num_train = int(f.readline().rstrip())

        for i in range(num_train):
            item = [int(xki) for xki in f.readline().split()]
            X_train.append(item)

        num_test = int(f.readline().rstrip())

        for i in range(num_test):
            item = [int(xki) for xki in f.readline().split()]
            X_test.append(item)

    # weights = [1] * (num_features + 1)
    try:
        weights = gauss([xi[:-1] for xi in X_train[:num_features]])
        weights.append(X_train[0][-1] - sum(X_train[0][i] * weights[i] for i in range(num_features)))
    except Exception:
        # print('N * N Gauss failed')
        weights = [1] * (num_features + 1)

    time_start = time.time()

    nrmse_train = []
    nrmse_test = []

    time1 = time.time()
    time2 = time.time()
    epoch_duration = time2 - time1

    # gradient descent

    while time2 - time_start + epoch_duration < 10:
        for b in range((num_train - 1) // batch_size + 1):
            cur_batch_size = min(batch_size, num_train - b * batch_size)
            sum_wx = [weights[-1]] * cur_batch_size
            sum_wxx = [weights[-1]] * cur_batch_size
            y_diff = [0] * num_train
            numerator = 0
            denominator = 0

            for i in range(cur_batch_size):
                wx = []
                wxx = []
                for j in range(num_features):
                    wx.append(weights[j] * X_train[b * batch_size + i][j])
                    wxx.append(wx[j] * X_train[b * batch_size + i][j])

                    sum_wx[i] += wx[j]
                    sum_wxx[i] += wxx[j]

                y_diff[b * batch_size + i] = X_train[b * batch_size + i][-1] - sum_wx[i]

                numerator += y_diff[b * batch_size + i] * y_diff[b * batch_size + i] * sum_wxx[i]
                denominator += 2 * y_diff[b * batch_size + i] * y_diff[b * batch_size + i] * sum_wxx[i] * sum_wxx[i]

            learning_rate = numerator / (cur_batch_size * denominator)

            for i in range(num_features):
                for j in range(cur_batch_size):
                    weights[i] = weights[i] * (1 - learning_rate * tau) + 2 * X_train[b * batch_size + j][i] * y_diff[
                        b * batch_size + j] * learning_rate

            for j in range(cur_batch_size):
                weights[-1] = weights[-1] * (1 - learning_rate * tau) + 2 * y_diff[b * batch_size + j] * learning_rate

        print(learning_rate)

        nrmse_train.append(math.sqrt(sum(y_diff[i] * y_diff[i] for i in range(num_train)) / num_train))
        nrmse_test.append(math.sqrt(sum(math.pow(X_test[i][-1] - (sum(weights[j] * X_test[i][j] for j in range(num_features)) + weights[-1]), 2) for i in range(num_test)) / num_test))

        time1 = time2
        time2 = time.time()
        epoch_duration = time2 - time1

    print(nrmse_train)
    print(nrmse_test)
    plt.plot(nrmse_train, label='train')
    plt.plot(nrmse_test, label='test')
    plt.legend(loc='upper right')
    plt.title('NRMSE on epochs')
    plt.savefig('nrmse.png')
    plt.show()

    # pseudoinverse matrix
    
    X = np.array([xi[:-1] for xi in X_train])
    y = [xi[-1] for xi in X_train]

    n = X.shape[1]
    r = np.linalg.matrix_rank(X)

    U, sigma, VT = np.linalg.svd(X, full_matrices=False)

    D_plus = np.diag(np.hstack([1 / sigma[:r], np.zeros(n - r)]))

    V = VT.T

    X_plus = V.dot(D_plus).dot(U.T)

    w = X_plus.dot(y)

    error_train = math.sqrt(sum(math.pow(X_train[i][-1] - (sum(w[j] * X_train[i][j] for j in range(num_features)) + w[-1]), 2) for i in range(num_train)) / num_train)
    print(error_train)
    error_test = math.sqrt(sum(math.pow(X_test[i][-1] - (sum(w[j] * X_test[i][j] for j in range(num_features)) + w[-1]), 2) for i in range(num_test)) / num_test)
    print(error_test)


if __name__ == '__main__':
    main()

# for b in range((num_train - 1) // batch_size + 1):
#     cur_batch_size = min(batch_size, num_train - b * batch_size)
#     sum_wx = [weights[-1]] * cur_batch_size
#     sum_wxx = [weights[-1]] * cur_batch_size
#     y_diff = [0] * num_train
#     numerator = 0
#     denominator = 0
#
#     for i in range(cur_batch_size):
#         wx = []
#         wxx = []
#         for j in range(num_features):
#             wx.append(weights[j] * X_train[b * batch_size + i][j])
#             wxx.append(wx[j] * X_train[b * batch_size + i][j])
#
#             sum_wx[i] += wx[j]
#             sum_wxx[i] += wxx[j]
#
#         y_diff[b * batch_size + i] = X_train[b * batch_size + i][-1] - sum_wx[i]
#
#         numerator += y_diff[b * batch_size + i] * y_diff[b * batch_size + i] * sum_wxx[i]
#         denominator += 2 * y_diff[b * batch_size + i] * y_diff[b * batch_size + i] * sum_wxx[i] * sum_wxx[i]
#
#     learning_rate = 2 * numerator / (cur_batch_size * denominator)
#
#     for i in range(num_features):
#         for j in range(cur_batch_size):
#             weights[i] = weights[i] * (1 - learning_rate * tau) + 2 * X_train[b * batch_size + j][i] * y_diff[
#                 b * batch_size + j] * learning_rate
#
#     for j in range(cur_batch_size):
#         weights[-1] = weights[-1] * (1 - learning_rate * tau) + 2 * y_diff[b * batch_size + j] * learning_rate