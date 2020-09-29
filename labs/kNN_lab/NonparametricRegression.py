import numpy as np
import pandas as pd
import random
import math

import matplotlib.pyplot as plt
from matplotlib import style

style.use("fivethirtyeight")

# metrics

def manhattan_distance(x, y):
    return sum([abs(x[i] - y[i]) for i in range(len(x))])


def euclidean_distance(x, y):
    return math.sqrt(sum([(x[i] - y[i]) ** 2 for i in range(len(x))]))


def chebyshev_distance(x, y):
    return max([abs(x[i] - y[i]) for i in range(len(x))])


# kernels

def uniform_kernel(x):
    if abs(x) >= 1:
        return 0
    else:
        return 1/2


def triangular_kernel(x):
    return max(0, 1 - abs(x))


def epanechnikov_kernel(x):
    return max(0, 3/4 * (1 - x ** 2))


def quartic_kernel(x):
    return max(0, 15/16 * (1 - x ** 2) ** 2)


def triweight_kernel(x):
    return max(0, 35/32 * (1 - x ** 2) ** 3)


def tricube_kernel(x):
    return max(0, 70/81 * (1 - abs(x) ** 3) ** 3)


def gaussian_kernel(x):
    return 1 / math.sqrt(2 * math.pi) * (math.exp(-1/2 * x ** 2))


def cosine_kernel(x):
    if abs(x) >= 1:
        return 0
    else:
        return math.pi / 4 * math.cos(math.pi/2 * x)


def logistic_kernel(x):
    return 1/(math.exp(x) + 2 + math.exp(-x))


def sigmoid_kernel(x):
    return 2/(math.pi * (math.exp(x) + math.exp(-x)))


distances = {'manhattan': manhattan_distance,
             'euclidean': euclidean_distance,
             'chebyshev': chebyshev_distance}

kernels = {'uniform': uniform_kernel,
           'triangular': triangular_kernel,
           'epanechnikov': epanechnikov_kernel,
           'quartic': quartic_kernel,
           'triweight': triweight_kernel,
           'tricube': tricube_kernel,
           'gaussian': gaussian_kernel,
           'cosine': cosine_kernel,
           'logistic': logistic_kernel,
           'sigmoid': sigmoid_kernel}


def f_macro(cm):
    k = len(cm)
    tp = [0] * k
    tn = [0] * k
    fp = [0] * k
    fn = [0] * k
    prc = [0] * k
    rc = [0] * k
    for i in range(k):
        for j in range(k):
            if i == j:
                tp[i] += cm[i][j]
                for jj in range (k - 1):
                    tn[(jj + i + 1) % k] += cm[i][j]
            else:
                fp[j] += cm[i][j]
                fn[i] += cm[i][j]
    for i in range(k):
        if (tp[i] + fp[i]) != 0:
            prc[i] = tp[i] / (tp[i] + fp[i])
        if (tp[i] + fn[i]) != 0:
            rc[i] = tp[i] / (tp[i] + fn[i])

    prcw = 0
    rcw = 0
    sum = 0
    for i in range(k):
        if (tp[i] + fp[i]) != 0:
            prcw += tp[i] * (tp[i] + fn[i]) / (tp[i] + fp[i])
        rcw += tp[i]
        sum += tp[i] + fn[i]

    if sum != 0:
        prcw = prcw / sum
        rcw = rcw / sum

    f = [0] * k
    for i in range(k):
        if prc[i] + rc[i] != 0:
            f[i] = (2 * prc[i] * rc[i]) / (prc[i] + rc[i])

    # fmicro = 0
    # for i in range(k):
    #     fmicro += (tp[i] + fn[i]) * f[i]
    # if sum != 0:
    #     fmicro /= sum
    if prcw + rcw != 0:
        fmacro = 2 * prcw * rcw / (prcw + rcw)
    else:
        fmacro = 0

    return fmacro


def knn_loo(data, test, distance, kernel, h, num_classes, fixed):
    votes = [0] * num_classes
    wsize = h
    if not fixed:
        dsts = sorted([distances[distance](test[:-1], d[:-1]) for d in data])
        wsize = dsts[h + 1]
    for d in data:
        votes[int(d[-1])] += kernels[kernel](distances[distance](d[:-1], test[:-1]) / wsize)
    return np.argmax(votes)


def main():

    df = pd.read_csv('file3e9021e6eda.csv')

    print(df)
    num_classes = df['Species'].nunique()
    df.loc[df['Species'] == 'setosa', 'Species'] = 0
    df.loc[df['Species'] == 'versicolor', 'Species'] = 1
    df.loc[df['Species'] == 'virginica', 'Species'] = 2

    print(df)

    # fill empty as outliers
    df.replace("?", -99999, inplace=True)

    # normalize
    df = (df-df.min())/(df.max()-df.min())
    df['Species'] *= (num_classes - 1)
    full_data = df.astype(float).values.tolist()
    print(full_data)

    random.shuffle(full_data)
    print(full_data)


    # fixed window:

    f_list_best = []
    f_best = 0
    d_k_best = ['', '']

    for distance in distances.keys():
        for kernel in kernels.keys():
            f_list = []
            flag = False
            for ws in range(5, 70, 5):
                window_size = ws / 100
                print(distance, kernel, window_size)

                data = full_data

                score = 0
                cm = np.empty(shape=(num_classes, num_classes))
                cm.fill(0)

                for j in range(len(data)):
                    val = data[j]
                    train = data[:j]
                    if j < len(data) - 1:
                        train += data[(j + 1):]
                    ans = knn_loo(train, val, distance, kernel, window_size, num_classes, fixed=True)
                    cm[int(val[-1])][ans] += 1
                    if ans == val[-1]:
                        score += 1

                f_cur = f_macro(cm)
                print('precision: ', score / len(data))
                print('f-measure: ', f_cur)
                print(cm)
                f_list.append([window_size, f_cur])
                if f_cur > f_best:
                    f_best = f_cur
                    d_k_best = [distance, kernel]
                    flag = True
            if flag:
                f_list_best = f_list.copy()

    print('best')
    print(f_best)
    print(d_k_best)

    print(f_list_best)

    plt.plot([f_list_best[i][0] for i in range(len(f_list_best))], [f_list_best[i][1] for i in range(len(f_list_best))])
    plt.xlabel('window_size')
    plt.ylabel('f-measure')
    plt.suptitle(d_k_best[0] + ' ' + d_k_best[1] + ' fixed')
    plt.savefig('fixed.png')
    plt.show()



    # variable window

    f_list_best = []
    f_best = 0
    d_k_best = ['', '']

    for distance in distances.keys():
        for kernel in kernels.keys():
            f_list = []
            flag = False
            for ws in range(1, 25, 1):
                window_size = ws
                print(distance, kernel, window_size)

                data = full_data

                score = 0
                cm = np.empty(shape=(num_classes, num_classes))
                cm.fill(0)

                for j in range(len(data)):
                    val = data[j]
                    train = data[:j]
                    if j < len(data) - 1:
                        train += data[(j + 1):]
                    ans = knn_loo(train, val, distance, kernel, window_size, num_classes, fixed=False)
                    cm[int(val[-1])][ans] += 1
                    if ans == val[-1]:
                        score += 1

                f_cur = f_macro(cm)
                print('precision: ', score / len(data))
                print('f-measure: ', f_cur)
                print(cm)
                f_list.append([window_size, f_cur])
                if f_cur > f_best:
                    f_best = f_cur
                    d_k_best = [distance, kernel]
                    flag = True
            if flag:
                f_list_best = f_list.copy()

    print('best')
    print(f_best)
    print(d_k_best)

    print(f_list_best)

    plt.plot([f_list_best[i][0] for i in range(len(f_list_best))], [f_list_best[i][1] for i in range(len(f_list_best))])
    plt.xlabel('window_size')
    plt.ylabel('f-measure')
    plt.suptitle(d_k_best[0] + ' ' + d_k_best[1] + ' variable')
    plt.savefig('variable.png')
    plt.show()


if __name__ == '__main__':
    main()
