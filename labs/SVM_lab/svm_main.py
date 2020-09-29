import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import random


class SmoAlgorithm:
    def __init__(self, X, y, kernel_func, c, tol=0.001):

        self.X = X
        self.y = y
        self.m, self.n = np.shape(self.X)

        self.kernel = kernel_func
        self.C = c
        self.k = kernel_matrix(self.kernel, X, X)

        self.alphas = np.zeros(self.m)
        self.b = 0
        self.w = np.zeros(self.n)

        self.errors = np.zeros(self.m)
        self.eps = 1e-3
        self.tol = tol

        self.MAX_ITER = 3000

    def predict(self, x):
        res = int(
            np.sign(np.sum([self.alphas[j] * self.y[j] * self.kernel(self.X[j], x) for j in range(self.m)]) - self.b))
        return res if res != 0 else 1

    def output(self, i):
        return sum([self.alphas[j] * self.y[j] * self.k[j][i] for j in range(self.m)]) - self.b

    def take_step(self, i1, i2):
        if i1 == i2:
            return False

        a1 = self.alphas[i1]
        y1 = self.y[i1]
        X1 = self.X[i1]
        e1 = self.get_error(i1)

        s = y1 * self.y2

        if y1 != self.y2:
            L = max(0, self.a2 - a1)
            H = min(self.C, self.C + self.a2 - a1)
        else:
            L = max(0, self.a2 + a1 - self.C)
            H = min(self.C, self.a2 + a1)

        if L == H:
            return False

        k11 = self.k[i1][i1]
        k12 = self.k[i1][i2]
        k22 = self.k[i2][i2]

        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2_new = self.a2 + self.y2 * (e1 - self.e2) / eta

            if a2_new < L:
                a2_new = L
            elif a2_new > H:
                a2_new = H
        else:
            f1 = y1 * (e1 + self.b) - a1 * k11 - s * self.a2 * k12
            f2 = self.y2 * (self.e2 + self.b) - s * a1 * k12 - self.a2 * k22
            L1 = self.a2 + s * (self.a2 - L)
            H1 = a1 + s * (self.a2 - H)
            Lobj = L1 * f1 + L * f2 + 0.5 * (L1 ** 2) * k11 + 0.5 * (L ** 2) * k22 + s * L * L1 * k12
            Hobj = H1 * f1 + H * f2 + 0.5 * (H1 ** 2) * k11 + 0.5 * (H ** 2) * k22 + s * H * H1 * k12

            if Lobj < Hobj - self.eps:
                a2_new = L
            elif Lobj > Hobj + self.eps:
                a2_new = H
            else:
                a2_new = self.a2

        if abs(a2_new - self.a2) < self.eps * (a2_new + self.a2 + self.eps):
            return False

        a1_new = a1 + s * (self.a2 - a2_new)

        new_b = self.compute_b(e1, a1, a1_new, a2_new, k11, k12, k22, y1)

        delta_b = new_b - self.b

        self.b = new_b

        delta1 = y1 * (a1_new - a1)
        delta2 = self.y2 * (a2_new - self.a2)

        for i in range(self.m):
            if 0 < self.alphas[i] < self.C:
                self.errors[i] += delta1 * self.k[i1][i] + delta2 * self.k[i2][i] - delta_b

        self.errors[i1] = 0
        self.errors[i2] = 0

        self.alphas[i1] = a1_new
        self.alphas[i2] = a2_new

        return True

    def compute_b(self, e1, a1, a1_new, a2_new, k11, k12, k22, y1):
        b1 = e1 + y1 * (a1_new - a1) * k11 + self.y2 * (a2_new - self.a2) * k12 + self.b

        b2 = self.e2 + y1 * (a1_new - a1) * k12 + self.y2 * (a2_new - self.a2) * k22 + self.b

        if (0 < a1_new) and (self.C > a1_new):
            new_b = b1
        elif (0 < a2_new) and (self.C > a2_new):
            new_b = b2
        else:
            new_b = (b1 + b2) / 2
        return new_b

    def get_error(self, i1):
        if 0 < self.alphas[i1] < self.C:
            return self.errors[i1]
        else:
            return self.output(i1) - self.y[i1]

    def second_heuristic(self, non_bound_indices):
        i1 = -1
        if len(non_bound_indices) > 1:
            max = 0

            for j in non_bound_indices:
                e1 = self.errors[j] - self.y[j]
                step = abs(e1 - self.e2)
                if step > max:
                    max = step
                    i1 = j
        return i1

    def examine_example(self, i2):
        self.y2 = self.y[i2]
        self.a2 = self.alphas[i2]
        self.X2 = self.X[i2]
        self.e2 = self.get_error(i2)

        r2 = self.e2 * self.y2

        if not ((r2 < -self.tol and self.a2 < self.C) or (r2 > self.tol and self.a2 > 0)):
            # The KKT conditions are met, so SMO takes another example
            return 0

        # Choose the Lagrange multiplier which maximizes the absolute error.
        non_bound_idx = list(self.get_non_bound_indices())
        i1 = self.second_heuristic(non_bound_idx)

        if i1 >= 0 and self.take_step(i1, i2):
            return 1

        # Look for examples making positive progress by looping over all non-zero and non-C lambda
        # (or alpha...) starting at a random point.
        rand_i = random.randrange(self.m)
        all_indices = list(range(self.m))
        for i1 in all_indices[rand_i:] + all_indices[:rand_i]:
            if self.take_step(i1, i2):
                return 1

        # Extremely degenerate circumstances, SMO skips the first example.
        return 0

    def error(self, i2):
        return self.output(i2) - self.y2

    def get_non_bound_indices(self):
        return np.where(np.logical_and(self.alphas > 0, self.alphas < self.C))[0]

    def first_heuristic(self):
        num_changed = 0
        non_bound_idx = self.get_non_bound_indices()
        for i in non_bound_idx:
            num_changed += self.examine_example(i)
        return num_changed

    def main_routine(self):
        num_changed = 0
        examine_all = True
        iter = 0
        while iter < self.MAX_ITER and (num_changed > 0 or examine_all):
            iter += 1
            num_changed = 0

            if examine_all:
                for i in range(self.m):
                    num_changed += self.examine_example(i)
                else:
                    num_changed += self.first_heuristic()

                if examine_all:
                    examine_all = False
                elif num_changed == 0:
                    examine_all = True


def kernel_matrix(kernel_func, A, B):
    n, *_ = A.shape
    m, *_ = B.shape
    f = lambda i, j: kernel_func(A[i], B[j])
    return np.fromfunction(np.vectorize(f), (n, m), dtype=int)


def kernel_linear(a, b):
    return np.dot(a, b)


def kernel_sigmoid(a, b, gamma=1, k0=0):
    return np.tanh(gamma * np.dot(a, b) + k0)


def kernel_gaussian(a, b, gamma=0.5):
    return np.exp(-gamma * np.power(np.linalg.norm(a - b), 2))


def kernel_poly(a, b, gamma=1, k0=1, degree=3):
    return np.power(gamma * np.dot(a, b) + k0, degree)


def build_kernel_function(name, gamma=1, k0=1, degree=3):
    if name == 'linear':
        return lambda a, b: kernel_linear(a, b)
    elif name == 'sigmoid':
        return lambda a, b: kernel_sigmoid(a, b, gamma, k0)
    elif name == 'gaussian':
        return lambda a, b: kernel_gaussian(a, b, gamma)
    elif name == 'poly':
        return lambda a, b: kernel_poly(a, b, gamma, k0, degree)


C_vals = [0.01, 0.1, 1, 10, 50]

linear_params = np.array([{'name': 'linear', 'C': C} for C in C_vals])
sigmoid_params = np.array([{'name': 'sigmoid', 'gamma': gamma, 'k0': k0, 'C': C}
                           for gamma in [0.5, 0.75, 1.0]
                           for k0 in [0.0, 0.5, 1.0]
                           for C in C_vals])
gaussian_params = np.array([{'name': 'gaussian', 'gamma': gamma, 'C': C}
                            for gamma in [0.1, 0.3, 0.5]
                            for C in C_vals])
poly_params = np.array([{'name': 'poly', 'gamma': gamma, 'k0': k0, 'C': C, 'degree': degree}
                        for gamma in [0.01, 0.5, 1.0]
                        for k0 in [0, 0.5, 1]
                        for degree in [3, 4, 5]
                        for C in C_vals])
all_params = [linear_params, sigmoid_params, gaussian_params, poly_params]
# all_params = np.array(linear_params + sigmoid_params + gaussian_params + poly_params)


def build_cls(params, X, y):
    name = params['name']
    C = params['C']
    gamma = params.get('gamma', None)
    k0 = params.get('k0', None)
    degree = params.get('degree', None)
    kernel_func = build_kernel_function(name, gamma, k0, degree)
    cls = SmoAlgorithm(X, y, kernel_func, C)
    cls.main_routine()
    return cls


def eval_score(params, X, y):
    kf = KFold(n_splits=5)
    f_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        cls = build_cls(params, X_train, y_train)
        y_pred = np.apply_along_axis(lambda x: cls.predict(x), 1, X_test)
        f_scores.append(f1_score(y_test, y_pred))
    return np.average(np.array(f_scores))


def find_best_params(X, y):
    res = []
    for k_p in all_params:
        scores = []
        for p in k_p:
            print(p)
            score = eval_score(p, X, y)
            print(score)
            scores.append(score)
        res.append(k_p[np.argmax(np.array(scores))])
        res[-1]['score'] = max(scores)

    return res


def draw(name, p, clf, X, y, step):
    stepx = step
    stepy = step
    x_min, y_min = np.amin(X, 0)
    x_max, y_max = np.amax(X, 0)
    x_min -= stepx
    x_max += stepx
    y_min -= stepy
    y_max += stepy
    xx, yy = np.meshgrid(np.arange(x_min, x_max, stepx),
                         np.arange(y_min, y_max, stepy))

    mesh_dots = np.c_[xx.ravel(), yy.ravel()]
    zz = np.apply_along_axis(lambda t: clf.predict(t), 1, mesh_dots)
    zz = np.array(zz).reshape(xx.shape)

    plt.figure(figsize=(10, 10))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    x0, y0 = X[y == -1].T
    x1, y1 = X[y == 1].T

    plt.pcolormesh(xx, yy, zz, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(x0, y0, color='red', s=100)
    plt.scatter(x1, y1, color='blue', s=100)

    sup_ind = clf.get_non_bound_indices()
    X_sup = X[sup_ind]
    x_sup, y_sup = X_sup.T

    plt.scatter(x_sup, y_sup, color='white', marker='x', s=60)
    plt.suptitle(p)
    plt.savefig(name + '_' + p['name'] + '.png')
    plt.show()


def process_dataset(filename):
    df = pd.read_csv(filename + '.csv')
    X = df.values[:, :-1]
    y = df.values[:, -1]
    y = np.vectorize(lambda y_i: 1 if y_i == 'P' else -1)(y)
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    params = find_best_params(X, y)
    for p in params:
        print(filename + ' dataset, ' + p['name'] + ' kernel:')
        print(p)
        draw(filename, p, build_cls(p, X, y), X, y, 0.05)


if __name__ == '__main__':
    for f in ['chips', 'geyser']:
        process_dataset(f)
