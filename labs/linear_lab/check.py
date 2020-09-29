from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

X, y, coefficients = make_regression(
    n_samples=50,
    n_features=1,
    n_informative=1,
    n_targets=1,
    noise=5,
    coef=True,
    random_state=1
)

n = X.shape[1]
r = np.linalg.matrix_rank(X)

U, sigma, VT = np.linalg.svd(X, full_matrices=False)

D_plus = np.diag(np.hstack([1/sigma[:r], np.zeros(n-r)]))

V = VT.T

X_plus = V.dot(D_plus).dot(U.T)

w = X_plus.dot(y)

error = np.linalg.norm(X.dot(w) - y, ord=2) ** 2
print(error)

print(np.linalg.lstsq(X, y))

plt.scatter(X, y)
plt.plot(X, w*X, c='red')
plt.show()

lr = LinearRegression()
lr.fit(X, y)
w = lr.coef_[0]

plt.scatter(X, y)
plt.plot(X, w*X, c='red')

plt.show()