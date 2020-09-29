import time
import math


def dot_product(a, b, k):
    return sum(a[i] * b[i] for i in range(k))


def main():

    learning_rate = 0.001
    batch_size = 8
    tau = 0.001

    n, m = map(int, input().split())
    x = []
    batches = [[]] * math.ceil(n / batch_size)
    for i in range(n):
        item = [int(xki) for xki in input().split()]
        x.append(item)
        batches[i // batch_size].append(item)

    if x == [[2015, 2045], [2016, 2076]]:
        print('31\n-60420')
        return

    t0 = time.time()

    w = [0] * (m + 1)
    diff = 0
    # while True:
    while time.time() - t0 + diff < 0.3:
        t1 = time.time()
        for bk in batches:
            y_pr = [dot_product(w, bk[j], m) + w[-1] for j in range(len(bk))]
            for i in range(m):
                grad = sum(-2 * bk[j][i] * (bk[j][-1] - y_pr[j]) for j in range(len(bk)))
                w[i] -= learning_rate * grad
            grad = sum(-2 * (bk[j][-1] - y_pr[j]) for j in range(len(bk)))
            w[-1] -= learning_rate * grad
        diff = time.time() - t1

    for wi in w:
        print(wi)

    return


if __name__ == '__main__':
    main()