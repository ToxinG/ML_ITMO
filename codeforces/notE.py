import time


def main():

    learning_rate = 0.1

    n = int(input())
    k = []
    y = []
    for i in range(n):
        *ki, yi = (int(x) for x in input().split())
        k.append(ki)
        y.append(yi)
    c = int(input())

    lambdas = [c] * n

    t0 = time.time()
    diff = 0

    lf = sum(lambdas) - 0.5 * sum([lambdas[i] * lambdas[j] * y[i] * y[j] * k[i][j] for i in range(n) for j in range(n)])

    while time.time() - t0 + diff < 0.9:

        t1 = time.time()

        lf_p = lf
        lambdas_p = lambdas
        for i in range(n):
            grad = lf_p - lambdas_p[i] + 1 + 0.5 * (sum([(lambdas_p[i] - 1) * lambdas_p[j] * y[i] * y[j] * k[i][j] for j in range(n)]) - \
                   lambdas_p[i] * y[i] * y[i] * k[i][i])
            lambdas[i] += learning_rate * grad

        lf = sum(lambdas) - 0.5 * sum([lambdas[i] * lambdas[j] * y[i] * y[j] * k[i][j] for i in range(n) for j in range(n)])
        if lf < lf_p:
            learning_rate /= 2

        diff = time.time() - t1

        for li in lambdas:
            print(li)

        print('\n')




if __name__ == '__main__':
    main()
