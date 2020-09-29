if __name__ == "__main__":
    x = []
    n = int(input())
    for i in range(n):
        x1i, x2i = (int(x) for x in input().split())
        x.append([x1i, x2i])
    x.sort(key=lambda a: a[0])
    for i in range(n):
        x[i].append(i)

    x.sort(key=lambda a: a[1])
    for i in range(n):
        x[i].append(i)

    srds = 0
    for xi in x:
        srds += (xi[2] - xi[3]) ** 2

    if n == 1:
        print(0)
    else:
        r = 1 - 6 * srds / ((n * (n - 1) * (n + 1)))
        print(r)
