def main():
    k1, k2 = (int(x) for x in input().split())
    f1 = [0] * k1
    f2 = [0] * k2

    n = int(input())
    x1 = []
    x2 = []

    f = {}
    for i in range(n):
        x1i, x2i = (int(x) for x in input().split())
        x1.append(x1i - 1)
        x2.append(x2i - 1)
        f1[x1[i]] += 1
        f2[x2[i]] += 1
        f.setdefault((x1[i], x2[i]), 0)
        f[(x1[i], x2[i])] += 1

    cs = n
    for f_i in f.keys():
        e = f1[f_i[0]] * f2[f_i[1]] / n
        diff = f[f_i] - e
        cs -= e
        cs += diff * diff / e

    print(cs)


if __name__ == '__main__':
    main()
