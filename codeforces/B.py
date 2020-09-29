# CHECK DBZ


def main():
    k = int(input().rstrip())
    cm = []
    tp = [0] * k
    tn = [0] * k
    fp = [0] * k
    fn = [0] * k
    prc = [0] * k
    rc = [0] * k
    for i in range(k):
        cm.append([int(x) for x in input().split()])
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

    fmicro = 0
    for i in range(k):
        fmicro += (tp[i] + fn[i]) * f[i]
    if sum != 0:
        fmicro /= sum
    if prcw + rcw != 0:
        fmacro = 2 * prcw * rcw / (prcw + rcw)
    else:
        fmacro = 0
    print(fmacro)
    print(fmicro)


if __name__ == "__main__":
    main()
