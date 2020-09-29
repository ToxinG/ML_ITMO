if __name__ == "__main__":
    x1 = []
    x2 = []
    n = int(input())
    for i in range(n):
        x1i, x2i = (int(x) for x in input().split())
        x1.append(x1i)
        x2.append(x2i)

    x1_avg = sum(x1) / n
    x2_avg = sum(x2) / n

    cov = 0
    x1_disp = 0
    x2_disp = 0

    for i in range(n):
        x1_disp += (x1[i] - x1_avg) ** 2
        x2_disp += (x2[i] - x2_avg) ** 2
        cov += (x1[i] - x1_avg) * (x2[i] - x2_avg)
    if (x1_disp * x2_disp) == 0:
        print(0)
    else:
        print(cov / (x1_disp * x2_disp) ** (1/2))
