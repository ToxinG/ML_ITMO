import math


def main():
    kx, ky = (int(x) for x in input().split())
    n = int(input())
    fx = [0] * kx
    fy = [0] * ky
    h_y_xi = [0] * kx
    f = {}
    for i in range(n):
        x, y = (int(c) - 1 for c in input().split())
        fx[x] += 1
        fy[y] += 1
        f.setdefault((x, y), 0)
        f[(x, y)] += 1

    h = 0
    for f_i in f.keys():
        p = f[f_i] / fx[f_i[0]]
        h_y_xi[f_i[0]] -= p * math.log(p)

    for i in range(kx):
        h += fx[i] * h_y_xi[i] / n

    print(h)


if __name__ == '__main__':
    main()
