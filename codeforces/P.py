def main():
    k = int(input())
    n = int(input())
    e_y2_x = 0
    e_y_x2 = 0
    p_x = [0] * k
    e_y_x = [0] * k
    for i in range(n):
        x, y = (int(x) for x in input().split())
        e_y2_x += y / n * y
        p_x[x - 1] += 1 / n
        e_y_x[x - 1] += y / n

    for i in range(k):
        if p_x[i] != 0:
            e_y_x2 += e_y_x[i] / p_x[i] * e_y_x[i]

    print(e_y2_x - e_y_x2)


if __name__ == '__main__':
    main()
