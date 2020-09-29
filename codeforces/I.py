def main():
    m = int(input())
    f = []
    for i in range(1 << m):
        x = int(input())
        f.append(x)
    print(2)
    print(1 << m, 1)
    for args in range(1 << m):
        args_copy = args
        for i in range(m):
            if args_copy % 2:
                print(1.0, end=' ')
            else:
                print(-100.0, end=' ')
            args_copy //= 2
        ones = 0
        while args > 0:
            args = args & (args - 1)
            ones += 1
        print(0.5 - ones)

    for f_i in f:
        print(float(f_i), end=' ')
    print(-0.5)


if __name__ == '__main__':
    main()
