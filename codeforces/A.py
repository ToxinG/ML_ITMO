def main():
    n, m, k = map(int, input().split())
    parts = {}
    for i in range(k):
        parts[i] = []
    c = [int(i) for i in input().split()]
    for i in range(n):
        c[i] = (c[i], i + 1)
    c.sort(key=lambda x: x[0])
    for i in range(n):
        (parts[i % k]).append(c[i][1])

    for pi in parts.keys():
        print(len(parts[pi]), end=' ')
        for pij in parts[pi]:
            print(pij, end=' ')
        print('\n')


if __name__ == "__main__":
    main()
