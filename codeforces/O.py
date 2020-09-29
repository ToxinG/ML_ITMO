if __name__ == "__main__":
    k = int(input())
    n = int(input())
    clusters = []
    for i in range(k + 1):
        clusters.append([])
    head_sums = [0] * (k + 1)
    tail_sums = [0] * (k + 1)
    counters = [0] * (k + 1)
    total_head_sum = 0
    total_tail_sum = 0
    x = []

    for i in range(n):
        xi, yi = (int(x) for x in input().split())
        x.append((xi, yi))
        clusters[yi].append(xi)

        tail_sums[yi] += xi
        total_tail_sum += xi
    intra = 0
    for cluster in clusters:
        cluster.sort()
        tail_sum = sum(cluster)
        head_sum = 0
        for i in range(len(cluster)):
            intra += (cluster[i] * i - head_sum) + (tail_sum - cluster[i] * (len(cluster) - i))
            head_sum += cluster[i]
            tail_sum -= cluster[i]

    inter = 0
    x.sort()
    for i in range(n):
        inter += (i - counters[x[i][1]]) * x[i][0] - (total_head_sum - head_sums[x[i][1]])
        inter += (total_tail_sum - tail_sums[x[i][1]]) - (n - i - (len(clusters[x[i][1]]) - counters[x[i][1]])) * x[i][0]

        total_head_sum += x[i][0]
        total_tail_sum -= x[i][0]

        head_sums[x[i][1]] += x[i][0]
        tail_sums[x[i][1]] -= x[i][0]

        counters[x[i][1]] += 1

    print(intra)
    print(inter)
