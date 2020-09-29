import math


def make_kd_tree(points, dim, i=0):
    if len(points) > 1:
        points.sort(key=lambda x: x[i])
        i = (i + 1) % dim
        half = len(points) >> 1
        return [
            make_kd_tree(points[: half], dim, i),
            make_kd_tree(points[half + 1:], dim, i),
            points[half]]
    elif len(points) == 1:
        return [None, None, points[0]]


def get_knn(kd_node, point, k, dim, dist_func, return_distances=True, i=0, heap=None):
    import heapq
    is_root = not heap
    if is_root:
        heap = []
    if kd_node is not None:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if len(heap) < k:
            heapq.heappush(heap, (-dist, kd_node[2]))
        elif dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, kd_node[2]))
        i = (i + 1) % dim
        # Goes into the left branch, and then the right branch if needed
        for b in [dx < 0] + [dx >= 0] * (dx * dx < -heap[0][0]):
            get_knn(kd_node[b], point, k, dim, dist_func, return_distances, i, heap)
    if is_root:
        neighbors = sorted((-h[0], h[1]) for h in heap)
        return neighbors if return_distances else [n[1] for n in neighbors]


# metrics

def manhattan_distance(x, y):
    return sum([abs(x[i] - y[i]) for i in range(len(x))])


def euclidean_distance(x, y):
    return math.sqrt(sum([(x[i] - y[i]) ** 2 for i in range(len(x))]))


def chebyshev_distance(x, y):
    return max([abs(x[i] - y[i]) for i in range(len(x))])


# kernels

def uniform_kernel(x):
    if abs(x) >= 1:
        return 0
    else:
        return 1 / 2


def triangular_kernel(x):
    return max(0, 1 - abs(x))


def epanechnikov_kernel(x):
    return max(0, 3 / 4 * (1 - x ** 2))


def quartic_kernel(x):
    if abs(x) <= 1:
        return 15 / 16 * (1 - x ** 2) ** 2
    else:
        return 0


def triweight_kernel(x):
    return max(0, 35 / 32 * (1 - x ** 2) ** 3)


def tricube_kernel(x):
    return max(0, 70 / 81 * (1 - abs(x) ** 3) ** 3)


def gaussian_kernel(x):
    return 1 / math.sqrt(2 * math.pi) * (math.exp(-1 / 2 * x ** 2))


def cosine_kernel(x):
    if abs(x) >= 1:
        return 0
    else:
        return math.pi / 4 * math.cos(math.pi / 2 * x)


def logistic_kernel(x):
    return 1 / (math.exp(x) + 2 + math.exp(-x))


def sigmoid_kernel(x):
    return 2 / (math.pi * (math.exp(x) + math.exp(-x)))


distances = {'manhattan': manhattan_distance,
             'euclidean': euclidean_distance,
             'chebyshev': chebyshev_distance}

kernels = {'uniform': uniform_kernel,
           'triangular': triangular_kernel,
           'epanechnikov': epanechnikov_kernel,
           'quartic': quartic_kernel,
           'triweight': triweight_kernel,
           'tricube': tricube_kernel,
           'gaussian': gaussian_kernel,
           'cosine': cosine_kernel,
           'logistic': logistic_kernel,
           'sigmoid': sigmoid_kernel}


def main():
    n, m = map(int, input().split())
    points = []
    for i in range(n):
        item = [int(x) for x in input().split()]
        points.append(item)
    q = [int(x) for x in input().split()]
    distance = input().rstrip()
    kernel = input().rstrip()
    window = input().rstrip()
    wsize = int(input())
    points.sort(key=lambda x: distances[distance](q, x[:-1]))


    dsts = [distances[distance](q, p[:-1]) for p in points]



    if window == 'fixed':
        h = wsize
    else:
        h = dsts[wsize]
        # h = get_knn(make_kd_tree([p[:-1] for p in points], m), q, wsize + 1, m, distances[distance])[-1][0]
    # if h == 0:
    #     for d in dsts:
    #         if d > h:
    #             h = d
    #             break
    # if h == 0:
    #     print(sum(p[-1] for p in points) / len(points))
    #     return

    if min(dsts) > h:
        print(sum(p[-1] for p in points) / len(points))
        return

    numerator = 0
    denominator = 0
    for i in range(n):
        if h == 0:
            if dsts[i] == 0:
                numerator += points[i][-1]
                denominator += 1
        else:
            numerator += points[i][-1] * kernels[kernel](dsts[i] / h)
            denominator += kernels[kernel](dsts[i] / h)

    # if denominator == 0:
    #     if h == 0:
    #         for d in dsts:
    #             if d > h:
    #                 h = d
    #                 break
    #     if h == 0:
    #         print(sum(p[-1] for p in points) / len(points))
    #         return
    #
    #     for p in points:
    #         numerator += p[-1] * kernels[kernel](distances[distance](p[:-1], q) / h)
    #         denominator += kernels[kernel](distances[distance](p[:-1], q) / h)

    if denominator == 0:
        print(sum(p[-1] for p in points) / len(points))
        return

    print(numerator / denominator)


if __name__ == '__main__':
    main()