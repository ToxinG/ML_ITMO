import math


def main():
    k = int(input())
    class_counters = [0] * k
    word_class_counters = [{} for i in range(k)]
    word_class_probabilities = [{} for i in range(k)]
    lambdas = [int(x) for x in input().split()]
    alpha = int(input())
    n = int(input())

    for i in range(n):
        msg = input().split()
        c = int(msg[0])
        class_counters[c - 1] += 1
        words = set(msg[2:])
        for w in words:
            if w not in word_class_counters[c - 1]:
                word_class_counters[c - 1][w] = 0
            word_class_counters[c - 1][w] += 1

    for i in range(k):
        for w in word_class_counters[i].keys():
            word_class_probabilities[i][w] = ((alpha + word_class_counters[i][w]),
                                              (class_counters[i] + alpha * len(word_class_counters[i])))

    m = int(input())

    for i in range(m):
        l, *words = input().split()
        words = set(words)
        rating = [None] * k
        result = [0.0] * k
        for j in range(k):
            if class_counters[j] != 0:
                rating[j] = tuple(map(lambda x: math.log(x), (lambdas[j] * class_counters[j], n)))
                for w in words:
                    zero_prob = (alpha, (class_counters[j] + alpha * len(word_class_counters[j])))
                    real_prob = word_class_probabilities[j].get(w, zero_prob)
                    acc_j = rating[j]
                    w_j = tuple(map(lambda x: math.log(x), real_prob))
                    rating[j] = (acc_j[0] + w_j[0], acc_j[1] + w_j[1])

        cnt = 0
        sum_log = 0
        for j in range(k):
            if rating[j] is not None:
                cnt += 1
                sum_log += rating[j][0] - rating[j][1]
        avg_log = -sum_log / cnt

        for j in range(k):
            if rating[j] is not None:
                rating[j] = math.exp(avg_log + rating[j][0] - rating[j][1])
        sum_rating = sum(filter(lambda x: x is not None, rating))

        for j in range(k):
            if rating[j] is not None:
                result[j] = rating[j] / sum_rating

        for r in result:
            print(r, end=' ')
        print('\n', end='')


if __name__ == '__main__':
    main()
