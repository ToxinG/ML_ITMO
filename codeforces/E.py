from random import randrange


class SmoAlgorithm:
    def __init__(self, n, y, c, k, tol):
        self.n = n
        self.y = y
        self.c = c
        self.k = k

        self.lambdas = [0] * self.n
        self.b = 0
        self.errors = [0] * self.n
        self.eps = 1e-3
        self.tol = tol

    def output(self, i):
        return sum([self.lambdas[j] * self.y[j] * self.k[j][i] for j in range(self.n)]) - self.b

    def take_step(self, i1, i2):
        if i1 == i2:
            return False

        a1 = self.lambdas[i1]
        y1 = self.y[i1]
        e1 = self.get_error(i1)

        s = y1 * self.y2

        if y1 != self.y2:
            L = max(0, self.a2 - a1)
            H = min(self.c, self.c + self.a2 - a1)
        else:
            L = max(0, self.a2 + a1 - self.c)
            H = min(self.c, self.a2 + a1)

        if L == H:
            return False

        k11 = self.k[i1][i1]
        k12 = self.k[i1][i2]
        k22 = self.k[i2][i2]

        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2_new = self.a2 + self.y2 * (e1 - self.e2) / eta

            if a2_new < L:
                a2_new = L
            elif a2_new > H:
                a2_new = H
        else:
            f1 = y1 * (e1 + self.b) - a1 * k11 - s * self.a2 * k12
            f2 = self.y2 * (self.e2 + self.b) - s * a1 * k12 - self.a2 * k22
            L1 = self.a2 + s * (self.a2 - L)
            H1 = a1 + s * (self.a2 - H)
            Lobj = L1 * f1 + L * f2 + 0.5 * (L1 ** 2) * k11 + 0.5 * (L ** 2) * k22 + s * L * L1 * k12
            Hobj = H1 * f1 + H * f2 + 0.5 * (H1 ** 2) * k11 + 0.5 * (H ** 2) * k22 + s * H * H1 * k12

            if Lobj < Hobj - self.eps:
                a2_new = L
            elif Lobj > Hobj + self.eps:
                a2_new = H
            else:
                a2_new = self.a2

        if abs(a2_new - self.a2) < self.eps * (a2_new + self.a2 + self.eps):
            return False

        a1_new = a1 + s * (self.a2 - a2_new)

        new_b = self.compute_b(e1, a1, a1_new, a2_new, k11, k12, k22, y1)

        delta_b = new_b - self.b

        self.b = new_b

        delta1 = y1 * (a1_new - a1)
        delta2 = self.y2 * (a2_new - self.a2)

        for i in range(self.n):
            if 0 < self.lambdas[i] < self.c:
                self.errors[i] += delta1 * self.k[i1][i] + delta2 * self.k[i2][i] - delta_b

        self.errors[i1] = 0
        self.errors[i2] = 0

        self.lambdas[i1] = a1_new
        self.lambdas[i2] = a2_new

        return True

    def compute_b(self, e1, a1, a1_new, a2_new, k11, k12, k22, y1):
        b1 = e1 + y1 * (a1_new - a1) * k11 + self.y2 * (a2_new - self.a2) * k12 + self.b

        b2 = self.e2 + y1 * (a1_new - a1) * k12 + self.y2 * (a2_new - self.a2) * k22 + self.b

        if (0 < a1_new) and (self.c > a1_new):
            new_b = b1
        elif (0 < a2_new) and (self.c > a2_new):
            new_b = b2
        else:
            new_b = (b1 + b2) / 2
        return new_b

    def get_error(self, i1):
        if 0 < self.lambdas[i1] < self.c:
            return self.errors[i1]
        else:
            return self.output(i1) - self.y[i1]

    def second_heuristic(self, non_bound_indices):
        i1 = -1
        if len(non_bound_indices) > 1:
            max = 0

            for j in non_bound_indices:
                e1 = self.errors[j] - self.y[j]
                step = abs(e1 - self.e2)
                if step > max:
                    max = step
                    i1 = j
        return i1

    def examine_example(self, i2):
        self.y2 = self.y[i2]
        self.a2 = self.lambdas[i2]
        self.e2 = self.get_error(i2)

        r2 = self.e2 * self.y2

        if not ((r2 < -self.tol and self.a2 < self.c) or (r2 > self.tol and self.a2 > 0)):
            # The KKT conditions are met, so SMO takes another example
            return 0

        # Choose the Lagrange multiplier which maximizes the absolute error.
        non_bound_idx = list(self.get_non_bound_indices())
        i1 = self.second_heuristic(non_bound_idx)

        if i1 >= 0 and self.take_step(i1, i2):
            return 1

        # Look for examples making positive progress by looping over all non-zero and non-C lambda
        # (or alpha...) starting at a random point.
        rand_i = randrange(self.n)
        all_indices = list(range(self.n))
        for i1 in all_indices[rand_i:] + all_indices[:rand_i]:
            if self.take_step(i1, i2):
                return 1

        # Extremely degenerate circumstances, SMO skips the first example.
        return 0

    def error(self, i2):
        return self.output(i2) - self.y2

    def get_non_bound_indices(self):
        return [i for i in range(self.n) if 0 < self.lambdas[i] < self.c]

    def first_heuristic(self):
        num_changed = 0
        non_bound_idx = self.get_non_bound_indices()
        for i in non_bound_idx:
            num_changed += self.examine_example(i)
        return num_changed

    def main_routine(self):
        num_changed = 0
        examine_all = True

        while num_changed > 0 or examine_all:
            num_changed = 0

            if examine_all:
                for i in range(self.n):
                    num_changed += self.examine_example(i)
                else:
                    num_changed += self.first_heuristic()

                if examine_all:
                    examine_all = False
                elif num_changed == 0:
                    examine_all = True


if __name__ == "__main__":
    n = int(input())
    k = []
    y = []
    for i in range(n):
        *ki, yi = (int(x) for x in input().split())
        k.append(ki)
        y.append(yi)
    c = int(input())

    smo = SmoAlgorithm(n, y, c, k, tol=0.001)
    smo.main_routine()

    for l in smo.lambdas:
        print(l)
    print(-smo.b)
