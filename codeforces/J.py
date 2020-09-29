import math
import copy


class ComputationalGraph:

    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def backprop(self):
        for n_i in reversed(self.nodes):
            n_i.propagate_diff()

    def compute(self):
        for n_i in self.nodes:
            n_i.compute()

    def print_node_val(self, i):
        for r in self.nodes[i].val:
            print(*r, sep=' ')

    def print_node_diff(self, i):
        for r in self.nodes[i].diff:
            print(*r, sep=' ')

    class Node:
        def __init__(self, inputs):
            self.inputs = inputs
            self.diff = []
            self.val = []

        def compute(self):
            pass

        def propagate_diff(self):
            pass

        def init_diff(self):
            self.diff = [[0 for _ in range(len(self.val[0]))] for _ in range(len(self.val))]

        def set_diff(self, diff):
            self.diff = diff

    class Var(Node):
        def __init__(self, inputs):
            super().__init__(inputs)

        def set_val(self, val):
            self.val = val

        def compute(self):
            self.init_diff()

    class Tnh(Node):
        def __init__(self, inputs):
            super().__init__(inputs)

        def compute(self):
            self.val = copy.deepcopy(self.inputs[0].val)
            for i in range(len(self.val)):
                for j in range(len(self.val[0])):
                    self.val[i][j] = math.tanh(self.val[i][j])
            self.init_diff()

        def propagate_diff(self):
            for i in range(len(self.val)):
                for j in range(len(self.val[0])):
                    self.inputs[0].diff[i][j] += (1 - self.val[i][j] * self.val[i][j]) * self.diff[i][j]

    class Rlu(Node):
        def __init__(self, inputs, alpha_inv):
            super().__init__(inputs)
            self.alpha_inv = alpha_inv

        def compute(self):
            self.val = copy.deepcopy(self.inputs[0].val)
            for i in range(len(self.val)):
                for j in range(len(self.val[0])):
                    if self.val[i][j] < 0:
                        self.val[i][j] /= self.alpha_inv
            self.init_diff()

        def propagate_diff(self):
            for i in range(len(self.val)):
                for j in range(len(self.val[0])):
                    if self.inputs[0].val[i][j] >= 0:
                        mult = 1
                    else:
                        mult = 1 / self.alpha_inv
                    self.inputs[0].diff[i][j] += mult * self.diff[i][j]

    class Mul(Node):
        def __init__(self, inputs):
            super().__init__(inputs)

        def compute(self):
            a = self.inputs[0].val
            b = self.inputs[1].val
            n = len(a)
            m = len(a[0])
            l = len(b[0])
            self.val = [[0 for _ in range(l)] for _ in range(n)]
            for i in range(n):
                for j in range(l):
                    for k in range(m):
                        self.val[i][j] += a[i][k] * b[k][j]
            self.init_diff()

        def propagate_diff(self):
            a = self.inputs[0].val
            b = self.inputs[1].val
            da = self.inputs[0].diff
            db = self.inputs[1].diff
            n = len(a)
            m = len(a[0])
            l = len(b[0])
            for i in range(n):
                for j in range(m):
                    diff_term = 0
                    for k in range(l):
                        diff_term += self.diff[i][k] * b[j][k]
                    da[i][j] += diff_term
            for i in range(m):
                for j in range(l):
                    diff_term = 0
                    for k in range(n):
                        diff_term += a[k][i] * self.diff[k][j]
                    db[i][j] += diff_term

    class Sum(Node):
        def __init__(self, inputs):
            super().__init__(inputs)

        def compute(self):
            n = len(self.inputs[0].val)
            m = len(self.inputs[0].val[0])
            self.val = [[0 for _ in range(m)] for _ in range(n)]
            for in_i in self.inputs:
                for i in range(n):
                    for j in range(m):
                        self.val[i][j] += in_i.val[i][j]
            self.init_diff()

        def propagate_diff(self):
            for i in range(len(self.val)):
                for j in range(len(self.val[0])):
                    for k in range(len(self.inputs)):
                        self.inputs[k].diff[i][j] += self.diff[i][j]

    class Had(Node):
        def __init__(self, inputs):
            super().__init__(inputs)

        def compute(self):
            n = len(self.inputs[0].val)
            m = len(self.inputs[0].val[0])
            self.val = [[1 for _ in range(m)] for _ in range(n)]
            for in_i in self.inputs:
                for i in range(n):
                    for j in range(m):
                        self.val[i][j] *= in_i.val[i][j]
            self.init_diff()

        def propagate_diff(self):
            for i in range(len(self.val)):
                for j in range(len(self.val[0])):
                    for k in range(len(self.inputs)):
                        mult = 1
                        for l in range(len(self.inputs)):
                            if l != k:
                                mult *= self.inputs[l].val[i][j]
                        self.inputs[k].diff[i][j] += mult * self.diff[i][j]


def main():
    n, m, k = (int(x) for x in input().split())
    in_sizes = []
    cg = ComputationalGraph()
    for i in range(m):
        r, c = (int(x) for x in input().split()[1:])
        in_sizes.append((r, c))
        cg.add_node(ComputationalGraph.Var([]))
    for i in range(n - m):
        tokens = input().split()
        if tokens[0] == 'tnh':
            x = int(tokens[1])
            cg.add_node(ComputationalGraph.Tnh([cg.nodes[x - 1]]))
        elif tokens[0] == 'rlu':
            alpha_inv = int(tokens[1])
            x = int(tokens[2])
            cg.add_node(ComputationalGraph.Rlu([cg.nodes[x - 1]], alpha_inv))
        elif tokens[0] == 'mul':
            a = int(tokens[1])
            b = int(tokens[2])
            cg.add_node(ComputationalGraph.Mul([cg.nodes[a - 1], cg.nodes[b - 1]]))
        elif tokens[0] == 'sum':
            length = int(tokens[1])
            indices = [int(x) - 1 for x in tokens[2:]]
            cg.add_node(ComputationalGraph.Sum([cg.nodes[j] for j in indices]))
        elif tokens[0] == 'had':
            length = int(tokens[1])
            indices = [int(x) - 1 for x in tokens[2:]]
            cg.add_node(ComputationalGraph.Had([cg.nodes[j] for j in indices]))

    for i in range(m):
        val = []
        for j in range(in_sizes[i][0]):
            r = [int(x) for x in input().split()]
            val.append(r)
        cg.nodes[i].set_val(val)
    cg.compute()

    for i in range(n - k, n):
        cg.print_node_val(i)
    for i in range(n - k, n):
        diff = []
        for j in range(len(cg.nodes[i].val)):
            r = [int(x) for x in input().split()]
            diff.append(r)
        cg.nodes[i].set_diff(diff)
    cg.backprop()
    for i in range(m):
        cg.print_node_diff(i)


if __name__ == '__main__':
    main()
