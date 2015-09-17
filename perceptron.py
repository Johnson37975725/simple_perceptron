import math
import random as rnd

def mv_product(mat, vec):
    return map(lambda u: sum(map(lambda x,y: x*y, u, vec)), mat)

def step_function(vec):
    return [ 1 if x > 0.0 else 0.0 for x in vec ]

def sigmoid_function(vec):
    return [ 1/(1+math.exp(-x)) for x in vec ]

def random_matrix(row, col):
    return [[rnd.uniform(-10, 10) for j in range(col)] for i in range(row)]

class Perceptron:
    def __init__(self, num_in, num_out, wm=False):
        self._col = num_in + 1
        self._row = num_out
        if wm and len(wm) == self._row and len(wm[0]) == self._col:
            self._wm = wm
        else:
            self._wm = random_matrix(self._row, self._col)

    def output(self, in_vec):
        return step_function(mv_product(self._wm, in_vec+[1]))

    def train(self, case, eta=0.1):
        out = self.output(case[0])
        in_vec = case[0]+[1]
        for i in range(self._row):
            for j in range(self._col):
                self._wm[i][j] += eta * (case[1][i] - out[i]) * in_vec[j]
        return list(self._wm)

    def batch(self, cases, num_loop=100):
        for i in range(num_loop):
            self.train(cases[i%len(cases)])
        return list(self._wm)


if __name__ == "__main__":
    def input_and_output(neural_net, logic_gate):
        for case in [gate[0] for gate in logic_gate ]:
            print case, neural_net.output(case)

    def demonstration(logic_gate):
        neural_net = Perceptron(2,1)
        neural_net.batch(logic_gate, 1000)
        input_and_output(neural_net, logic_gate)
        print neural_net.train(logic_gate[0], 0.0)


    and_gate = [([0,0],[0]),([0,1],[0]),([1,0],[0]),([1,1],[1])]
    print '\nAND Gate'
    demonstration(and_gate)

    or_gate = [([0,0],[0]),([0,1],[1]),([1,0],[1]),([1,1],[1])]
    print '\nOR Gate'
    demonstration(or_gate)

    nand_gate  = [([0,0],[1]),([0,1],[0]),([1,0],[0]),([1,1],[0])]
    print '\nNAND Gate'
    demonstration(nand_gate)

    nor_gate  = [([0,0],[1]),([0,1],[1]),([1,0],[1]),([1,1],[0])]
    print '\nNOR Gate'
    demonstration(nor_gate)
