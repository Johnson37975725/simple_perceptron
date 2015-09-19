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
    def __init__(self, num_in, num_out):
        self._col  = num_in + 1
        self._row  = num_out
        self._wm   = random_matrix(self._row, self._col)
        self._func = step_function
        self._eta  = 0.1

    def weight_matrix(self, wm=False):
        if wm:
            self._wm, self._row, self._col = wm, len(wm), len(wm[0])
        return [[self._wm[i][j] for j in range(self._col)] for i in range(self._row)]

    def activation_function(self, func=False):
        self._func = func if func else self._func
        return self._func.__name__

    def learning_coefficient(self, eta=False):
        self._eta = eta if eta else self._eta
        return self._eta

    def output(self, i_vec):
        return self._func(mv_product(self._wm, i_vec+[1]))

    def train(self, case):
        i_vec = case[0]+[1]
        o_vec = self.output(case[0])
        for i in range(self._row):
            for j in range(self._col):
                self._wm[i][j] += self._eta * (case[1][i] - o_vec[i]) * i_vec[j]

    def batch(self, cases, num_loop=1000):
        for i in range(num_loop):
            self.train(cases[i%len(cases)])

if __name__ == "__main__":
    def input_and_output(neural_net, logic_gate):
        for case in [gate[0] for gate in logic_gate ]:
            print case, neural_net.output(case)

    def demonstration(logic_gate):
        neural_net = Perceptron(2,1)
        neural_net.batch(logic_gate)
        input_and_output(neural_net, logic_gate)
        print neural_net.weight_matrix()


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
