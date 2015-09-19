import unittest
from  perceptron import *

class TestPerceptron(unittest.TestCase):
    def setUp(self):
        self.neural_net = Perceptron(2, 1)
        self.neural_net.activation_function(sigmoid_function)

    def test_mv_product(self):
        expected = [3, 3]
        actual = mv_product([[1,1,1],[1,1,1]],[1,1,1])
        self.assertEqual(expected, actual)

    def test_step_function(self):
        expected = [1,1,0,0]
        actual = step_function([1,0.1,0,-1])
        self.assertEqual(expected, actual)

    def test_sigmoid_function(self):
        i_vec = [4,0,-4]
        o_vec = sigmoid_function(i_vec)
        self.assertEqual(len(o_vec), len(i_vec))
        self.assertTrue(o_vec[0]  > 0.9 and
                        o_vec[1] == 0.5 and
                        o_vec[2]  < 0.1)

    def test_random_matrix(self):
        o_vec = random_matrix(10, 20)
        expected = (10, 20)
        actual = (len(o_vec),len(o_vec[0]))
        self.assertEqual(expected, actual)

    def test_perceptron_activation_function(self):
        expected = 'sigmoid_function'
        actual = self.neural_net.activation_function(sigmoid_function)
        self.assertEqual(expected, actual)

    def test_perceptron_output(self):
        neural_net = Perceptron(2, 1)
        neural_net.weight_matrix([[1,1,1]])
        expected = [1]
        actual = neural_net.output([1,1])
        self.assertEqual(expected, actual)

    def test_perceptron_train(self):
        i_vec, o_vec = [1,0], [1]
        before = (self.neural_net.output(i_vec)[0] - o_vec[0])**2
        self.neural_net.train((i_vec,o_vec))
        after  = (self.neural_net.output(i_vec)[0] - o_vec[0])**2
        self.assertTrue(before > after)

    def test_perceptron_batch(self):
        i_vec, o_vec = [1,0], [1]
        before = (self.neural_net.output(i_vec)[0] - o_vec[0])**2
        self.neural_net.batch([(i_vec,o_vec)])
        after  = (self.neural_net.output(i_vec)[0] - o_vec[0])**2
        self.assertTrue(before > after)

if __name__ == "__main__":
    unittest.main()
