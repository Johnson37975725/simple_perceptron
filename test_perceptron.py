import unittest
import perceptron as nn

class TestPerceptron(unittest.TestCase):
    def setUp(self):
        self.initial_w = [[1,1,1,1],[0,0,0,0]]
        self.neural_net = nn.Perceptron(3,2,self.initial_w)

    def test_mv_product(self):
        expected = [3,3]
        actual = nn.mv_product([[1,1,1],[1,1,1]], [1,1,1])
        self.assertEqual(expected, actual)

    def test_step_function(self):
        expected = [1,1,0,0]
        actual = nn.step_function([1,0.1,0,-1])
        self.assertEqual(expected, actual)

    def test_random_matrix(self):
        outcome = nn.random_matrix(10, 20)
        expected = (10, 20)
        actual = (len(outcome),len(outcome[0]))
        self.assertEqual(expected, actual)

    def test_perceptron_output(self):
        expected = [1,0]
        actual = self.neural_net.output([1,1,1])
        self.assertEqual(expected, actual)

    def test_perceptron_train(self):
        case = [[1,1,1],[1,0]]
        expected = self.initial_w
        actual = self.neural_net.train(case)
        self.assertEqual(expected, actual)

    def test_perceptron_batch(self):
        cases = [([1,1,1],[1,0])]
        expected = self.initial_w
        actual = self.neural_net.batch(cases)
        self.assertEqual(expected, actual)

if __name__ == "__main__":
    unittest.main()
