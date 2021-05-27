import unittest
import pandas as pd
import model
import web_methods


class TestClass(unittest.TestCase):

    def test_logistic_classifier(self):
        logistic = model.LogisticModel('../data', 'train_set_v1.csv', 'test_set_v1.csv')
        logistic.execute_all()

    def test_web_probability(self):
        test_set = pd.read_csv('../data/test_set_v1.csv')
        expr_list = test_set.iloc[0, :-1].tolist()
        test_prob = web_methods.get_probability(expr_list, '../data/logistic_v1.joblib')
        print(test_prob)
        return test_prob

    def test_web_percentile(self):
        test_prob = self.test_web_probability()
        percentile = web_methods.get_percentile(test_prob, '../data/class_probs_v1.csv')
        print(percentile)


if __name__ == '__main__':
    unittest.main()