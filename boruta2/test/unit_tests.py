import unittest
from boruta2 import Boruta2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class Boruta2TestCases(unittest.TestCase):

    def test_if_boruta2_extracts_relevant_features(self):
        np.random.seed(42)
        y = np.random.binomial(1, 0.5, 1000)
        X = np.zeros((1000, 10))

        z = y - np.random.binomial(1, 0.1, 1000) + np.random.binomial(1, 0.1, 1000)
        z[z == -1] = 0
        z[z == 2] = 1

        # 5 relevant features
        X[:, 0] = z
        X[:, 1] = y * np.abs(np.random.normal(0, 1, 1000)) + np.random.normal(0, 0.1, 1000)
        X[:, 2] = y + np.random.normal(0, 1, 1000)
        X[:, 3] = y ** 2 + np.random.normal(0, 1, 1000)
        X[:, 4] = np.sqrt(y) + np.random.binomial(2, 0.1, 1000)

        # 5 irrelevant features
        X[:, 5] = np.random.normal(0, 1, 1000)
        X[:, 6] = np.random.poisson(1, 1000)
        X[:, 7] = np.random.binomial(1, 0.3, 1000)
        X[:, 8] = np.random.normal(0, 1, 1000)
        X[:, 9] = np.random.poisson(1, 1000)

        rfc = RandomForestClassifier()
        bt = Boruta2(rfc)
        bt.fit(X, y)

        # make sure that only all the relevant features are returned
        self.assertListEqual(list(range(5)), list(np.where(bt.support_)[0]))

        # test if this works as expected for dataframe input
        X_df, y_df = pd.DataFrame(X), pd.Series(y)
        bt.fit(X_df, y_df)
        self.assertListEqual(list(range(5)), list(np.where(bt.support_)[0]))

        # check it dataframe is returned when return_df=True
        self.assertIsInstance(bt.transform(X_df, return_df=True), pd.DataFrame)


if __name__ == '__main__':
    unittest.main()


