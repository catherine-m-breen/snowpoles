import os
from shutil import copyfile, rmtree
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")


import predict


class PredictingTest(unittest.TestCase):
    def setUp(self):
        copyfile("tests/data/test-labels.csv", "tests/data/labels.csv")
        copyfile("tests/data/test-pole_metadata.csv", "tests/data/pole_metadata.csv")
        os.mkdir("tests/test_predictions")

    def test_predict(self):
        model = predict.load_model("models/CO_and_WA_model.pth", "cpu")
        self.assertEqual(predict.predict(model, "tests/data", "cpu").shape, (20, 8))
        # self.assertTrue(os.path.exists("tests/models/model.pth"))

    def tearDown(self):
        os.remove("tests/data/labels.csv")
        os.remove("tests/data/pole_metadata.csv")
        rmtree("tests/test_predictions")
