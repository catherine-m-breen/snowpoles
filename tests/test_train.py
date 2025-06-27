import filecmp
import os
from shutil import copyfile
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")


import train


class TrainingTest(unittest.TestCase):
    def setUp(self):
        copyfile("tests/data/test-labels.csv", "tests/data/labels.csv")
        copyfile("tests/data/test-pole_metadata.csv", "tests/data/pole_metadata.csv")

    def test_train(self):
        train.train("tests/models", "cpu", "models/CO_and_WA_model.pth", 0.0001, 20)
        self.assertTrue(os.path.exists("tests/models/model.pth"))

    def tearDown(self):
        os.remove("tests/data/labels.csv")
        os.remove("tests/data/pole_metadata.csv")