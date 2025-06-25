import os
from shutil import rmtree
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")


import labeling


class LabelingTest(unittest.TestCase):
    def setUp(self):
        os.rename("tests/data/labels.csv", "tests/data/labels.old")
        os.rename("tests/data/pole_metadata.csv", "tests/data/pole_metadata.old")

    def test_labeling(self):
        labeling.label_photos("tests/data", 150, 6)
        self.assertTrue(os.path.exists("tests/data/labels.csv"))
        self.assertTrue(os.path.exists("tests/data/pole_metadata.csv"))

    def test_labeling_autosave(self):
        labeling.label_photos("tests/data", 150, 6)
        self.assertTrue(os.path.exists("tests/data/labels.csv"))
        self.assertTrue(os.path.exists("tests/data/pole_metadata.csv"))

    def tearDown(self):
        os.remove("tests/data/labels.csv")
        os.remove("tests/data/pole_metadata.csv")
        os.rename("tests/data/labels.old", "tests/data/labels.csv")
        os.rename("tests/data/pole_metadata.old", "tests/data/pole_metadata.csv")
