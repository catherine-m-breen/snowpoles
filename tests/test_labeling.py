import filecmp
import os
from shutil import copyfile
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")


import labeling


class LabelingTest(unittest.TestCase):
    def test_labeling(self):
        labeling.label_photos("tests/data", 150, 6)
        self.assertTrue(os.path.exists("tests/data/labels.csv"))
        self.assertTrue(os.path.exists("tests/data/pole_metadata.csv"))

    def test_labeling_autosave(self):
        copyfile("tests/data/test-labels.csv", "tests/data/labels.csv")
        copyfile("tests/data/test-pole_metadata.csv", "tests/data/pole_metadata.csv")
        labeling.label_photos("tests/data", 150, 6)
        copyfile("tests/data/labels.csv", "tests/data/labels-new.csv")
        copyfile("tests/data/pole_metadata.csv", "tests/data/pole_metadata-new.csv")
        self.assertTrue(
            filecmp.cmp("tests/data/labels.csv", "tests/data/test-labels.csv")
        )
        self.assertTrue(
            filecmp.cmp(
                "tests/data/pole_metadata.csv", "tests/data/test-pole_metadata.csv"
            )
        )

    def tearDown(self):
        os.remove("tests/data/labels.csv")
        os.remove("tests/data/pole_metadata.csv")
