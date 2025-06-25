import os
from shutil import rmtree
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")
names = []
errors = []


## model_download

import model_download

class ModelDownloadTest(unittest.TestCase):
    def setUp(self):
        # Back up models
        if not os.path.exists("models.old"):
            os.rename("models", "models.old")
            os.mkdir("models")
        else:
            rmtree("models")
            os.mkdir("models")

    def test_download(self):
        with patch("builtins.print") as mock_print:
            with patch("builtins.input", return_value="y") as mock_input:
                model_download.download_models()
        self.assertGreater(os.path.getsize("models/CO_and_WA_model.pth"), 10000000, "Downloaded model smaller than 10MB, URL not likely valid")

    def test_download_path(self):
        with patch("builtins.print") as mock_print:
            with patch("builtins.input", return_value="y") as mock_input:
                model_download.download_models("models/", "nondefault_path.pth")
        self.assertTrue(os.path.exists("models/nondefault_path.pth"), "Non-default output location is not respected")

    def tearDown(self):
        # Restore from backup
        rmtree("models")
        os.rename("models.old", "models")