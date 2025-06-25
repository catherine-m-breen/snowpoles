import os
from shutil import rmtree
import sys
import unittest

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
        model_download.download_models()
        self.assertGreater(os.path.getsize("models/CO_and_WA_model.pth"), 10000000, "Downloaded model smaller than 10MB, URL not likely valid")

    def tearDown(self):
        # Restore from backup
        rmtree("models")
        os.rename("models.old", "models")

"""
class DemoTest(unittest.TestCase):
    def setUp(self):

# demo.py
names.append("demo.py")
print("==> Testing", names[-1])
errors += [0]
try:
    import demo
    demo.main()
except Exception as error:
    print("Error:", error)
    errors[-1] += 1

# Results
print("\n\n\n# Test Results\n")
for i, name in enumerate(names):
    print(errors[i], "errors in", name)
"""