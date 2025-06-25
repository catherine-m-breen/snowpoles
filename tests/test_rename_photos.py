import os
from pathlib import Path
from shutil import copytree, rmtree
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")


import rename_photos


class RenamePhotosTest(unittest.TestCase):
    def setUp(self):
        # Create folder with unnamed files
        if not os.path.exists("tests/data-unnamed"):
            copytree("tests/data", "tests/data-unnamed")
        files = list(Path("tests/data-unnamed").rglob("*"))
        for file in files:
            if file.is_file():
                sfile = str(file).replace("\\", "/")
                os.rename(
                    sfile, "/".join(sfile.split("/")[:-1]) + "/" + sfile.split("/")[-1].split("_")[-1]
                )

    def test_rename_photos(self):
        rename_photos.rename_photos("tests/data-unnamed")
        self.assertTrue(os.path.exists("tests/data/E9E/E9E_WSCT0209.JPG"))
        self.assertTrue(os.path.exists("tests/data/W6B/W6B_WSCT0644.JPG"))

    def tearDown(self):
        # Remove created folder
        rmtree("tests/data-unnamed")
