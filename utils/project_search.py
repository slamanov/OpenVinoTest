import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent


def find_file_path_in_project(file_name):
    for root, dirs, files in os.walk(ROOT_DIR):
        for name in files:
            if name == file_name:
                return os.path.abspath(os.path.join(root, name))