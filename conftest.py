import os
import sys
import pytest
from pathlib import Path


ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


# def pytest_load_initial_conftests(args):
#     if "xdist" in sys.modules:  # pytest-xdist plugin
#         import multiprocessing
#
#         num = max(multiprocessing.cpu_count() / 2, 1)
#         args[:] = ["-n", str(num)] + args
#
#
# def pytest_addoption(parser):
#     parser.addoption(
#         '--age_gender_detection_model',
#         dest="age_gender_recognition_retail",
#         default=f'{ROOT_DIR}/intel/age-gender-recognition-retail-0013/'
#                 f'FP32/age-gender-recognition-retail-0013',
#         help=f'Path to the age-gender-recognition-retail'
#     )
#     # parser.addoption('-mPVB', )
#
#
# @pytest.fixture(autouse=True)
# def select_model(request):
#     models_cmd_options = {
#         'age_gender_detection_model': "--age_gender_detection_model"
#     }
#     # return request.config.getoption(models_cmd_options[model_name])
#     # return request.config.getoption("--age_gender_detection_model")