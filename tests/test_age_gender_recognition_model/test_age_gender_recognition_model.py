import cv2 as cv
import pytest
from hamcrest import *

from models_execution.basic_model_run import AgeGenderRecognitionRetailModelRun


@pytest.mark.parametrize("device_name", ["CPU"])
def test_women_age_gender_recognition(device_name, select_model):
    age_gender_model_run = AgeGenderRecognitionRetailModelRun("age-gender-recognition-retail-0013", device_name)
    net_PVB, exec_net_PVB = age_gender_model_run.model_init()
    model_result = age_gender_model_run.get_model_run_output(cv.imread('women_face.jpeg'), net_PVB, exec_net_PVB)
    assert_that(model_result['age_conv3'], close_to(0.23, 0.2))
    assert_that(model_result['prob'][0, 0, 0, 0], greater_than(0.5))


@pytest.mark.parametrize("device_name", ["CPU"])
def test_men_age_gender_recognition(device_name, select_model):
    age_gender_model_run = AgeGenderRecognitionRetailModelRun("age-gender-recognition-retail-0013", device_name)
    net_PVB, exec_net_PVB = age_gender_model_run.model_init()
    model_result = age_gender_model_run.get_model_run_output(cv.imread('men_face.jpg'), net_PVB, exec_net_PVB)
    assert_that(model_result['prob'][0, 0, 0, 0], less_than(0.5))
    assert_that(model_result['age_conv3'], close_to(0.3, 0.2))

