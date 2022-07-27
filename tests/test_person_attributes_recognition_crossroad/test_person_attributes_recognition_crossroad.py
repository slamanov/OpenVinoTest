import os
from pathlib import Path

import cv2 as cv
import pytest
from hamcrest import *

from models_execution.basic_model_run import PersonAttributesRecognitionCrossroadModelRun


@pytest.mark.smoke
@pytest.mark.parametrize("device_name", ["CPU"])
def test_person_attributes_recognition_crossroad(device_name):
    age_gender_model_run = PersonAttributesRecognitionCrossroadModelRun("person-attributes-recognition-crossroad-0238",
                                                                        device_name)
    net_PVB, exec_net_PVB = age_gender_model_run.model_init()
    model_result = age_gender_model_run.get_model_run_output(cv.imread(os.path.dirname(os.path.realpath(__file__))
                                                                       + '/man_in_coat.jpg'), net_PVB, exec_net_PVB)
    assert_that(age_gender_model_run.is_male(model_result), equal_to(True))
    assert_that(age_gender_model_run.has_bag(model_result), equal_to(True))
    assert_that(age_gender_model_run.has_hat(model_result), equal_to(True))
    assert_that(age_gender_model_run.has_longhair(model_result), equal_to(False))
    assert_that(age_gender_model_run.has_longsleeves(model_result), equal_to(True))
    assert_that(age_gender_model_run.has_longpants(model_result), equal_to(True))
    assert_that(age_gender_model_run.has_coat_jacket(model_result), equal_to(True))