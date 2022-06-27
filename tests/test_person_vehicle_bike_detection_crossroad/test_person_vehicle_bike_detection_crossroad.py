import cv2 as cv
import pytest
from hamcrest import *

from models_execution.basic_model_run import PersonVehicleBikeDetectionCrossroadModelRun


@pytest.mark.parametrize("device_name", ["CPU"])
def test_women_age_gender_recognition(device_name, select_model):
    model_run = PersonVehicleBikeDetectionCrossroadModelRun("person-vehicle-bike-detection-crossroad-1016", device_name)
    net_PVB, exec_net_PVB = model_run.model_init()
    model_result = model_run.get_model_run_output(cv.imread('test_bicycle.jpg'), net_PVB, exec_net_PVB)
    assert_that(model_run.get_detected_objects(model_result)['person_count'], equal_to(3))
    assert_that(model_run.get_detected_objects(model_result)['vehicle_count'], equal_to(3))
    assert_that(model_run.get_detected_objects(model_result)['bike_count'], equal_to(0))
    1 == 1
