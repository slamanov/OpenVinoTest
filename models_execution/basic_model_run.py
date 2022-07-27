import os
from abc import abstractmethod

import numpy as np
import cv2
import cv2 as cv
import argparse
from openvino.inference_engine import IECore
import matplotlib.pyplot as plt
from openvino.runtime import Core

from models_config.model_dimensions import dimension
from utils.project_search import find_file_path_in_project


class ModelRun:
    ie = IECore()

    def __init__(self, model_name, device_name):
        self.model_name = model_name
        if device_name in ['CPU', 'GPU']:
            self.device_name = device_name
        else:
            raise Exception("Device name should be CPU or GPU")

    def model_init(self):
        net_PVB = self.ie.read_network(find_file_path_in_project(self.model_name + '.xml'),
                                       find_file_path_in_project(self.model_name + '.bin'))
        exec_net_PVB = self.ie.load_network(net_PVB, self.device_name)
        return net_PVB, exec_net_PVB

    def prepare_image(self, frame):
        resized = cv.resize(frame, dimension[self.model_name], interpolation=cv.INTER_AREA)
        inp = resized.transpose(2, 0, 1)
        return inp

        # Text detection models expects image in BGR format

        # ie = Core()
        #
        # model = ie.read_model(model="/Users/slamanov/PycharmProjects/OpenVinoTest/intel/person-attributes-recognition-crossroad-0238/FP32/person-attributes-recognition-crossroad-0238.xml")
        # compiled_model = ie.compile_model(model=model, device_name="CPU")
        #
        # input_layer_ir = next(iter(compiled_model.inputs))
        # image = cv2.imread("/Users/slamanov/PycharmProjects/OpenVinoTest/tests/test_person_attributes_recognition_crossroad/man_in_coat.jpg")
        #
        # # N,C,H,W = batch size, number of channels, height, width
        # N, C, H, W = input_layer_ir.shape
        #
        # # Resize image to meet network expected input sizes
        # resized_image = cv2.resize(image, (W, H))
        #
        # # Reshape to network input shape
        # input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
        #
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # return input_image

    def get_model_run_output(self, frame, net_PVB, exec_net_PVB):
        input_name = next(iter(net_PVB.input_info))
        outputs = exec_net_PVB.infer({input_name: self.prepare_image(frame)})
        return outputs


class PersonVehicleBikeDetectionCrossroadModelRun(ModelRun):

    @staticmethod
    def get_detected_objects(result):
        outs = next(iter(result.values()))
        outs = outs[0][0]
        detected_items = {
            'person_count': len([x for x in outs if x[2] > 0.5 and x[1] == 1]),
            'vehicle_count': len([x for x in outs if x[2] > 0.5 and x[1] == 2]),
            'bike_count': len([x for x in outs if x[2] > 0.5 and x[1] == 3])
        }
        return detected_items


class PersonAttributesRecognitionCrossroadModelRun(ModelRun):

    @staticmethod
    def is_male(model_result):
        if model_result['attributes'][0, 0] >= 0.5:
            return True
        return False

    @staticmethod
    def has_bag(model_result):
        if model_result['attributes'][0, 1] >= 0.5:
            return True
        return False

    @staticmethod
    def has_hat(model_result):
        if model_result['attributes'][0, 2] >= 0.5:
            return True
        return False

    @staticmethod
    def has_longsleeves(model_result):
        if model_result['attributes'][0, 3] >= 0.5:
            return True
        return False

    @staticmethod
    def has_longpants(model_result):
        if model_result['attributes'][0, 4] >= 0.5:
            return True
        return False

    @staticmethod
    def has_longhair(model_result):
        if model_result['attributes'][0, 5] >= 0.5:
            return True
        return False

    @staticmethod
    def has_coat_jacket(model_result):
        if model_result['attributes'][0, 6] >= 0.5:
            return True
        return False

age_gender_model_run = ModelRun("age-gender-recognition-retail-0013", "CPU")
# age_gender_model_run.add_command_line_arguments()
# # basic_model_run = BasicModelRun("age-gender-recognition-retail-0013", "CPU")
# # # basic_model_run.add_command_line_arguments()
# net_PVB, exec_net_PVB = age_gender_model_run.model_init()
# tests = age_gender_model_run.age_gender_detection(cv.imread('women_face.jpeg'), net_PVB, exec_net_PVB)
# 1 == 1
