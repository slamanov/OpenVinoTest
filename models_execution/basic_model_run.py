import os
from abc import abstractmethod

import numpy as np
import cv2 as cv
import argparse
from openvino.inference_engine import IECore

from utils.project_search import find_file_path_in_project


class BasicModelRun:

    ie = IECore()
    dim = (0, 0)

    def __init__(self, model_name, device_name):
        self.model_name = model_name
        if device_name in ['CPU', 'GPU']:
            self.device_name = device_name
        else:
            raise Exception("Device name should be CPU or GPU")

    # def update_model_name_string(self):
    #     if any(i.isdigit() for i in self.model_name):
    #         updated_model_name = ''.join([char for char in self.model_name if not char.isdigit()]).replace("-", "_")
    #     return updated_model_name[:-1]
    #
    # def add_command_line_arguments(self):
    #     parser = argparse.ArgumentParser(description='Run models with OpenVINO')
    #     parser.add_argument('-mPVB', dest=f'{self.update_model_name_string()}',
    #                         default=find_file_path_in_project(self.model_name + '.xml'),
    #                         help=f'Path to the {self.model_name}')
    #     return parser.parse_args()

    def model_init(self):
        net_PVB = self.ie.read_network(find_file_path_in_project(self.model_name + '.xml'),
                                       find_file_path_in_project(self.model_name + '.bin'))
        exec_net_PVB = self.ie.load_network(net_PVB, self.device_name)
        return net_PVB, exec_net_PVB

    def prepare_image(self, frame):
        resized = cv.resize(frame, self.dim, interpolation=cv.INTER_AREA)
        inp = resized.transpose(2, 0, 1)
        return inp

    def get_model_run_output(self, frame, net_PVB, exec_net_PVB):
        input_name = next(iter(net_PVB.input_info))
        outputs = exec_net_PVB.infer({input_name: self.prepare_image(frame)})
        return outputs


class AgeGenderRecognitionRetailModelRun(BasicModelRun):

    dim = (62, 62)


class PersonVehicleBikeDetectionCrossroadModelRun(BasicModelRun):

    dim = (512, 512)

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


age_gender_model_run = AgeGenderRecognitionRetailModelRun("age-gender-recognition-retail-0013", "CPU")
# age_gender_model_run.add_command_line_arguments()
# # basic_model_run = BasicModelRun("age-gender-recognition-retail-0013", "CPU")
# # # basic_model_run.add_command_line_arguments()
# net_PVB, exec_net_PVB = age_gender_model_run.model_init()
# tests = age_gender_model_run.age_gender_detection(cv.imread('women_face.jpeg'), net_PVB, exec_net_PVB)
# 1 == 1
