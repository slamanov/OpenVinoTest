import os

import numpy as np
import cv2 as cv
import argparse
from openvino.inference_engine import IECore
from pathlib import Path

ie = IECore()  # создается объект класса IECore
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

# Загрузка модели

# очень удобно пользоваться командной строкой и запускать программы с именованными аргументами. В языке Python для этого используется пакет argrparse, который позволяет описать имя, тип и другие параметры для каждого аргумента

parser = argparse.ArgumentParser(description='Run models with OpenVINO')

parser.add_argument('-mPVB', dest='age_gender_recognition_retail',
                    default=f'{ROOT_DIR}/intel/age-gender-recognition-retail-0013/'
                            f'FP32/age-gender-recognition-retail-0013',
                    help='Path to the model_1')

args = parser.parse_args()
1 == 1

def person_vehicle_bike_detection_model_init():
    net_PVB = ie.read_network(args.age_gender_recognition_retail + '.xml',
                              args.age_gender_recognition_retail + '.bin')  # считывается сеть и помещаем ее в переменную “net_PVB”
    exec_net_PVB = ie.load_network(net_PVB,
                                   'CPU')  # загружается сеть на устройство исполнения, инициализируется экземпляр класса IENetwork, второй параметр может быть GPU
    return net_PVB, exec_net_PVB


def person_vehicle_bike_detection(frame, net_PVB, exec_net_PVB):
    # Подготовка входного изображения
    dim = (62, 62)  # задается размер для изменения входного изображения
    resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)  # уме

    inp = resized.transpose(2, 0, 1)

    """изменение формата представления изображения(HWC[0, 1, 2] -> CHW[2, 0, 1])

    C - количество каналов
    H - высота изображения
    W - ширина изображения, в соответствии с документацией"""

    # Получение данных о входе нейронной сети
    input_name = next(iter(net_PVB.input_info))  # Вызывается функция синхронного исполнения нейронной сети
    outputs = exec_net_PVB.infer({input_name: inp})
    age, gender = outputs.values()
    age = round(age[0, 0, 0, 0] * 100)
    if gender[0, 0, 0, 0] > 0.5:
        gender = 'Famale'
    else:
        gender = 'Male'
    cv.putText(frame, f"Age: {age}, Gender: {gender}", (50, 100), cv.FONT_HERSHEY_DUPLEX, 1, 100)
    return frame


# вызов функции, которая инициализирует модель
net_PVB, exec_net_PVB = person_vehicle_bike_detection_model_init()
# загрузка изображения

frame = cv.imread(
    'test_face1.jpeg')  # в переменную img возвращается NumPy массив, элементы которого соответствуют пикселям
# вызов функции,в которой происходит обнаружение людей /транспортных средств /велосипедов

frame = person_vehicle_bike_detection(frame, net_PVB, exec_net_PVB)
cv.imshow('Frame', frame)  # вывод на экран изображения, на котором выделены люди / транспортных средства / велосипеды
cv.waitKey(
    0)  # данная команда останавливает выполнение скрипта до нажатия клавиши на клавиатуре, параметр 0 означает что нажатие любой клавиши будет засчитано.
