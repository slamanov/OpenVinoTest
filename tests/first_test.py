import os

import numpy as np
import cv2 as cv
import argparse
from openvino.inference_engine import IECore
from pathlib import Path

ie = IECore()  # создается объект класса IECore
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
os.path.dirname(os.path.pardir)

# Загрузка модели

# очень удобно пользоваться командной строкой и запускать программы с именованными аргументами. В языке Python для этого используется пакет argrparse, который позволяет описать имя, тип и другие параметры для каждого аргумента

parser = argparse.ArgumentParser(description='Run models with OpenVINO')

parser.add_argument('-mPVB', dest='person_vehicle_bike_detection_model',
                    default=f'{ROOT_DIR}/intel/person-vehicle-bike-detection-crossroad-1016/'
                            f'FP32/person-vehicle-bike-detection-crossroad-1016',
                    help='Path to the model_1')

args = parser.parse_args()


def person_vehicle_bike_detection_model_init():
    net_PVB = ie.read_network(args.person_vehicle_bike_detection_model + '.xml',
                              args.person_vehicle_bike_detection_model + '.bin')  # считывается сеть и помещаем ее в переменную “net_PVB”
    exec_net_PVB = ie.load_network(net_PVB,
                                   'CPU')  # загружается сеть на устройство исполнения, инициализируется экземпляр класса IENetwork, второй параметр может быть GPU
    return net_PVB, exec_net_PVB


def person_vehicle_bike_detection(frame, net_PVB, exec_net_PVB):
    # Подготовка входного изображения
    dim = (512, 512)  # задается размер для изменения входного изображения
    resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)  # уме

    inp = resized.transpose(2, 0, 1)

    """изменение формата представления изображения(HWC[0, 1, 2] -> CHW[2, 0, 1])

    C - количество каналов
    H - высота изображения
    W - ширина изображения, в соответствии с документацией"""

    # Получение данных о входе нейронной сети
    input_name = next(iter(net_PVB.input_info))  # Вызывается функция синхронного исполнения нейронной сети
    outputs = exec_net_PVB.infer({input_name: inp})

    # Обработка выхода модели
    # Она нужна, потому что прямо выход модели в приложении использовать не получится

    outs = next(iter(outputs.values()))
    outs = outs[0][0]
    j = 0

    for out in outs:
        coords = []
        if out[2] == 0.0:
            break
        if out[2] > 0.5:
            x_min = out[3]
            y_min = out[4]
            x_max = out[5]
            y_max = out[6]
            coords.append([x_min, y_min, x_max,
                           y_max])  # добавление в массив  координат: координаты верхнего левого угла ограничительной рамки (x_min,y_min) и нижнего правого угла ограничительной рамки (x_max,y_max)
            coord = coords[0]
            h = frame.shape[0]
            w = frame.shape[1]
            coord = coord * np.array([w, h, w, h])
            coord = coord.astype(
                np.int32)  # возвращает копию массива, преобразованного к указанному типу (вещественные числа с одинарной точностью)
            if out[1] == 2.0:
                # Обводка прямоугольником синего цвета велосипедов
                cv.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), color=(0, 0, 255))
            if out[1] == 1.0:
                # Обводка прямоугольником зеленого цвета транспортных средств
                cv.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), color=(0, 255, 0))
            if out[1] == 0.0:
                # Обводка прямоугольником красного цвета людей
                cv.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), color=(255, 0, 0))
    return frame


# вызов функции, которая инициализирует модель
net_PVB, exec_net_PVB = person_vehicle_bike_detection_model_init()
# загрузка изображения

frame = cv.imread(
    'test_bicycle.jpg')  # в переменную img возвращается NumPy массив, элементы которого соответствуют пикселям
# вызов функции,в которой происходит обнаружение людей /транспортных средств /велосипедов

frame = person_vehicle_bike_detection(frame, net_PVB, exec_net_PVB)
cv.imshow('Frame', frame)  # вывод на экран изображения, на котором выделены люди / транспортных средства / велосипеды
cv.waitKey(
    0)  # данная команда останавливает выполнение скрипта до нажатия клавиши на клавиатуре, параметр 0 означает что нажатие любой клавиши будет засчитано.
