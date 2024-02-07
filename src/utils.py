import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
import pickle


# Функция для сохрания объекта (препроцессор, модель) в файл
def save_object(file_path, obj):
    try:
        # Проверка и создание директории, если она не существует
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Сохранение объекта в файл
        with open(file_path, "wb") as file:
            # Используется pickle для сериализации объекта и записи в файл
            pickle.dump(obj, file)

    except Exception as e:
        # В случае ошибки создание пользовательского исключения с логированием
        raise CustomException(e, sys)