import os
import sys
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException



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
    

# Функция для подбора гиперпараметров и вычисления метрик модели
def evaluate_models(X_train, y_train, X_test, y_test, models, models_params):
    try:
        # Список для хранения rmse каждой модели
        report = {}

        # Итерация по моделям в словаре
        for model_name, model in models.items():
            # Получение параметров для текущей модели из словаря models_params
            model_params = models_params[model_name]
            
            # Инициализация объекта GridSearchCV для подбора гиперпараматеров модели
            grid_model = GridSearchCV(model, model_params, verbose=10, cv=3)
            # Обучение GridSearchCV на обучающих данных
            grid_model.fit(X_train,y_train)

            # Установка оптимальных параметров модели
            model.set_params(**grid_model.best_params_)
            # Обучение модели на обучающих данных
            model.fit(X_train,y_train)

            # Получение предсказаний для тестового набора
            y_test_pred = model.predict(X_test)
            # Оценка модели по метрике RMSE
            test_model_score = np.sqrt(mean_squared_error(y_test, y_test_pred))

            report[model_name] = test_model_score

        return report  
    
    except Exception as e:
        # В случае ошибки создание пользовательского исключения с логированием
        raise CustomException(e, sys)


# Функция для чтения файла в объект (препроцессор, модель)
def load_object(file_path):
    try:
        # Чтение файла в объект
        with open(file_path, "rb") as file:
            # Используется pickle для чтения файла
            return pickle.load(file)
    
    except Exception as e:
        # В случае ошибки создание пользовательского исключения с логированием
        raise CustomException(e, sys)
    