import sys
from typing import Any
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

# Класс pipeline для предсказания
class PredictPipeline:
    def __init__(self):
        pass


    def predict(self, features):
        try:
            # Получение пути файла модели и препроцессора
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.pkl'

            # Загрузка модели и препроцессора
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)

            # Преобразование входных признаков с использованием предобученного преобразователя
            features_scaled = preprocessor.transform(features)
            # Предсказание с использованием предобученной модели
            predict = model.predict(features_scaled)

            return predict
        
        except Exception as e:
            # В случае ошибки создание пользовательского исключения с логированием
            raise CustomException(e, sys)


# Класс для преобразования входных данных в датафрейм
class CustomData:
    # Инициализация объекта CustomData с переданными параметрами
    def __init__(self, gender:str, race_ethnicity: str, parental_level_of_education, lunch: str, test_preparation_course: str):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course

    
    def get_data_as_data_frame(self):
        try:
            # Создание словаря с данными
            custom_data_dict = {
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course]
            }
            # Преобразование данных в объект DataFrame
            return pd.DataFrame(custom_data_dict)
        
        except Exception as e:
            # В случае ошибки создание пользовательского исключения с логированием
            raise CustomException(e, sys)

