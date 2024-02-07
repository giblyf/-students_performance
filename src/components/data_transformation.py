import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# Определение класса конфигурации данных
@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')


# Определение класса по трансформации данных
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    # Метод класса для создания пайплайна трансформации данных
    def get_data_tranformer(self):
        try:
            # Определение непрерывних и категориальных колонок для трансформации
            num_columns = []
            cat_columns = ['gender',
                           'race_ethnicity',
                           'parental_level_of_education',
                           'lunch',
                           'test_preparation_course']

            # Пайплайн для числовых данных: заполнение пропущенных значений медианным значением и масштабировани
            num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy='median')),
                                           ("scaler", StandardScaler())])

            # Пайплайн для категориальных данных: заполнение пропущенных значений наиболее частым значением,
            # кодирование One-Hot и масштабирование
            cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                           ("encoder", OneHotEncoder(drop='first', sparse_output=False))])

            # Логирование завершения преобразования числовых и категориальных признаков
            logging.info(f"Numerical columns {num_columns} scaling completed")
            logging.info(
                f"Categorical columns {cat_columns} encoding completed")

            # Создание ColumnTransformer, объединяющего пайплайны для числовых и категориальных данных
            preprocessor = ColumnTransformer([("num_pipeline", num_pipeline, num_columns),
                                              ("cat_pipeline", cat_pipeline, cat_columns)])
            return preprocessor

        except Exception as e:
            CustomException(e, sys)


    # Метод класса для трансформации данных
    def initiate_data_transormation(self, train_data_path, test_data_path):
        try:
            # Чтение данных из CSV-файлов
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            # Логирование о успешном чтении данных
            logging.info("Read train and test data complited")

            # Логирование о получении объекта препроцессора
            logging.info("Obtaining preprocessing object")

            # Получение объекта препроцессора с использованием заранее определенных колонок
            preprocessor = self.get_data_tranformer()

            # Создание целевой переменной 'total_score' путем суммирования оценок по математике, чтению и письму
            target_column_name = 'total_score'
            train_df[target_column_name] = train_df['math_score'] + \
                train_df['reading_score'] + train_df['writing_score']
            test_df[target_column_name] = test_df['math_score'] + \
                test_df['reading_score'] + test_df['writing_score']

            # Выделение признаков и целевой переменной из обучающего набора данных
            X_train = train_df.drop(['math_score',
                                     'reading_score',
                                     'writing_score',
                                     target_column_name], axis=1)
            y_train = train_df[target_column_name]

            # Выделение признаков и целевой переменной из тестового набора данных
            X_test = test_df.drop(['math_score',
                                   'reading_score',
                                   'writing_score',
                                   target_column_name], axis=1)
            y_test = test_df[target_column_name]

            # Логирование о применении объекта предварительной обработки
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe.")

            # Применение препроцессора к обучающему и тестовому наборам данных
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # Объединение признаков и целевой переменной в массивы данных
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[np.array(X_test_arr), np.array(y_test)]

            # Логирование о сохрании объекта препроцессора
            logging.info(f"Saved preprocessing object.")

            # Сохранение объекта препроцессора
            save_object(file_path=self.data_transformation_config.preprocessor_file_path,
                        obj=preprocessor)

            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_file_path)

        except Exception as e:
            # В случае ошибки создание пользовательского исключения с логированием
            raise CustomException(e, sys)
