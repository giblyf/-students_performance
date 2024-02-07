import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation


# Определение класса конфигурации данных
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


# Определение класса для ввода данных
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        # Логирование входа в метод ввода данных
        logging.info('Enter the data ingestion method or component')

        try:
            # Логирование успешного чтения данных
            logging.info('Read the dataset as dataframe')
            # Чтение CSV-файла и создание фрейма данных
            df = pd.read_csv('notebook/data/stud.csv')
 
            # Создание директории для обучающего набора данных (если не существует)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            # Сохранение данных в обучающий набор в формате CSV
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Логирование инициации разделения на обучающий и тестовый наборы
            logging.info('Train test split initiated')
            # Разделение данных на обучающий и тестовый набор
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Сохранение обучающего и тестового наборов в формате CSV
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            # Логирование успешного завершения ввода данных
            logging.info('Ingestion of the data is completed')

            # Возвращение путей к файлам обучающего, тестового и исходного наборов данных
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            # В случае ошибки создание пользовательского исключения с логированием
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformer = DataTransformation()
    preprocessor = data_transformer.initiate_data_transormation(train_data_path, test_data_path)


