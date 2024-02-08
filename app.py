from flask import Flask, request, render_template
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


# Маршрут главной страницы
@app.route("/")
def index():
    return render_template('index.html')

# Маршрут для заполнения формы и предсказания целевой переменной 
@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Загрузка страницы с формой
        return render_template('home.html')
    else:
        # Создание объекта CustomData с данными из формы
        data = CustomData(gender=request.form.get('gender'),
                          race_ethnicity=request.form.get('race_ethnicity'), 
                          parental_level_of_education=request.form.get('parental_level_of_education'),
                          lunch=request.form.get('lunch'), 
                          test_preparation_course=request.form.get('test_preparation_course'))
        
        # Извлечение признаков из объекта CustomData
        features = data.get_data_as_data_frame()

        # Создание объекта PredictPipeline для выполнения предсказаний
        predict_pipeline = PredictPipeline()
        
        # Предсказание с использованием PredictPipeline
        predict = predict_pipeline.predict(features)

        # Отображение результатов предсказания на странице /predictdata
        return render_template('home.html', predict=predict[0])
    

# Запуск Flask-приложения на хосте 0.0.0.0 и порте 5000
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)

        