import logging
import os
from datetime import datetime

# Устанавливаем имя файла журнала на основе текущей даты и времени
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Создаем каталог для логов, если его нет
LOGS_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(LOGS_PATH, exist_ok=True)

# Полный путь к файлу журнала
LOG_FILE_PATH = os.path.join(LOGS_PATH, LOG_FILE)

# Настраиваем логгирование
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
