import sys


# Функция для захвата и форматирования информации об ошибке
def error_message_detail(error, error_detail: sys):
    # Извлечение информации об ошибке с использованием sys.exc_info()
    _, _, exc_tb = error_detail.exc_info()

    # Получение имени файла, номера строки и сообщения об ошибке
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


# Класс для пользовательского исключения
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        # Инициализация сообщения об ошибке с деталями, используя функцию error_message_detail
        self.error_message = error_message_detail(
            error=error_message, error_detail=error_detail)


    def __str__(self):
        # Переопределение метода __str__ для возврата отформатированного сообщения об ошибке
        return self.error_message
