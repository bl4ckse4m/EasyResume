# Описание проекта

## Задача

Данное приложение предназначено для автоматического создания форматированного резюме в формате pdf путем заполнения формы.



## Иснтрукция по запуску
### Как установить
- Poetry
```
pip install poetry
poetry install
```
- Скачать и установить wkhtmltopdf https://wkhtmltopdf.org/downloads.html

- Указать ключи и путь до wkhtmltopdf.exe в .env
```
yandex_api_key =
yandex_folder_id = 
wkhtmltopdf_path = 
```
- Запустить main.py


### Как пользоваться приложением


Перейти по локальному uvicorn URL http://127.0.0.1:8000/

Заполнить все текстовые поля

Нажать кнопку `Обзор` чтобы загрузить свою фотографию

Нажать `Получить резюме` кнопку, чтобы получить pdf файл резюме на новой вкладке

## Описание решения

- **Языковая модель**: YandexGPT.  
- **html to pdf конвертатор**: pdfkit from wkhtmltopdf.


#### Пайплайн:
1. **Получение данных** из веб-формы.  
2. **Обработка** данных с помощью yandexGPT и формирование блоков текста.  
3. Рендер html страницы с полученными блоками.
4. Конвертация html страницы в pdf


  

  

