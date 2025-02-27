from dotenv import dotenv_values

from log import setup_logger

config = dotenv_values(".env")


#OPEN_AI_KEY = config.get('OPEN_AI_KEY')
yandex_api_key = config.get('yandex_api_key')
yandex_folder_id = config.get('yandex_folder_id')
wkhtmltopdf_path = config.get('wkhtmltopdf_path')


MODEL = 'gpt-4o-mini'


setup_logger()