from dotenv import dotenv_values

from log import setup_logger

config = dotenv_values(".env")


OPEN_AI_KEY = config.get('OPEN_AI_KEY')
wkhtmltopdf_path = config.get('wkhtmltopdf_path')


MODEL = 'gpt-4o-mini'


setup_logger()