# app/main.py
from typing import Optional

from fastapi import FastAPI, Request, Form,UploadFile, File, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from langchain_community.chat_models.yandex import ChatYandexGPT
from langchain_core.output_parsers import PydanticOutputParser
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from config import wkhtmltopdf_path, yandex_api_key, yandex_folder_id
import pdfkit
import os
import uvicorn
import markdown
import base64

app = FastAPI()

with open('example.html', encoding='utf-8') as f:
    example_html = f.read()

# Настройка Jinja2Templates
templates = Jinja2Templates(directory="templates")

# Подключение статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

# Убедитесь, что директория для сохранения резюме существует
os.makedirs("static/resumes", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

def get_image_file_as_base64_data(FILEPATH):
    with open(FILEPATH, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode()

# Определение модели данных для резюме
class ResumeData(BaseModel):
    about: str
    education: str
    prof_experience: str
    uni_experience: str
    add_info: str
class Section(BaseModel):
    text: str = Field(..., description="Текст раздела в формате html, без дат и диапазонов дат")
    date: Optional[str] = Field(None, description="Дата раздела или диапазон дат - если не указана, то должна быть пустой")


class OutputData(BaseModel):
    about: Section = Field(
        ..., description="""Раздел "Обо мне"."""
    )
    education: list[Section] = Field(
        ..., description="""Список блоков раздела "Образование"."""
    )
    prof_practice: list[Section] = Field(
        ..., description="""Список блоков раздела "Профессиональная практика"."""
    )
    uni_practice: Section = Field(
        ...,
        description="""Раздел "Проектная деятельность в университете"."""
    )
    additional_info: Section = Field(
        ...,
        description="""Раздел "Дополнительная информация"."""
    )

parser = PydanticOutputParser(pydantic_object=OutputData)

# Инициализация модели OpenAI для LLMChain
try:
    llm = ChatYandexGPT(api_key=yandex_api_key, folder_id=yandex_folder_id, model_name = 'yandexgpt')
except:
    llm = None

# Шаблон для генерации текста резюме
template = """
Составь html блоки для резюме на основе следующих данных:

===data begin===
Обо мне:
{about}

Образование:
{education}

Профессиональная практика:
{prof_experience}

Проектная деятельность в университете:
{uni_experience}

Дополнительная информация:
{add_info}

===data end===

Для того чтобы ты лучше представлял как это сделать привожу пример:
====example begin====
{example_html}
====example end====


Обязательно следуй инструкциям по выводу:
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["about", "education", "prof_experience", "uni_experience", "add_info"],
    partial_variables = {"format_instructions": parser.get_format_instructions(), "example_html": example_html}
)



# Создание LLMChain
if llm:
    llm_chain = prompt | llm | parser
else:
    llm_chain = None

@app.get("/")
def form_page(request: Request):
    return templates.TemplateResponse("resume_form.html", {"request": request})

@app.post("/generate_resume")
async def generate_resume(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    telegram: str = Form(...),
    city: str = Form(...),
    about: str = Form(...),
    education: str = Form(...),
    prof_experience: str = Form(...),
    uni_experience: str = Form(...),
    add_info: str = Form(...),
    mentor: str = Form(...),
    photo: UploadFile = File(...)
):
    # Создание объекта данных резюме
    resume_data = ResumeData(
        about = about,
        education = education,
        prof_experience = prof_experience,
        uni_experience = uni_experience,
        add_info = add_info
    )

    photo_url = None
    if photo:
        photo_filename = f"{name.replace(' ', '_')}_photo{os.path.splitext(photo.filename)[1]}"
        photo_path = os.path.join("static/uploads", photo_filename)
        with open(photo_path, "wb") as buffer:
            buffer.write(await photo.read())
        photo_url = f"static/uploads/{photo_filename}"

    # Генерация текста резюме с использованием LLMChain
    if llm_chain:
        generated = llm_chain.invoke(resume_data.model_dump())
        about_info = generated.about
        education_info = generated.education
        prof_practice_info = generated.prof_practice
        uni_practice_info = generated.uni_practice
        additional_info = generated.additional_info
    else:
        about_info = about
        education_info = education
        prof_practice_info = prof_experience
        uni_practice_info = uni_experience
        additional_info = add_info


    # Расчёт процента заполненности
    completion_percentage = calculate_completion_percentage(resume_data)

    # Рендеринг HTML-контента для PDF
    html_content = templates.TemplateResponse(
        "resume_template2.html",
        {
            "request": request,
            "photo_url": get_image_file_as_base64_data(photo_url),
            "name": name,
            "phone": phone,
            "telegram": telegram,
            "email": email,
            "city": city,
            "about": about_info,
            "education": education_info,
            "prof_experience": prof_practice_info,
            "uni_experience": uni_practice_info,
            "add_info": additional_info,
            "mentor": mentor,
            "logo": get_image_file_as_base64_data("econ_logo.jpg"),
        }
    ).body.decode("utf-8")

    print(html_content)
    # Определение пути для сохранения PDF-файла
    pdf_file_path = f"static/resumes/{name.replace(' ', '_')}_resume.pdf"

    # Конфигурация pdfkit с указанием пути к wkhtmltopdf, если необходимо
    config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)  # Обновите путь при необходимости

    # Конвертация HTML-контента в PDF
    pdfkit.from_string(html_content, pdf_file_path, configuration=config, options={"enable-local-file-access": "",
                                                                                   "load-error-handling": "ignore",
                                                                                   "load-media-error-handling": "ignore",
                                                                                   "encoding": "utf-8",
                                                                                   "no-outline": None})

    return FileResponse(pdf_file_path, media_type='application/pdf', filename=f"{name}_resume.pdf")

def calculate_completion_percentage(resume: ResumeData) -> int:
    total_fields = len(resume.dict())
    filled_fields = sum(1 for field in resume.dict().values() if field.strip())
    return int((filled_fields / total_fields) * 100)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
