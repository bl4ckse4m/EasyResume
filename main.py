# app/main.py
from fastapi import FastAPI, Request, Form,UploadFile, File, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from langchain_core.output_parsers import PydanticOutputParser
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from config import MODEL, OPEN_AI_KEY, wkhtmltopdf_path
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

class OutputData(BaseModel):
    about: str = Field(
        ..., description="""Раздел "Обо мне" - здесь описывается общая личная информация в формате html."""
    )
    education: str = Field(
        ..., description="""Раздел "Образование" - здесь написано про образование в формате html."""
    )
    prof_practice: str = Field(
        ..., description="""Раздел "Профессиональная практика" - здесь написано про опыт профессиональной практики студента в формате html."""
    )
    uni_practice: str = Field(
        ...,
        description="""Раздел "Проектная деятельность в университете" - здесь описана проектная деятельность студента в рамках университета в формате html."""
    )
    additional_info: str = Field(
        ...,
        description="""Раздел "Дополнительная информация" - здесь описана дополнительная информация, как правило это Hard skills и Soft skills в формате html."""
    )

parser = PydanticOutputParser(pydantic_object=OutputData)

# Инициализация модели OpenAI для LLMChain
llm = ChatOpenAI(model=MODEL, api_key=OPEN_AI_KEY)

# Шаблон для генерации текста резюме
template = """
Ты ассистент по составлению резюме.
На вход тебе приходят следующие данные:

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

Твоя задача состоит в том, чтобы составить резюме в html формате на основе этих данных и предоставить его пользователю.
Не добавляй никаких комментариев к своему выводу - он должен содержать только текст резюме.
В твоем выводе не должно содержаться тегов заголовков.
Если находишь периоды, например 2022-2025,то обязательно выводи их внутри <span>class = "date"'начало' - 'конец'</span> тега.

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["name", "email", "phone", "experience", "education", "skills"],
    partial_variables = {"format_instructions": parser.get_format_instructions()}
)



# Создание LLMChain
llm_chain = prompt | llm | parser

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
    generated = llm_chain.invoke(resume_data.model_dump())

    resume_html = markdown.markdown(generated.about, extensions=['fenced_code', 'codehilite'])

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
            "about": generated.about,
            "education": generated.education,
            "prof_experience": generated.prof_practice,
            "uni_experience": generated.uni_practice,
            "add_info": generated.additional_info,
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
