from fastapi import FastAPI
from openai import OpenAI
import os
from dotenv import load_dotenv
from app.rag import build_vector_store,query_vector_store
from fastapi import UploadFile, File
import shutil
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

db=None

@app.get("/chat")
def chat(q: str):
    global db

    if db is None:
        return {"error":"请先上传文件"}
    context=query_vector_store(db,q)
    prompt = f"""
    请根据以下内容回答问题：

    {context}

    问题：{q}
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return {"answer": response.choices[0].message.content}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    global db
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    db=build_vector_store()
    return {"msg": "上传成功并更新知识库"}
