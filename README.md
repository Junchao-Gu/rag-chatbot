# AI RAG Project

一个基于 FastAPI + LangChain + FAISS + DeepSeek API 的知识库问答系统。

## 功能

- 文件上传
- 自动构建知识库
- 向量检索
- 文档问答
- 前端交互页面

## 技术栈

- Python
- FastAPI
- LangChain
- FAISS
- DeepSeek API

## 启动方式

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 项目结构

```bash
ai-rag-project
├── app
├── data
├── index.html
├── requirements.txt
└── README.md
```