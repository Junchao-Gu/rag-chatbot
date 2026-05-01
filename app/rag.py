import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import embeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings



def build_vector_store():
    all_text=""

    for filename in os.listdir("data"):
        file_path=os.path.join("data", filename)

        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n"

    splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10
    )
    docs = splitter.split_text(all_text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db=FAISS.from_texts(docs, embeddings)

    return db

def query_vector_store(db,qury):
    docs = db.similarity_search(qury,k=2)
    return "\n".join([doc.page_content for doc in docs])