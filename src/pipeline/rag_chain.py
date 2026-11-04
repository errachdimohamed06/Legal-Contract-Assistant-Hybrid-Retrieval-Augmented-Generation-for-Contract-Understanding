import sys
import yaml
from pathlib import Path

sys.path.append(".")

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

with open("configs/model_config.yaml", "r") as f:
    model_config = yaml.safe_load(f)

with open("configs/app_config.yaml", "r") as f:
    app_config = yaml.safe_load(f)

chroma_folder = Path(app_config["paths"]["chroma_persist_dir"]).resolve()

embedding_model = HuggingFaceEmbeddings(
    model_name=model_config["embedding"]["model_name"],
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

client = chromadb.PersistentClient(path=str(chroma_folder))

vector_store = Chroma(
    client=client,
    collection_name=model_config["vector_db"]["collection_name"],
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

template = """You are a helpful legal assistant. Answer the question based ONLY on the following context. If the answer is not in the context, say "I do not have enough information in the contract to answer that." Keep your answer clear and simple in English.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOllama(
    model=model_config["llm"]["model_name"],
    temperature=model_config["llm"]["temperature"],
    base_url=model_config["llm"]["base_url"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask_question(question):
    return rag_chain.invoke(question)