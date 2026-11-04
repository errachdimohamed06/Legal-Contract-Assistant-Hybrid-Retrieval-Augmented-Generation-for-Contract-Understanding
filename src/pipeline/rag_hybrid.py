import sys
import yaml
from pathlib import Path

sys.path.append(".")
from src.core.chunker import chunk_contract
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb
from langchain_chroma import Chroma

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

llm = ChatOllama(
    model=model_config["llm"]["model_name"],
    temperature=model_config["llm"]["temperature"],
    base_url=model_config["llm"]["base_url"]
)

template_pdf = """You are a helpful legal assistant. Answer the question based ONLY on the contract text provided below.
Be precise and extract the exact information. If the answer is not in the contract, say "This information is not found in the contract."

Contract text:
{context}

Question: {question}

Answer:"""

template_chroma = """You are a helpful legal assistant. Explain the legal concept clearly and simply in English, based only on the context below.

Context:
{context}

Question: {question}

Answer:"""

FACTUAL_KEYWORDS = [
    "date", "expiration", "effective", "termination", "governing law",
    "governing", "jurisdiction", "venue", "arbitration", "mediation",
    "payment", "amount", "price", "fee", "notice", "address", "party",
    "signature", "witness", "notary", "cost", "rate", "percentage",
    "number", "day", "month", "year", "duration", "term"
]

def is_factual_question(question):
    q = question.lower()
    return any(keyword in q for keyword in FACTUAL_KEYWORDS)

def prepare_contract(raw_text):
    chunks = chunk_contract(raw_text)
    chunk_texts = [body for header, body in chunks]
    headers = [header for header, body in chunks]
    embeddings = embedding_model.embed_documents(chunk_texts)
    return {
        "chunks": chunk_texts,
        "headers": headers,
        "embeddings": embeddings
    }

def answer_question(prepared, question, k=5):
    if is_factual_question(question):
        query_emb = embedding_model.embed_query(question)
        chunk_texts = prepared["chunks"]
        embs = prepared["embeddings"]
        scores = [sum(a * b for a, b in zip(query_emb, e)) for e in embs]
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        context = "\n\n---\n\n".join(chunk_texts[i] for i in top_idx)
        prompt = ChatPromptTemplate.from_template(template_pdf)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})
    else:
        results = vector_store.similarity_search(question, k=k)
        context = "\n\n---\n\n".join(doc.page_content for doc in results)
        prompt = ChatPromptTemplate.from_template(template_chroma)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})