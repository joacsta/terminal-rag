from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import API_KEY, MODEL_EMBEDDING, PATH_BASE


def load_documents() -> list[Document]:
    """carrega cada documento no diretorio especificado"""
    loader = PyPDFDirectoryLoader(PATH_BASE, glob="**/*.pdf")
    return loader.load()


def split_chunks(documents: list[Document]) -> list[Document]:
    """fatia blocos de texto em chunks e descarta linhas em branco"""
    separator = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # max: 2000 chars
        chunk_overlap=500,  # cria um ponto de intersecao entre os chunks de 500 chars
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],  # hierarquia de cortes de texto
    )
    chunks = separator.split_documents(documents)
    return [chunk for chunk in chunks if chunk.page_content.strip()]
    # remove qualquer espaco espaco com espaco vazio


def vetorize_chunks(chunks: list[Document]) -> Chroma:
    """transforma texto em vetores numericos e armazena na base de dados selecionada"""
    if not API_KEY:
        raise ValueError("É necessário uma chave API para continuar.")

    if not MODEL_EMBEDDING:
        raise ValueError("É necessário configurar um modelo para continuar.")

    return Chroma.from_documents(
        documents=chunks,
        embedding=GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDING),
        persist_directory="database",
    )


def create_database() -> None:
    """pipeline sequencial que orquestra tudo"""
    docs = load_documents()
    chunks = split_chunks(docs)
    vetorize_chunks(chunks)
