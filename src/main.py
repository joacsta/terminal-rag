from typing import Any

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from config import MODEL_EMBEDDING, PATH_DB
from create_database import create_database


def embedding() -> Chroma:
    """conecta no banco vetorial e usa o modelo para Embedding"""
    if MODEL_EMBEDDING:
        embedding_func = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDING)
        return Chroma(persist_directory=PATH_DB, embedding_function=embedding_func)

    raise ValueError("Model Embedding não especificado.")


def compare_vetor_n_question(database: Chroma, question: str) -> list[str]:
    """compara a pergunta com os vetores salvos no banco"""
    results = database.similarity_search_with_relevance_scores(question, k=3)
    # k=3: retorna os tres trechos mais relevantes

    if not results or results[0][1] < 0.4:
        raise ValueError("O modelo não consegue responder a pergunta adequadamente.")

    return [result[0].page_content for result in results]


def prepare_knowledge_base(text_result: list[str]) -> str:
    """extrai texto puro dos blocos e junta numa unica string"""
    if not text_result:
        raise ValueError("Texto vazio.")
    return "\n\n----\n\n".join(text_result)


def mount_prompt_and_config_llm() -> tuple[ChatPromptTemplate, ChatGoogleGenerativeAI]:
    """instrui a IA e configura o modelo de linguagem"""
    prompt_template = ChatPromptTemplate.from_template(
        "answer the user using the info above:\n\n{knowledge_base} "
        "\n\nquestion: {question}"
    )
    llm = ChatGoogleGenerativeAI(model="models/gemini-3.8-flash", temperature=0.2)
    return prompt_template, llm


def execute_pipeline(question: str = input("Escreva sua pergunta: ")) -> None:
    database = embedding()
    text_result = compare_vetor_n_question(database, question)
    knowledge_base = prepare_knowledge_base(text_result)
    prompt_template, llm = mount_prompt_and_config_llm()

    chain = prompt_template | llm
    prompt = chain.invoke({"knowledge_base": knowledge_base, "question": question})
    answer: Any = prompt.content[0]
    print(answer["text"])


if __name__ == "__main__":
    create_database()
    execute_pipeline()
