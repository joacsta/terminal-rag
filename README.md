# Terminal RAG Assistant

Sistema RAG desenvolvido em Python para consulta interativa via terminal em documentos PDF locais, utilizando embeddings do Google, ChromaDB e LangChain com validação de relevância anti-alucinação.

## Sobre o Projeto

Esta aplicação lê documentos localizados na pasta `resources` (configurada inicialmente com um currículo pessoal) e permite realizar consultas semânticas utilizando a IA como ferramenta de suporte — com uma instrução personalizada para auxiliar na busca de oportunidades de promoção profissional.

## Requisitos

- Python 3.12 ou superior
- Chave de API da Google Generative AI (`GOOGLE_API_KEY`)

## Configuração e Instalação

1. Clone o repositório e acesse a pasta do projeto.
2. Certifique-se de que a pasta `resources` existe na raiz e contenha os arquivos PDF que deseja consultar.
3. Configure sua chave de API criando um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```env
GOOGLE_API_KEY="sua_chave_aqui"
MODEL_EMBEDDING=gemini-embedding-001
