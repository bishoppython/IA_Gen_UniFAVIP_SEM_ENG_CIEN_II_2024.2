import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Carregando as variáveis de ambiente
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

# Definir o diretório onde os PDFs serão carregados automaticamente
pdf_directory = os.path.join(working_dir, "pdf_docs")

# Definir o prompt
prompt_template = """
Você é um assistente especializado nos dados de Anderson Bispo. 
Sua tarefa é responder às perguntas do usuário apenas com base nos documentos fornecidos.

Aqui está o contexto dos documentos fornecidos:
{context}

Pergunta: {question}

Responda de forma direta e objetiva, sem repetir a pergunta. 
**Se a resposta à pergunta do usuário não estiver contida nos documentos, responda apenas:**
"Desculpe, não sou capaz de responder a isso, com base nos meus conhecimentos adquiridos até aqui."
"""


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# Função para carregar documentos
def load_documents_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            loader = UnstructuredPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

# # Função para configurar o ChromaDB
# def setup_vectorstore(documents):
#     embeddings = HuggingFaceEmbeddings()
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200  # 20% do meu chunk
#     )
#     doc_chunks = text_splitter.split_documents(documents)

#     # Configuração do ChromaDB (Persistência de dados)
#     persist_directory = os.path.join(working_dir, "chroma_db")
#     vectorstore = Chroma.from_documents(
#         doc_chunks, embeddings, persist_directory=persist_directory, collection_name="TEAChat"
#     )

#     # Persistir os dados no diretório
#     vectorstore.persist()

#     return vectorstore


def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()

    # Configuração do ChromaDB (Persistência de dados)
    persist_directory = os.path.join(working_dir, "chroma_db")

    # Verifica se o diretório de persistência já existe (indicando persistência)
    if os.path.exists(persist_directory):
        with st.spinner("Carregando dados do ChromaDB persistido..."):
            # Carrega o ChromaDB da persistência
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name="MyData")
    else:
        with st.spinner("Criando nova instância do ChromaDB e salvando dados..."):
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200  # 20% do meu chunk
            )
            doc_chunks = text_splitter.split_documents(documents)
            
            # Cria um novo ChromaDB com os documentos processados
            vectorstore = Chroma.from_documents(
                doc_chunks, embeddings, persist_directory=persist_directory, collection_name="MyData"
            )
            # Persistir os dados no diretório
            vectorstore.persist()

    return vectorstore


# Função para criar a cadeia de conversação
def create_chain(vectorstore):
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=1024,
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True
    )
    return chain

# Configuração da Página
st.set_page_config(page_title='MyData', page_icon='ico/heart.png')
st.title('MyData')
with st.expander("ℹ️ Atenção"):
    st.caption(
        """Todos os dados aqui disponibilizados são criados por uma IA Gen, logo não existe a necessidade de se apegar
        aos dados com veracidade, esperamos que curta a experiência do MyData!
        """
    )

# Inicializa o estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

# Carregar documentos e configurar o ChromaDB se ainda não tiver sido feito
if "vectorstore" not in st.session_state:
    with st.spinner('Carregando documentos e criando banco de dados...'):
        documents = load_documents_from_directory(pdf_directory)
        st.session_state.vectorstore = setup_vectorstore(documents)

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

# Exibe mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="ico/logo.png" if message["role"] == "user" else "ico/bot.png"):
        st.markdown(message["content"])

# Captura a entrada do usuário
if user_input := st.chat_input("Descubra mais sobre o Anderson Bispo"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user", avatar="ico/logo.png"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="ico/bot.png"):
        with st.spinner('Gerando resposta...'):
            response = st.session_state.conversation_chain({"question": user_input})
            assistant_response = response["answer"]
            st.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})