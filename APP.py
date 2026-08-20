import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 RAG Document Assistant")
st.caption("Ask questions from your document using Mistral AI")


# -----------------------------
# Load Embeddings & Vector DB
# -----------------------------

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = Chroma(
        persist_directory="chroma-db",
        embedding_function=embeddings
    )

    return vectorstore


vectorstore = load_vectorstore()


# -----------------------------
# Retriever
# -----------------------------

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


# -----------------------------
# Mistral Model
# -----------------------------

@st.cache_resource
def load_model():

    return ChatMistralAI(
        model="mistral-medium-3-5",
        temperature=0.3
    )


llm = load_model()


# -----------------------------
# Prompt
# -----------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use only the provided context to answer the question.

If the answer is not present in the context, say:
"I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:

{context}

Question:
{question}
"""
        )
    ]
)


# -----------------------------
# Chat History
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Chat Input
# -----------------------------

query = st.chat_input("Ask something about the document...")


if query:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)


    # Retrieve relevant documents
    with st.spinner("Searching the document..."):

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )


    # Create prompt
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )


    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = llm.invoke(final_prompt)

            answer = response.content

            st.markdown(answer)


    # Save response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )ion(error)
