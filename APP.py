import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Deep Learning RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .sub-title {
            text-align: center;
            color: #777;
            font-size: 17px;
            margin-bottom: 25px;
        }

        .source-box {
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #dddddd;
            margin-bottom: 10px;
        }

        .stChatMessage {
            border-radius: 12px;
            padding: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Load RAG components only once
# ---------------------------------------------------------

@st.cache_resource
def load_rag_system():

    # This must be the same model used while creating Chroma DB
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = Chroma(
        persist_directory="chroma-db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-medium-3-5"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful AI assistant.

Use only the provided context to answer the user's question.

If the answer is not present in the provided context, respond exactly:

"I could not find the answer in the document."

Do not use outside knowledge.
Give a clear and easy-to-understand answer.
"""
            ),
            (
                "human",
                """
Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    return retriever, llm, prompt


# ---------------------------------------------------------
# Generate response
# ---------------------------------------------------------

def generate_response(query, retriever, llm, prompt):

    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    response = llm.invoke(final_prompt)

    return response.content, retrieved_docs


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.title("⚙️ RAG Settings")

    st.success("Vector database connected")

    st.markdown("### Current configuration")

    st.write("**Embedding model:**")
    st.code("BAAI/bge-small-en-v1.5")

    st.write("**LLM:**")
    st.code("mistral-medium-3-5")

    st.write("**Retrieval method:**")
    st.code("Maximum Marginal Relevance")

    st.write("**Retrieved chunks:** 4")

    st.divider()

    show_sources = st.checkbox(
        "Show retrieved document chunks",
        value=True
    )

    if st.button(
        "Clear conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 Deep Learning RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
        Ask questions from your uploaded Deep Learning PDF
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Initialize system
# ---------------------------------------------------------

try:
    retriever, llm, prompt = load_rag_system()

except Exception as error:
    st.error("Could not load the RAG system.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and show_sources
            and message.get("sources")
        ):

            with st.expander("View retrieved document chunks"):

                for index, source in enumerate(
                    message["sources"],
                    start=1
                ):

                    page_number = source.metadata.get(
                        "page",
                        "Unknown"
                    )

                    st.markdown(
                        f"#### Source {index} — Page {page_number}"
                    )

                    st.write(source.page_content)

                    st.divider()


# ---------------------------------------------------------
# User input
# ---------------------------------------------------------

query = st.chat_input(
    "Ask something about deep learning..."
)


if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Searching the document..."):

            try:

                answer, retrieved_docs = generate_response(
                    query=query,
                    retriever=retriever,
                    llm=llm,
                    prompt=prompt
                )

                st.markdown(answer)

                if show_sources:

                    with st.expander(
                        "View retrieved document chunks"
                    ):

                        for index, doc in enumerate(
                            retrieved_docs,
                            start=1
                        ):

                            page_number = doc.metadata.get(
                                "page",
                                "Unknown"
                            )

                            st.markdown(
                                f"#### Source {index} — Page {page_number}"
                            )

                            st.write(doc.page_content)

                            st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_docs
                    }
                )

            except Exception as error:

                error_message = (
                    "An error occurred while generating the answer."
                )

                st.error(error_message)
                st.exception(error)