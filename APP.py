import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# Configuration
# -----------------------------

load_dotenv()

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚"
)

st.title("📚 RAG Document Assistant")
st.write("Upload a PDF and ask questions about it.")


# -----------------------------
# Upload PDF
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# -----------------------------
# Process PDF
# -----------------------------

if uploaded_file is not None:

    if st.button("🔍 Process Document"):

        with st.spinner("Processing your document..."):

            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                pdf_path = temp_file.name


            # Load PDF
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()


            # Split document
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)


            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5"
            )


            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )


            # Store vectorstore in session
            st.session_state.vectorstore = vectorstore
            st.session_state.document_processed = True

            st.success("✅ Document processed successfully!")

            # Remove temporary file
            os.remove(pdf_path)


# -----------------------------
# Chat
# -----------------------------

if st.session_state.get("document_processed", False):

    st.divider()

    st.subheader("💬 Ask Questions")

    query = st.chat_input(
        "Ask something about your document..."
    )


    if query:

        # Show user message
        with st.chat_message("user"):
            st.write(query)


        # Retriever
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )


        with st.spinner("Searching the document..."):

            docs = retriever.invoke(query)


            # Create context
            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )


        # Prompt
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


        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )


        # Mistral
        llm = ChatMistralAI(
            model="mistral-medium-3-5",
            temperature=0.3
        )


        with st.chat_message("assistant"):

            with st.spinner("Generating answer..."):

                response = llm.invoke(final_prompt)

                st.write(response.content)
