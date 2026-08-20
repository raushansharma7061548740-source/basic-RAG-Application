from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorestore = Chroma(
    persist_directory="chroma-db",
    embedding_function = embeddings
)

retriever = vectorestore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm = ChatMistralAI(model = "mistral-medium-3-5")

#prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
               """you are a helpful AI assistant.
                use only the provided context to answer the question.
                if the answer is not presented in the context,
                say:"I could not find the answer in the document."
                """),
        (
            "human",
             """Context:
{context}

             question:
{question}
             """
        )
    ]
)

print("Rag system Created")

print("press 0 to exit")

while True:
    query = input("you :" )
    if query == 0:
        break

    docs = retriever.invoke(query)

    Context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context":Context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print(f"\n AI : {response.content}")