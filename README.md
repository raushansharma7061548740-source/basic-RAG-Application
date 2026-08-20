# 📚 RAG Document Q&A

This is a simple **RAG (Retrieval-Augmented Generation) project** that lets you ask questions about your documents.

Instead of asking the AI to answer from its general knowledge, the system first searches the relevant information from the document and then gives that information to the AI to generate an answer.

## 🛠️ Built With

* **Python**
* **LangChain**
* **ChromaDB** – stores and searches document embeddings
* **HuggingFace Embeddings** – converts text into vectors
* **Mistral AI** – generates the final answer

## 🔄 How It Works

```text
Question
   ↓
Search relevant information
   ↓
ChromaDB
   ↓
Relevant document content
   ↓
Mistral AI
   ↓
Answer
```

I used **MMR retrieval** so the system can find relevant information while avoiding too much repetition.

## 🚀 Run It

Install the required packages:

```bash
pip install -r requirements.txt
```

Add your Mistral API key to `.env`:

```env
MISTRAL_API_KEY=your_api_key
```

Then run:

```bash
python app.py
```

## 💡 One Important Part

The AI is instructed to **only use the information retrieved from the document**.

If the answer isn't available, it says:

> "I could not find the answer in the document."

This helps reduce made-up answers (hallucinations).

## 🔮 What's Next?

I plan to improve this project by adding:

* 📄 PDF upload
* 🌐 Streamlit UI
* 💬 Chat history
* 📌 Source references
* 📚 Multiple document support

## 👨‍💻 About

Built by **Raushan Kumar** while learning and exploring **RAG, LLMs, LangChain, and Generative AI**.
