📄 PDF & Website Chatbot (LangChain + FAISS + Streamlit)

A modern chatbot that lets you chat with your PDFs and websites.
This project uses LangChain, FAISS vector storage, and a free LLM API to make document/URL-based Q&A faster and cost-efficient.

🚀 Features

📂 Upload PDFs or scrape websites → chatbot ingests the content

🧩 Splits data into small chunks for efficient storage

⚡ Stores embeddings in FAISS for quick retrieval

💬 Query only the relevant chunks (reduces token usage)

🎯 Works with free LLM APIs for cost-efficient usage

🌐 Simple Streamlit UI → no setup needed for end-users

🛠️ How It Works

Input → User uploads PDFs or provides website URLs

Processing → Documents are split into chunks

Embedding → Chunks are converted to vectors using embeddings model

Storage → Saved in FAISS vector DB (faiss_index.pkl)

Querying → User asks a question → query is embedded → similar chunks retrieved

Response → Relevant context sent to LLM → chatbot replies

👤 Who Is This For?

🔍 Students/Researchers → Quickly extract key info from research papers or reports

💼 Professionals → Summarize company docs, contracts, or technical manuals

📰 Content Analysts → Scrape & analyze website/blog content interactively

💡 Why It’s Useful

✅ Saves time by answering from specific docs instead of full search

✅ Saves money by reducing API token usage (only relevant chunks sent)

✅ Works with both PDFs & URLs


-- [Live Demo link ](https://insightbot-srun8r9jphm2kpgdqopzgd.streamlit.app/)
