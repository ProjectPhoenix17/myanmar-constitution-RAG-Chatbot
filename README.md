# 🇲🇲 Myanmar Constitution RAG Chatbot

A Domain-Specific Intelligent Chatbot using Generative AI and Retrieval-Augmented Generation (RAG).

---

## 📖 Overview

This project is a domain-specific intelligent chatbot developed to answer questions related to the 2008 Constitution of the Republic of the Union of Myanmar.

The chatbot combines Retrieval-Augmented Generation (RAG) with a fine-tuned MyanmarGPT (GPT-2) language model to generate accurate and context-aware responses.

This research is part of my PhD study entitled:

> Design and Implementation of a Domain-Specific Intelligent Chatbot Using Generative AI

---

## ✨ Features

- Constitution Question Answering
- Exact Keyword Matching
- Semantic Search using LaBSE
- FAISS Vector Database
- Hybrid Retrieval
- Fine-tuned MyanmarGPT Language Model
- Streamlit Web Interface

---

## 🛠 Technologies Used

- Python 3.10
- Streamlit
- PyTorch
- Transformers (Hugging Face)
- FAISS
- Sentence Transformers (LaBSE)
- Pandas
- NumPy

---

## 📂 Project Structure

RAG_Chatbot/
│
├── app.py
├── llm_model.py
├── labse.py
├── hybrid_search.py
├── backend_exact_match.py
├── build_index.py
├── README.md
├── requirements.txt
├── .gitignore
└── data/
---

## ⚙ Installation

Clone the repository

git clone https://github.com/ProjectPhoenix17/myanmar-constitution-RAG-Chatbot.git
Go to project folder

cd myanmar-constitution-RAG-Chatbot
Install dependencies

pip install -r requirements.txt
---

## ▶ Usage

Run the Streamlit application

streamlit run app.py
---


## 🧠 System Architecture

User Question

↓

Hybrid Search

↓

Exact Match + Semantic Search (LaBSE)

↓

FAISS Retrieval

↓

Fine-tuned MyanmarGPT

↓

Generated Answer

---

## 📊 Dataset

- Myanmar 2008 Constitution
- CSV Knowledge Base
- FAISS Vector Database

---

## 🔬 Research Contributions

- Domain-Specific Generative AI Chatbot
- Hybrid Retrieval Framework
- Fine-tuned MyanmarGPT Model
- Constitution Question Answering System

---


## 👨‍💻 Author

Project Phoenix

PhD Research Student

Research Area:
Generative AI | Large Language Models | Retrieval-Augmented Generation | Natural Language Processing

---

## 📄 License

This project is developed for research and educational purposes.