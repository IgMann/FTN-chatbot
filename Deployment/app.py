# Importing libraries
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pypdf
import PyPDF2
from transliterate import translit

import langchain
from langchain.llms import Replicate
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate

import replicate
from flask import Flask, request, jsonify, render_template

from config import *
from functions import *

app = Flask(__name__)

# Database loading
database = database_loading(embedding_model=EMBEDDING_MODEL, device=DEVICE, database_path=DATABASE_PATH)

# API activation 
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# Chatbot creation
chatbot = Replicate(
    model=CHATBOT_MODEL,
    model_kwargs={"temperature": TEMPERATURE, "top_p": TOP_PERCENT, "max_new_tokens": MAX_NEW_TOKENS}
)

# QA chain creation
qa_chain = ConversationalRetrievalChain.from_llm(chatbot, database.as_retriever(search_kwargs={"k": 5}),
                                                 return_source_documents=True)

# Chat history initialization
chat_history = CHAT_HISTORY


@app.route('/')
def index():
    return render_template('index.html')


# API for question answering
@app.route("/sendMessage", methods=["POST"])
def process_question():
    question = request.form['user_input'].strip()
    question_converted = convert_to_latin(question)
    output = qa_chain({"question": question_converted, "chat_history": chat_history})
    answer = output["answer"]
    source_documents = output["source_documents"]

    faithfulness_score, faithfulness_justification = faithfulness_estimation(question_converted, answer,
                                                                             source_documents, CHATBOT_MODEL)

    # Depending of faithfulness score we return real or default answer
    if faithfulness_score >= FAITHFULNESS_LIMIT:
        answer = translator(answer, CHATBOT_MODEL)
        chat_history.append((question_converted, answer))
    else:
        answer = DEFAULT_ANSWER
        source_documents = (source_documents, faithfulness_justification)

    return render_template('index.html', result=answer, history=chat_history, faithfulness_score=faithfulness_score)


# API for database reloading
@app.route("/reload_base", methods=["GET"])
def reload_base():
    # Database loading
    database = database_loading(embedding_model=EMBEDDING_MODEL, device=DEVICE, database_path=DATABASE_PATH)

    return jsonify({"message": "Base data reloaded successfully"})


if __name__ == "__main__":
    app.run(debug=True)
