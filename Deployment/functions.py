# Importing libraries
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import replicate

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


# Database loading function
def database_loading(embedding_model, device, database_path):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": device})
    database = FAISS.load_local(database_path, embeddings)

    return database

# Script converting function
def convert_to_latin(sentence):
    latin_sentence = translit(sentence, 'sr', reversed=True)

    return latin_sentence

# Translation function
def translator(prompt, chatbot):
    output = replicate.run(
      chatbot,
      input={"system_prompt": "Prevesti prompt na srpski jezik ako već nije na srpskom i skratiti ga na jednu rečenicu, bez dodatnog teksta.",
              "prompt": prompt,
              "max_new_tokens":500}
    )

    return convert_to_latin("".join(output).split("\n")[-1])

# Faithfulness estimation function
def faithfulness_estimation(question, answer, sources, chatbot):
    prompt = f"""
      You must return the following fields in your response in two lines, one below the other:
      score: Your numerical score for the model's faithfulness based on the rubric
      justification: Your reasoning about the model's faithfulness score

      You are an impartial judge. You will be given an input that was sent to a machine
      learning model, and you will be given an output that the model produced. You
      may also be given additional information that was used by the model to generate the output.

      Your task is to determine a numerical score called faithfulness based on the input which is question and output which contain answer and sources.
      A definition of faithfulness and a grading rubric are provided below.
      You must use the grading rubric to determine your score. You must also justify your score.

      Examples could be included below for reference. Make sure to use them as references and to
      understand them before completing the task.

      Input question:
      {question}

      Output (answer):
      {answer}

      Output (sources):
      {sources}

      Metric definition:
      Faithfulness is only evaluated with the provided output and provided context, please ignore the provided input entirely when scoring faithfulness. Faithfulness assesses how much of the provided output is factually consistent with the provided context. A higher score indicates that a higher proportion of claims present in the output can be derived from the provided context. Faithfulness does not consider how much extra information from the context is not present in the output.

      Grading rubric:
      Faithfulness: Below are the details for different scores:
      - Score 1: None of the claims in the output can be inferred from the provided context.
      - Score 2: Some of the claims in the output can be inferred from the provided context, but the majority of the output is missing from, inconsistent with, or contradictory to the provided context.
      - Score 3: Half or more of the claims in the output can be inferred from the provided context.
      - Score 4: Most of the claims in the output can be inferred from the provided context, with very little information that is not directly supported by the provided context.
      - Score 5: All of the claims in the output are directly supported by the provided context, demonstrating high faithfulness to the provided context.

      Examples:

      1. example
        Example answer:
        Cena jednog boda za predmete koji se prenose u narednu godinu dobija se kada se cena skolarine podeli sa 180.

        Additional information used by the model (sources):
        The university's policy states that the cost per point for subjects carried over is calculated by dividing the total tuition fee by 180.

        Example score: 5
        Example justification: The output precisely matches the university's documented policy on the cost per point for subjects carried over to the next year.

      2. example
        Example answer:
        Za skolsku 2023/24. godinu, cena boda za predmete koji se prenose u narednu godinu iznosi 37 eura po bodu, dok je najvisa moguca cena za upis 90 eura po bodu, uz mogucnost da se ova cena poveca do maksimuma od 90 eura po bodu u zavisnosti od politike fakulteta.​

        Additional information used by the model (sources):
        The document specifies a method for calculating the cost per point for carried-over subjects, which does not align with the detailed prices provided in the output.

        Example score: 1
        Example justification: The output introduces specific cost figures and policies not mentioned in the original document, potentially leading to confusion or misinformation.


      You must return the following fields in your response in two lines, one below the other:
      score: Your numerical score for the model's faithfulness based on the rubric
      justification: Your reasoning about the model's faithfulness score

      Do not add additional new lines. Do not add any other fields.
    """

    output = replicate.run(
      chatbot,
      input={"system_prompt": "Calculate faithfulness score based on prompt and justify it.",
              "prompt": prompt,
              "max_new_tokens":500}
    )

    joined_output = ''.join(output)
    splited_output = joined_output.split("\n")

    score = np.nan
    justification = ''

    for line in splited_output:
        if len(line) > 5:
            words = line.split()
            words[0] = words[0].strip()
            processed_line = ' '.join(words)
            if "score" in words[0].lower():
                try:
                    score = eval(words[-1].strip())
                except:
                    raise ValueError(f"Score not found in line: {line}")
            elif "justification" in words[0].lower():
                justification = ' '.join(words[1:])

    return score, justification
