# https://youtu.be/Dh0sWMQzNH4
# my_pdf_processor.py
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import docx

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()

def read_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def process_pdf_query(pdf_path, query):
    text = read_pdf(pdf_path)

    # split into chunks
    char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, 
                                          chunk_overlap=200, length_function=len)

    text_chunks = char_text_splitter.split_text(text)
    
    # create embeddings
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(text_chunks, embeddings)
    
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")


    # process user query
    docs = docsearch.similarity_search(query)
    
    response = chain.run(input_documents=docs, question=query)
    return response
