# Importing Libraries
import os
from dotenv import load_dotenv

import streamlit as st

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate 
from typing_extensions import Concatenate
import chromadb

# Loading the environment variables
load_dotenv()


# Loading PDF -> Splitting the text -> Generating embeddings -> Storing in ChromaDB -> Returning VectorDB
def generate_embeddings() -> Chroma:
    
    # providing path of the PDF
    pdf = PdfReader("/Users/sohal/Downloads/SwiftBot/Celebrity Music and Public Persona_ A Case Study of Taylor Swif.pdf")

    # reading and extracting text from the PDF
    raw_text = ''
    for i, page in enumerate(pdf.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    
    # splitting the text
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 750,
        chunk_overlap  = 50,
        length_function = len,
        )
    texts = text_splitter.split_text(raw_text)

    # generating and storing embeddings
    client = chromadb.Client()
    if client.list_collections():
        embedding_collection = client.create_collection("embedding_collection")
    else:
        print("Collection exists")
    persist_directory = '/Users/sohal/Downloads/SwiftBot/chroma'
    vectordb = Chroma.from_texts(
        texts = texts,
        embedding = OpenAIEmbeddings(),
        persist_directory = persist_directory
    )
    vectordb.persist()
    return vectordb


# Function to generate response
def generate_response(question):
    
    # LLM model
    llm = OpenAI(temperature=0)
    
    vectordb = generate_embeddings()

    # Prompt template
    template = """As an AI assistant you provide answers based on the given context, ensuring accuracy and briefness. 

        You always follow these guidelines:

        -Only answer from the source documents
        -If the answer isn't available within the context, respond that it is not available
        -Otherwise, answer to your best capability, referring to source documents provided
        -Only use examples if explicitly requested
        -Do not introduce examples outside of the context
        -Do not answer if context is absent
        -Limit responses to three or four sentences for clarity and conciseness
        Question: {question}
        Context: {context}
        Answer:"""
    
    qa_prompt = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.6}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )

    response = qa_chain({"query": question})
    return response


# App framework
st.set_page_config(page_title="SwiftBot", page_icon=":robot:")
st.header("🤖 SwiftBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:= st.chat_input("Enter your query"):
    user_message = st.chat_message("user")
    user_message.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner('Generating...'):
        response = generate_response(prompt)
        bot_message = st.chat_message("assistant")
        if(len(response["source_documents"])>0):
            result = response["result"]
            source_text = response["source_documents"][0]
            answer = f"**Response** : {result}<br><br>**Source Text** : {source_text}"
            bot_message.markdown(answer, unsafe_allow_html=True)
        else:
            answer = f"Sorry I can't answer the following question based on the context provided."
            bot_message.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
