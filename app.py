import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader

# Section 1: Setup the local LLM and API key for Tavily search engine, set title and input for Streamlit
local_llm = "llama3"
try:
    tavily_api_key = os.environ['TAVILY_API_KEY']
except KeyError:
    st.error("TAVILY_API_KEY is not set. Please configure your environment variables.")
    st.stop()

st.title("Multi-PDF ChatBot using LLAMA3 & Adaptive RAG")
user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')

with st.sidebar:
    uploaded_files = st.file_uploader("Upload your file", type=['pdf'], accept_multiple_files=True)
    process = st.button("Process")

# Section 2: Process the PDF files upon button click
if process:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        st.stop()
    
    # Ensure the directory exists
    temp_dir = os.path.expanduser('~/adaptiverag/temp/')
    try:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        st.write(f"Directory ensured at: {temp_dir}")
    except Exception as e:
        st.error(f"Failed to ensure directory: {str(e)}")
        st.stop()

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        if uploaded_file.type != 'application/pdf':
            st.error("Only PDF files are supported. Please upload a PDF.")
            continue

        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        try:
            # Save the file to disk
            with open(temp_file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())
            
            # Load the PDF using PyPDFLoader and display the content
            try:
                loader = PyPDFLoader(temp_file_path)
                data = loader.load()
                st.write(f"Data loaded for {uploaded_file.name}")
            except Exception as e:
                st.error(f"Failed to load {uploaded_file.name}: {str(e)}")
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
