import streamlit as st

import os

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import GPT4AllEmbeddings

from langchain.prompts import PromptTemplate

from langchain_community.chat_models import ChatOllama

from langchain_core.output_parsers import JsonOutputParser

from langchain import hub

from langchain.schema import Document

from langgraph.graph import END, StateGraph

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain.schema import Document

from langchain_community.document_loaders import PyPDFLoader

from ollama import Ollama 


# Setup and Configuration

local_llm = "llama3"

tavily_api_key = os.environ['TAVILY_API_KEY'] = 'API_KEY' 

st.title("Multi-PDF ChatBot using LLAMA3 & Adaptive RAG")

user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')


# Sidebar for Uploads

with st.sidebar:

   uploaded_files = st.file_uploader("Upload your file", type=['pdf'], accept_multiple_files=True)

   process = st.button("Process")


# Temp Directory Handling

temp_dir = 'C:/temp/'

if not os.path.exists(temp_dir):

   os.makedirs(temp_dir)


# Process Uploaded PDFs

if process:

   if not uploaded_files:

       st.warning("Please upload at least one PDF file.")

       st.stop()


   for uploaded_file in uploaded_files:

       temp_file_path = os.path.join(temp_dir, uploaded_file.name)


       with open(temp_file_path, "wb") as file:

           file.write(uploaded_file.getbuffer()) 


       try:

           loader = PyPDFLoader(temp_file_path)

           data = loader.load() 

           st.write(f"Data loaded for {uploaded_file.name}")

       except Exception as e:

           st.error(f"Failed to load {uploaded_file.name}: {str(e)}")


# Data Preparation

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)

text_chunks = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(

       documents=text_chunks,

       collection_name="rag-chroma",

       embedding=GPT4AllEmbeddings(),

   )

retriever = vectorstore.as_retriever()


# Change LLM initialization to point to Ollama on LAN

llm = Ollama(model="llama3", url="http://192.168.0.134:5000") 


# Prompt Templates

prompt = PromptTemplate(

       template="""You are an expert at routing a user question to a vectorstore or web search. \n

       Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks. \n

       You do not need to be stringent with the keywords in the question related to these topics. \n

       Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n

       Return the a JSON with a single key 'datasource' and no premable or explaination. \n

       Question to route: {question}""",

       input_variables=["question"],

)

question_router = prompt | llm | JsonOutputParser()

# Prompt for grading document relevance

prompt = PromptTemplate(

       template="""You are a grader assessing relevance of a retrieved document to a user question. \n 

       Here is the retrieved document: \n\n {document} \n\n

       Here is the user question: {question} \n

       If the document contains keywords related to the user question, grade it as relevant. \n

       It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n

       Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n

       Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",

       input_variables=["question", "document"],

   )

retrieval_grader = prompt | llm | JsonOutputParser()


# Prompt for assessing answer quality (hallucination)

prompt = PromptTemplate(

   template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 

   Here are the facts:

   \n ------- \n

   {documents} 

   \n ------- \n

   Here is the answer: {generation}

   Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n

   Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",

   input_variables=["generation", "documents"],

)

hallucination_grader = prompt | llm | JsonOutputParser()


# Prompt for assessing answer usefulness

prompt = PromptTemplate(

   template="""You are a grader assessing whether an answer is useful to resolve a question. \n 

   Here is the answer:

   \n ------- \n

   {generation} 

   \n ------- \n

   Here is the question: {question}

   Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n

   Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",

   input_variables=["generation", "question"],

)

answer_grader = prompt | llm | JsonOutputParser()


# Prompt for re-writing questions

re_write_prompt = PromptTemplate(

   template="""You a question re-writer that converts an input question to a better version that is optimized \n 

    for vectorstore retrieval. Look at the initial and formulate an improved question. \n

    Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,

   input_variables=["generation", "question"],

)

question_rewriter = re_write_prompt | llm | StrOutputParser()

