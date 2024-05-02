import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain import hub
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

# Section 1: Setup LLM, API key, title, and input
local_llm = "dolphin-llama3:8b"
llm = ChatOllama(model=local_llm, format="json", temperature=0)

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

# Helper functions
def create_vectorstore():
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    text_chunks = text_splitter.split_documents(data)
    return Chroma.from_documents(
            documents=text_chunks,
            collection_name="rag-chroma",
            embedding=GPT4AllEmbeddings()
        )

def get_retriever():
    if not hasattr(get_retriever, 'retriever'):
        get_retriever.vectorstore = create_vectorstore()  
        get_retriever.retriever = get_retriever.vectorstore.as_retriever()
    return get_retriever.retriever

def load_pdf_data(file_path):
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        st.error(f"Failed to load PDF: {e}")
        return None

# ... (Other prompts, graders, web_search_tool as before)

# Workflow components
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# ... (retrieve, generate, grade_documents, transform_query, web_search, route_question, decide_to_generate, grade_generation_v_documents_and_question as before) 

# Workflow setup
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search) 
workflow.add_node("retrieve", retrieve) 
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate) 
workflow.add_node("transform_query", transform_query) 

# Build graph (same as before) 

# Section 2: Process PDF files upon button click
if process:
    temp_dir = os.path.expanduser('~/adaptiverag/temp/')
    try:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        st.write(f"Directory ensured at: {temp_dir}")
    except Exception as e:
        st.error(f"Failed to ensure directory: {str(e)}")
        st.stop()

    for uploaded_file in uploaded_files:
        if uploaded_file.type != 'application/pdf':
            st.error("Only PDF files are supported.")
            continue

        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        try:
            with open(temp_file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            data = load_pdf_data(temp_file_path)
            if data:
                st.write(f"Data loaded for {uploaded_file.name}")

                # ... Initialize vectorstore (if needed), 
                #     retrieve documents, and start workflow
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")

#  Compile and run the workflow 
app = workflow.compile()
inputs = {"question": user_input}
for output in app.stream(inputs):

# Workflow setup (continued)
workflow = StateGraph(GraphState)

# ... workflow nodes and graph building (as provided before)

# Section 2: Process PDF files upon button click
if process:
    # ... (PDF processing logic as provided before)

        try:
            # ... Initialize vectorstore (if needed) and retrieve documents

            # Start the workflow
            retriever = get_retriever() 
            docs = retriever.get_relevant_documents(user_input)  # Update to use user_input
            workflow.start({"question": user_input, "documents": docs})

        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")

#  Compile and run the workflow 
app = workflow.compile()
# Remove the inputs dictionary if you want to start with a fresh input each time

for output in app.stream():  # Use app.stream() without inputs
    for key, value in output.items():
        st.write(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    print("\n---\n")

# Final generation
st.write(output.get("generation", "No generation produced."))  # Handle potential lack of generation
