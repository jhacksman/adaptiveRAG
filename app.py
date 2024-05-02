from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import os

local_llm = "llama3"
tavily_api_key = os.environ['TAVILY_API_KEY'] = 'API_KEY'
st.title("Multi-PDF ChatBot using LLAMA3 & Adaptive RAG")
user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')

with st.sidebar:
    uploaded_files = st.file_uploader("Upload your file", type=['pdf'], accept_multiple_files=True)
    process = st.button("Process")
if process:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        st.stop()

# Ensure the temp directory exists
    temp_dir = 'C:/temp/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save the file to disk
        with open(temp_file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())  # Use getbuffer() for Streamlit's UploadedFile
        
        # Load the PDF using PyPDFLoader
        try:
            loader = PyPDFLoader(temp_file_path)
            data = loader.load()  # Assuming loader.load() is the correct method call
            st.write(f"Data loaded for {uploaded_file.name}")
        except Exception as e:
            st.error(f"Failed to load {uploaded_file.name}: {str(e)}")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
text_chunks = text_splitter.split_documents(data)

    # Add to vectorDB
vectorstore = Chroma.from_documents(
        documents=text_chunks,
        collection_name="rag-chroma",
        embedding=GPT4AllEmbeddings(),
    )
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

 prompt = PromptTemplate(
        template="""You are an expert at routing a user question to a vectorstore or web search. \n
        Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks. \n
        You do not need to be stringent with the keywords in the question related to these topics. \n
        Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
        Return the a JSON with a single key 'datasource' and no premable or explaination. \n
        Question to route: {question}""",
        input_variables=["question"],
)

question_router = prompt | llm | JsonOutputParser()
question = "llm agent memory"
docs = retriever.get_relevant_documents(question)
doc_txt = docs[1].page_content
question_router.invoke({"question": question})
llm = ChatOllama(model=local_llm, format="json", temperature=0)


Home Blog
LangGraph + Adaptive Rag + LLama3 Python Project: Easy AI/Chat for your Docs
Gao Dalie (高達烈)	by Gao Dalie (高達烈)	
April 30, 2024
in Blog	
0
adaptive rag
0
SHARES
69
VIEWS
Share on Facebook
Share on Twitter

in this post, I have a super quick tutorial for you showing how to create a fully local chatbot with LangGraph, Adaptive Rag and LLama3 to make a powerful Agent Chatbot for your business or personal use.

Adaptive RAG is a cool paper that dynamically selects the best RAG strategy based on query complexity.

and Llama 3 is the latest model in the Llama series published by Meta and is designed to be the best open-source model with performance comparable to the best-closed models currently available.

In this post, we’re going to look closely at what adaptive Rag, how the adaptive Retrieval Augmented Generation process works, and how Llama 3 7B and 70B stack up against other models working in “instruct” mode.

Table of Contents

    Before we start! 🦸🏻‍♀️
    What is Adaptive RAG :
    How does Adaptive RAG work?
    how does Llama 3 7B and 70B stack up against other models working in “instruct” mode?
        Let’s Start Coding

Before we start! 🦸🏻‍♀️

If you like this topic and you want to support me:

    Clap my article 50 times; that will really help me out.👏
    Follow me on Medium and subscribe to get my latest article🫶
    Follow me on my YouTube channel

What is Adaptive RAG :

Adaptive Rag is introduced as a novel framework that employs a classifier to dynamically select the most appropriate strategy for handling queries based on their complexity. this adaptive approach tailors the retrieval process to the specific needs of each query, balancing computational efficiency with accuracy.
How does Adaptive RAG work?

The Adaptive Rag framework employs a classifier to dynamically choose the best strategy for Large Language Models based on query complexity. This process begins with a smaller model trained to classify queries into different complexity levels using automatically annotated datasets. These datasets are created by combining predicted outcomes from different models and inherent biases found in existing data.

Once the classifier predicts the complexity of an incoming query, the Adaptive-RAG framework determines whether to use iterative retrieval, single-step retrieval, or non-retrieval LLMs to provide an answer.

This dynamic selection approach improves efficiency by assigning more resources for complex queries and enhances accuracy by matching the Best strategy to each task.

The framework can decide the most effective processing strategy by allocating a complexity label to each query. This adaptability allows for a more flexible system, offering better performance than rigid, one-size-fits-all approaches.

The result is a more efficient and responsive Question-Answer framework, capable of handling a wide range of query complexities with precision and speed.
how does Llama 3 7B and 70B stack up against other models working in “instruct” mode?

Meta developed great language models to innovate, extend, and optimize for simplicity by focusing on four elements:

‘model architecture,’ ‘pre-training data,’ ‘scaling up pre-training,’ and ‘fine-tuning instructions.’

Llama 3 uses a relatively standard decoder-only transformer architecture as its language model. Although not revolutionary, it employs a tokenizer with a vocabulary of 128,000 tokens, allowing it to encode language more efficiently, significantly improving its performance compared to Llama 2. It also uses grouped query attention (GQA) across 8B and 70B sizes to improve inference efficiency in Llama 3.

Meta has invested heavily in pre-training data for Llama 3, using over 15 trillion tokens, all collected from public sources. This is about seven times larger than the Llama 2 training data and contains about four times more code. Meta has developed efficient data usage and optimal training strategies to scale up the pre-training of Llama 3 models.

During this process, detailed scaling laws were established to predict model performance and optimize computing resources. For example, an 8B parameter model requires an optimal training complexity of approximately 200 billion tokens. Still, it has been found that further improvement can be seen by training up to 15 trillion tokens.

An innovative approach to instruction tuning was introduced to fine-tune the pre-trained model specifically for the chat use case. This approach combines supervised fine-tuning (SFT), rejection sampling, proximity policy optimization (PPO), and direct policy optimization (DPO).

By learning priority rankings via PPO and DPO, Meta can better choose how to generate answers, significantly improving performance in inference and coding tasks.
Let’s Start Coding

Before we can work with langGraph, Adaptive Rag and perform actions on your text data. we must import various libraries and packages. Here’s a list of the libraries and their purposes:

        Langchain: This is the main library that provides access to Langchain functionalities.

    LangChain_Community contains third-party integrations that implement the base interfaces defined in LangChain Core,
    langchain_core: compiles LCEL sequences to an optimized execution plan, with automatic parallelization, streaming, tracing, and async support
    Chroma: Part of the Vector store used for storing text embeddings.
    LangGraph: an alpha-stage library for building stateful, multi-actor applications with LLMs
    Streamlit: lets you transform Python scripts into interactive web apps in minutes.
    gpt4all: an ecosystem to train and deploy powerful and customized large language models that run locally on consumer-grade CPUs
    tavily-python: Search API is a search engine optimized for LLMs and RAG
    TextSplitter: A tool to split large documents into smaller, more manageable chunks.
    Ollama: allows you to run open-source large language models, such as Llama 3 locally.

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import os

We set up a variable named local_llm and assigned it the value ‘llama3’. Then, we set an environment variable Tavily API with an API key.

We use Streamlit’s function st.title to set the title of the web page. Afterwards, we created a text input field on the web page where users can enter a question. Additionally, we added a file uploader sidebar.

 Inside the sidebar, this line adds a file uploader tool, set to accept only PDF files. Finally, we added a button labelled ‘Process’ to process the uploaded PDF files.

local_llm = "llama3"
tavily_api_key = os.environ['TAVILY_API_KEY'] = 'API_KEY'
st.title("Multi-PDF ChatBot using LLAMA3 & Adaptive RAG")
user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')

with st.sidebar:
    uploaded_files = st.file_uploader("Upload your file", type=['pdf'], accept_multiple_files=True)
    process = st.button("Process")
if process:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        st.stop()

We set up a variable named temp_dir and assign it the path of a directory on the computer where temporary files will be stored. Then, we check if the directory specified by temp_dir exists on the computer. If the directory does not exist, this function creates it.

Next, we start a loop that will go through each file uploaded by the user. For each file, we construct the full path where the uploaded file will be saved by joining the temporary directory path and the file’s name. We then open a file at the path specified by temp_file_path and write the content of the uploaded file to the disk.

 Afterwards, we initialize a new instance of PyPDFLoader with the path to the saved file. Finally, we use the loader to read the PDF file and store its content in the variable ‘Data’.

# Ensure the temp directory exists
    temp_dir = 'C:/temp/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save the file to disk
        with open(temp_file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())  # Use getbuffer() for Streamlit's UploadedFile
        
        # Load the PDF using PyPDFLoader
        try:
            loader = PyPDFLoader(temp_file_path)
            data = loader.load()  # Assuming loader.load() is the correct method call
            st.write(f"Data loaded for {uploaded_file.name}")
        except Exception as e:
            st.error(f"Failed to load {uploaded_file.name}: {str(e)}")

We create a RecursiveCharacterTextSplitter instance, configuring it with a chunk_size of 250 and a chunk_overlap value of zero. We will utilize the split_text method, which requires a string input representing the text and returns an array of strings, each representing a chunk after the splitting process. Now that we have the data chunks, let’s store them in our Vector Database. I am using the GPT4AllEmbeddings; feel free to use your preference.

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    text_chunks = text_splitter.split_documents(data)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        collection_name="rag-chroma",
        embedding=GPT4AllEmbeddings(),
    )
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

We use PromptTemplate to create a template for a string prompt. that instructs an expert system on how to decide whether a user’s question should be directed to a vectorstore or a web search. Then, we set up a pipeline that uses the previously defined prompt as input, processes it through an unspecified LLM, and defines a sample question about LLM agent memory. Finally, the pipeline extracts the content of the second retrieved document

 prompt = PromptTemplate(
        template="""You are an expert at routing a user question to a vectorstore or web search. \n
        Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks. \n
        You do not need to be stringent with the keywords in the question related to these topics. \n
        Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
        Return the a JSON with a single key 'datasource' and no premable or explaination. \n
        Question to route: {question}""",
        input_variables=["question"],
)

question_router = prompt | llm | JsonOutputParser()
question = "llm agent memory"
docs = retriever.get_relevant_documents(question)
doc_txt = docs[1].page_content
question_router.invoke({"question": question})
llm = ChatOllama(model=local_llm, format="json", temperature=0)

Also, we use PromptTemplate for grading the relevance of a document about a user’s question. to determine whether the document contains keywords related to the question and to provide a binary score (‘yes’ or ‘no’) indicating relevance, which is returned in a simple JSON format with the key ‘score’.

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
question = "agent memory"
docs = retriever.get_relevant_documents(question)
doc_txt = docs[1].page_content
st.write(retrieval_grader.invoke({"question": question, "document": doc_txt}))

### Generate
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOllama(model=local_llm, temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
question = "agent memory"
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

### Hallucination Grader 
# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Prompt
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
hallucination_grader.invoke({"documents": docs, "generation": generation})

### Answer Grader 

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Prompt
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
answer_grader.invoke({"question": question,"generation": generation})

### Question Re-writer

# LLM
llm = ChatOllama(model=local_llm, temperature=0)

# Prompt 
re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
    input_variables=["generation", "question"],
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})

web_search_tool = TavilySearchResults(k=3,tavily_api_key=tavily_api_key)

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents 
    """
    question : str
    generation : str
    documents : List[str]

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}

### Edges ###

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})  
    print(source)
    print(source['datasource'])
    if source['datasource'] == 'web_search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score['score']

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score['score']
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search) # web search
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("grade_documents", grade_documents) # grade documents
workflow.add_node("generate", generate) # generatae
workflow.add_node("transform_query", transform_query) # transform_query

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()

inputs = {"question": user_input}
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            st.write(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        print("\n---\n")

    # Final generation
    st.write(value["generation"])

