import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the queries."),
    ("user", "Question: {question}")
])

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Load and process the document
loader = TextLoader("palki_sharma.txt")
doc = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20).split_documents(doc)

# Create the vector store
db = Chroma.from_documents(text_splitter, OpenAIEmbeddings())

# Define the output parser
output_parser =StrOutputParser()

# Define the chain
chain = prompt_template | llm | output_parser

# Streamlit app
st.title("LangChain Demo with OpenAI")
input_text = st.text_input("Search the topic you want")

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
