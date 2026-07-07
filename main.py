r'''
cd C:\Users\melvi\Downloads\VS_Code\Python\Retrieval_Augmented_Generation
git add .
git commit -m "Changed embedding method so I don't have to pay OpenAI"
git push
'''

from dotenv import load_dotenv
load_dotenv()

import bs4

from langchain_classic import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
from langchain_groq import ChatGroq

# Extracs and joins page contents from each document in one string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

'''Load the Webpage'''
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Python_(programming_language)")
docs = loader.load()

'''Split the Text'''
# Break text into chunks of 1,000 characters, but allow adjacent chunks to share 200 characters of overlap so context isn't lost."
text_split = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
splits = text_split.split_documents(docs)

'''Embed and Store Tokens'''
# Convert text splits to vectors and save them in a Chroma database
'''Embed and Store Tokens'''
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

'''Retreive and Generate'''
# Use prompt template
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:"""
)

# Create LLM with no creativity, only facts
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Uses Langchain Expression Language for input, process and and output a question
rag_chain = (
    {
    "context": retriever | format_docs, 
    "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("Who created python")