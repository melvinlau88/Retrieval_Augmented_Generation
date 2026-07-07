r'''
cd C:\Users\melvi\Downloads\VS_Code\Python\Retrieval_Augmented_Generation
git add .
git commit -m "Changed embedding method so I don't have to pay OpenAI"
git push
'''

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.load import dumps, loads


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

# Loads an embedding model that converts text into vectors
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embeds every document chunk and stores both the text and vectors in a Chroma vector database
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# A retriever that searches the vector database for the most relevant document chunks
retriever = vectorstore.as_retriever()

'''Retreive and Generate'''


# Create LLM with no creativity, only facts
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt
    | llm 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)


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

ans = rag_chain.invoke("Who created python")
print("")
print(ans)
print("")

