r'''
cd C:\Users\melvi\Downloads\VS_Code\Python\Retrieval_Augmented_Generation
git add .
git commit -m "test"
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

'''Load the Webpage'''
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Python_(programming_language)")
docs = loader.load()

'''Split the Text'''
# Break text into chunks of 1,000 characters, but allow adjacent chunks to share 200 characters of overlap so context isn't lost."
text_split = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
splits = text_split.split_documents(docs)

'''Embed and Store Tokens'''
# Convert text splits to vectors and save them in a Chroma database
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

'''Retreive and Generate'''
# Use prompt template
prompt = hub.pull("rlm/rag-prompt")

# Create LLM with No creativity, only facts
llm = ChatOpenAI(model_name="The Chatter Guy", temperature=0)

