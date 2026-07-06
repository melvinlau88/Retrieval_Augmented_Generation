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

# Load the Webpage https://en.wikipedia.org/wiki/Python_(programming_language)

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Python_(programming_language)")
docs = loader.load()

print(docs[0].page_content[:500])