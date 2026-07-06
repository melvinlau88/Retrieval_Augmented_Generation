r'''
cd C:\Users\melvi\Downloads\VS_Code\Python\Retrieval_Augmented_Generation
git add .
git commit -m "test"
git push
'''

from dotenv import load_dotenv
load_dotenv()

import bs4

from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
