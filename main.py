from dotenv import load_dotenv

import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.load import dumps, loads
from operator import itemgetter
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_community.utils.math import cosine_similarity
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import requests
from langchain_core.documents import Document
import datetime
from typing import Optional

# Basemodel acts a validation library, description and docstring used as specification
# for for LLM 
class RouteQuery(BaseModel):
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(...)

# Form for query to fill
class MovieSearch(BaseModel):
    """Search over a database of movies."""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to movie overviews/descriptions.",
    )
    min_rating: Optional[float] = Field(
        None,
        description="Minimum rating filter, inclusive. Only use if explicitly specified.",
    )
    max_rating: Optional[float] = Field(
        None,
        description="Maximum rating filter, exclusive. Only use if explicitly specified.",
    )
    earliest_release_date: Optional[datetime.date] = Field(
        None,
        description="Earliest release date filter, inclusive. Only use if explicitly specified.",
    )
    latest_release_date: Optional[datetime.date] = Field(
        None,
        description="Latest release date filter, exclusive. Only use if explicitly specified.",
    )

    def print_items(self) -> None:
        for field in type(self).model_fields: 
            value = getattr(self, field)
            if value is not None:
                print(f"{field}: {value}")

def format_docs(docs):
    """Extracs and joins page contents from each document in one string"""
    return "\n\n".join(doc.page_content for doc in docs)

def get_unique_union(documents: list[list]):
    """Merge results from multiple queries into one deduplicated list of Documents.
    Documents aren't hashable, so we serialize each to a JSON string to dedupe via
    a set, then parse them back into Document objects."""
    
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

def choose_route(result):
    if "python_docs" in result.datasource.lower():
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        return "chain for js_docs"
    else:
        return "golang_docs"
    
def prompt_router(input):
    """Pick the prompt template's embedding most similar to the question,
    using cosine similarity"""
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]

    if most_similar == movie_recommendation_template:
        print("Using RECOMMENDATION")
    else:
        print("Using FACTS")

    return PromptTemplate.from_template(most_similar)

def retrieve_with_filters(search: MovieSearch):
    """Convert a MovieSearch object into a Chroma filter and run a filtered similarity search."""
    conditions = []

    if search.min_rating is not None:
        conditions.append({"rating": {"$gte": search.min_rating}})
    if search.max_rating is not None:
        conditions.append({"rating": {"$lte": search.max_rating}})
    if search.earliest_release_date is not None:
        conditions.append({"release_date": {"$gte": str(search.earliest_release_date)}})
    if search.latest_release_date is not None:
        conditions.append({"release_date": {"$lte": str(search.latest_release_date)}})

    if len(conditions) == 0:
        filter = None
    elif len(conditions) == 1:
        filter = conditions[0]
    else:
        filter = {"$and": conditions}

    return vectorstore.similarity_search(search.content_search, k=5, filter=filter)

load_dotenv()
tmdb_token = os.getenv("TMDB_API_TOKEN")
print("Key loaded:", tmdb_token is not None, "| length:", len(tmdb_token) if tmdb_token else 0)

headers = {
    "Authorization": f"Bearer {tmdb_token}",
    "accept": "application/json"
}

movies = []
for page in range(1, 11):   
    response = requests.get(
        "https://api.themoviedb.org/3/movie/popular",
        headers=headers,                             
        params={"language": "en-US", "page": page}   
    )
    response.raise_for_status()
    movies.extend(response.json()["results"])

docs = []
for movie in movies:
    raw_date = movie.get("release_date", "")
    date_int = int(raw_date.replace("-", "")) if raw_date else 0   # e.g. "2015-03-12" -> 20150312

    doc = Document(
        page_content=f"{movie['title']}: {movie['overview']}",
        metadata={
            "title": movie["title"],
            "release_date": date_int,          # ← now an int
            "rating": movie.get("vote_average", 0),
            "genre_ids": ",".join(str(i) for i in movie.get("genre_ids", [])),
        },
    )
    docs.append(doc)


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

structured_llm = llm.with_structured_output(RouteQuery)


system = """You are an expert at routing a user question to the appropriate data source.
Based on the programming language the question is referring to, route it to the relevant data source."""

# Uses LangChain prompt template to create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])

router = prompt | structured_llm

full_chain = router | RunnableLambda(choose_route)


template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_generate = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_generate
    | llm 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

retrieval_chain = generate_queries | retriever.map() | get_unique_union

# A separate, proper answer-generating prompt — this was missing entirely
answer_template = """Answer the following question based on this context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(answer_template)


# Uses Langchain Expression Language for input, process and and output a question
rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")
    }
    | prompt
    | llm
    | StrOutputParser()
    ) 
    

movie_facts_template = """You are a knowledgeable film database expert. \
You are great at answering factual questions about plot, cast, release dates, and ratings. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

movie_recommendation_template = """You are an enthusiastic film critic and recommender. \
You are great at suggesting movies based on mood, genre, and similarity to other films. \
You explain your reasoning for each recommendation.

Here is a question:
{query}"""

prompt_templates = [movie_facts_template, movie_recommendation_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)


routing_chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | llm                          
    | StrOutputParser()
)

# Template for the Query Analyser

query_system = """You are an expert at converting user questions into database queries.
You have access to a database of movies. Given a question, return a database query
optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""

query_prompt = ChatPromptTemplate.from_messages([
    ("system", query_system),
    ("human", "{question}"),
])

query_structured_llm = llm.with_structured_output(MovieSearch)
query_analyzer = query_prompt | query_structured_llm


ans = rag_chain.invoke({"question": "What are some popular action movies right now?"})
print("")
print(ans)
print("")

