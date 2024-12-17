import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import requests

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="PDF Query API", version="1.0")

# Load your Google API Key from .env
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is missing. Set it in the environment variables.")

PDF_URL = "https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"

# Download the PDF
response = requests.get(PDF_URL)

# Save the PDF to the local file system
with open("ThaiRecipes.pdf", "wb") as f:
    f.write(response.content)

# Load the downloaded PDF
loader = PyPDFLoader("ThaiRecipes.pdf")
data = loader.load()

print("Loading and processing PDF...")

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Create a vector store
vectorstore = Chroma.from_documents(
    documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)

# Create a retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", 
    temperature=0, 
    max_tokens=None, 
    timeout=60  # Timeout in seconds
)

# Define system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise. Don't add your own knowledge base, and answer using a single line. "
    "If the question is out of context, tell that you can't answer this question."
    "\n\n"
    "{context}"
)

# Create the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create document chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Define request schema
class QueryRequest(BaseModel):
    query: str

# Define API endpoint
@app.post("/query", summary="Query the PDF Document", description="Provide a query to retrieve and answer questions from the PDF.")
async def query_pdf(request: QueryRequest):
    query = request.query

    if not query or query.strip() == "":
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Process the query using RAG chain
    try:
        response = rag_chain.invoke({"input": query})
        answer = response.get("answer", "I'm sorry, I couldn't generate an answer.")

        return {"query": query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")

# Root endpoint
@app.get("/", summary="Root Endpoint", description="Basic health check endpoint for the API.")
def root():
    return {"message": "Welcome to the PDF Query API powered by FastAPI and Gemini LLM!"}
