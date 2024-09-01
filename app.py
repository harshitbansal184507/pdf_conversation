from langchain_community.document_loaders import PyPDFLoader
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import getpass
import os 


loader = PyPDFLoader("sample_pdfs/DAA_synopsis.pdf")
docs = loader.load()


text_splitter= CharacterTextSplitter(
    separator = "\n" , 
    chunk_size = 2000 , 
    chunk_overlap= 200, 
)

texts= text_splitter.split_documents(docs)


embeddings= HuggingFaceEmbeddings()

DB=Chroma.from_documents(texts, embeddings)

llm = Ollama(model="llama3")
chain = RetrievalQA.from_chain_type(llm ,retriever = DB.as_retriever() )
query1 = "who submitted the synopsis "
result = chain.invoke({"query":query1})
print(result)
