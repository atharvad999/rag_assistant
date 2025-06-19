# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from dotenv import load_dotenv
import os
import shutil
import nltk
from typing import List

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"


class CustomOpenAIEmbeddings:
    """Custom embedding class using native OpenAI client"""
    
    def __init__(self, model="text-embedding-ada-002"):
        self.model = model
        self.client = client
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )
        return [embedding.embedding for embedding in response.data]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            encoding_format="float"
        )
        return response.data[0].embedding


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    print("Loaded documents:", [doc.metadata.get('source', 'unknown') for doc in documents])
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    if len(chunks) > 10:
        document = chunks[10]
        print("Sample chunk:", document.page_content)
        print("From source:", document.metadata.get('source', 'unknown'))
    else:
        print("Less than 11 chunks were created.")
    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents using native OpenAI client
    embeddings = CustomOpenAIEmbeddings(model="text-embedding-ada-002")
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
