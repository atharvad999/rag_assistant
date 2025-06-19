import argparse
import os
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

CHROMA_PATH = "chroma"


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


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB using native OpenAI client
    embedding_function = CustomOpenAIEmbeddings(model="text-embedding-ada-002")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
