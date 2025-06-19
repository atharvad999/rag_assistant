import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from typing import List
import time

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="📚 RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client
@st.cache_resource
def init_openai_client():
    return OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

class CustomOpenAIEmbeddings:
    """Custom embedding class using native OpenAI client"""
    
    def __init__(self, model="text-embedding-ada-002"):
        self.model = model
        self.client = init_openai_client()
    
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

# Constants
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

@st.cache_resource
def load_vector_db():
    """Load the vector database"""
    try:
        embedding_function = CustomOpenAIEmbeddings(model="text-embedding-ada-002")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        return db
    except Exception as e:
        st.error(f"Error loading vector database: {str(e)}")
        return None

@st.cache_resource
def init_chat_model():
    """Initialize the chat model"""
    try:
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    except Exception as e:
        st.error(f"Error initializing chat model: {str(e)}")
        return None

def query_rag_system(query_text: str, k: int = 3, relevance_threshold: float = 0.7):
    """Query the RAG system and return results"""
    db = load_vector_db()
    if not db:
        return None, []
    
    # Search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=k)
    
    if len(results) == 0 or results[0][1] < relevance_threshold:
        return "No relevant documents found for your query.", []
    
    # Prepare context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Get response from the model
    model = init_chat_model()
    if not model:
        return None, []
    
    response_text = model.predict(prompt)
    
    # Extract sources
    sources = [(doc.metadata.get("source", "Unknown"), score) for doc, score in results]
    
    return response_text, sources, results

def main():
    # Header
    st.title("📚 RAG Document Assistant")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key status
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            st.success("✅ OpenAI API Key loaded")
        else:
            st.error("❌ OpenAI API Key not found")
            st.info("Please add your OpenAI API key to the .env file")
        
        st.markdown("---")
        
        # Search parameters
        st.subheader("Search Parameters")
        k = st.slider("Number of documents to retrieve", 1, 10, 3)
        relevance_threshold = st.slider("Relevance threshold", 0.0, 1.0, 0.7, 0.1)
        
        st.markdown("---")
        
        # Database status
        st.subheader("Database Status")
        if os.path.exists(CHROMA_PATH):
            st.success("✅ Vector database found")
            db = load_vector_db()
            if db:
                try:
                    # Try to get a count of documents
                    test_results = db.similarity_search("test", k=1)
                    st.info(f"📊 Database loaded successfully")
                except Exception as e:
                    st.warning(f"⚠️ Database loaded but may have issues: {str(e)}")
        else:
            st.error("❌ Vector database not found")
            st.info("Please run `python create_database.py` first")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask a Question")
        
        # Query input
        query_text = st.text_area(
            "Enter your question about the documents:",
            placeholder="e.g., What happens to Alice when she falls down the rabbit hole?",
            height=100
        )
        
        # Search button
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if search_button and query_text:
            with st.spinner("Searching documents and generating response..."):
                try:
                    # Query the system
                    response, sources, raw_results = query_rag_system(
                        query_text, k=k, relevance_threshold=relevance_threshold
                    )
                    
                    if response:
                        # Display response
                        st.markdown("---")
                        st.subheader("🤖 Response")
                        st.markdown(response)
                        
                        # Display sources
                        if sources:
                            st.markdown("---")
                            st.subheader("📖 Sources")
                            
                            for i, (source, score) in enumerate(sources, 1):
                                with st.expander(f"Source {i}: {os.path.basename(source)} (Relevance: {score:.2f})"):
                                    # Get the actual content
                                    if raw_results and i <= len(raw_results):
                                        doc_content = raw_results[i-1][0].page_content
                                        st.markdown(f"**File:** `{source}`")
                                        st.markdown("**Content:**")
                                        st.text(doc_content[:500] + "..." if len(doc_content) > 500 else doc_content)
                    else:
                        st.warning("No relevant documents found for your query. Try rephrasing your question.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Please check your OpenAI API key and vector database.")
    
    with col2:
        st.subheader("📋 Quick Info")
        
        # Info cards
        with st.container():
            st.markdown("""
            <div style="background-color: #0a1f44; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                <h4>🎯 How to use:</h4>
                <ol>
                    <li>Make sure your vector database is created</li>
                    <li>Enter your question in the text area</li>
                    <li>Click the Search button</li>
                    <li>Review the AI response and sources</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div style="background-color: #0a1f44; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                <h4>💡 Tips:</h4>
                <ul>
                    <li>Be specific in your questions</li>
                    <li>Adjust relevance threshold for broader/narrower results</li>
                    <li>Check sources for verification</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Sample queries
        st.subheader("🎭 Sample Queries")
        sample_queries = [
            "What happens to Alice when she falls down the rabbit hole?",
            "Who does Alice meet in Wonderland?",
            "What is the Mad Hatter's tea party like?",
            "How does Alice change size in the story?",
            "What is the Queen of Hearts known for?"
        ]
        
        for query in sample_queries:
            if st.button(f"💭 {query}", key=f"sample_{hash(query)}", use_container_width=True):
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            📚 RAG Document Assistant powered by LangChain & OpenAI
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 