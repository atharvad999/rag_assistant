# 📚 LangChain RAG Tutorial - Document Assistant

A Retrieval-Augmented Generation (RAG) system built with LangChain, OpenAI, and Streamlit that allows you to ask questions about your documents and get AI-powered answers with source citations.

## 🚀 Features

- **📖 Document Processing**: Automatically processes markdown documents and creates vector embeddings
- **🔍 Intelligent Search**: Uses semantic search to find relevant document sections
- **🤖 AI-Powered Responses**: Generates contextual answers using OpenAI's GPT models
- **📋 Source Citations**: Shows exactly which documents and sections were used for each answer
- **🎨 Beautiful Web UI**: Interactive Streamlit interface with real-time search
- **⚙️ Configurable Parameters**: Adjustable relevance thresholds and result counts
- **🔧 Modern Architecture**: Uses latest OpenAI API with native client integration

## 🛠️ Tech Stack

- **LangChain** - Document processing and RAG orchestration
- **OpenAI API** - Embeddings and language model
- **ChromaDB** - Vector database for storing embeddings
- **Streamlit** - Web user interface
- **Python 3.11+** - Core programming language

## 📋 Prerequisites

- Python 3.11 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Git (for cloning the repository)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd langchain-rag-tutorial-main
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install "unstructured[md]"  # For markdown processing
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Add Your Documents
Place your markdown files in the `data/books/` directory. The repository includes `alice_in_wonderland.md` as an example.

### 6. Create Vector Database
```bash
python create_database.py
```

### 7. Run the Application

#### Option A: Streamlit Web UI (Recommended)
```bash
python -m streamlit run streamlit_app.py
```
Then open http://localhost:8501 in your browser.

#### Option B: Command Line Interface
```bash
python query_data.py "What happens to Alice when she falls down the rabbit hole?"
```

## 📁 Project Structure

```
langchain-rag-tutorial-main/
├── data/
│   └── books/
│       └── alice_in_wonderland.md    # Sample document
├── venv/                             # Virtual environment
├── chroma/                           # Vector database (created after running create_database.py)
├── create_database.py                # Document processing and database creation
├── query_data.py                     # Command-line query interface
├── streamlit_app.py                  # Web UI application
├── compare_embeddings.py             # Embedding comparison utilities
├── requirements.txt                  # Python dependencies
├── .env                             # Environment variables (create this)
└── README.md                        # This file
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Embedding Models
The system uses `text-embedding-ada-002` by default. You can modify the model in:
- `create_database.py` - Line where `CustomOpenAIEmbeddings` is initialized
- `query_data.py` - Same location
- `streamlit_app.py` - Same location

### Document Processing Parameters
In `create_database.py`, you can adjust:
- `chunk_size`: Size of text chunks (default: 300)
- `chunk_overlap`: Overlap between chunks (default: 100)

## 💡 Usage Examples

### Web Interface
1. Start the Streamlit app: `python -m streamlit run streamlit_app.py`
2. Enter your question in the text area
3. Click "🔍 Search" to get AI-powered answers
4. Review sources and relevance scores

### Command Line
```bash
# Ask about characters
python query_data.py "Who does Alice meet in Wonderland?"

# Ask about events
python query_data.py "What is the Mad Hatter's tea party like?"

# Ask about plot details
python query_data.py "How does Alice change size in the story?"
```

## 🎯 Features Explained

### Document Processing
- Automatically splits large documents into manageable chunks
- Creates vector embeddings for semantic search
- Preserves source metadata for citations

### Intelligent Search
- Uses cosine similarity for finding relevant content
- Configurable relevance thresholds
- Returns multiple sources with confidence scores

### AI Response Generation
- Uses retrieved context to generate accurate answers
- Maintains focus on source material
- Provides clear citations for verification

## 🔍 Troubleshooting

### Common Issues

1. **Import Error**: `ModuleNotFoundError: No module named 'langchain_community'`
   - Solution: Make sure you're running from the virtual environment
   - Run: `source venv/bin/activate` before running any scripts

2. **OpenAI API Error**
   - Check your API key in the `.env` file
   - Verify your OpenAI account has credits
   - Ensure the API key has the correct permissions

3. **Vector Database Not Found**
   - Run `python create_database.py` first to create the database
   - Make sure the `chroma/` directory exists

4. **No Documents Found**
   - Verify your documents are in `data/books/` directory
   - Ensure documents are in supported format (`.md`)

### Debug Mode
Run with debug information:
```bash
python -c "
import os
print('API Key set:', bool(os.environ.get('OPENAI_API_KEY')))
print('Chroma DB exists:', os.path.exists('chroma'))
print('Data directory:', os.listdir('data/books/') if os.path.exists('data/books/') else 'Not found')
"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📦 Dependencies

Key packages used in this project:
- `langchain==0.2.2` - RAG framework
- `langchain-community==0.2.3` - Community integrations
- `langchain-openai==0.1.8` - OpenAI integration
- `openai==1.31.1` - OpenAI API client
- `chromadb==0.5.0` - Vector database
- `streamlit==1.39.0` - Web interface
- `unstructured==0.14.4` - Document processing

## 🔐 Security Notes

- Never commit your `.env` file to version control
- Keep your OpenAI API key secure and private
- Consider using environment-specific API keys for development/production

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- Built following LangChain RAG tutorials
- Uses OpenAI's powerful embedding and language models
- Inspired by modern RAG architectures and best practices

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the [LangChain documentation](https://python.langchain.com/)
3. Check [OpenAI API documentation](https://platform.openai.com/docs/)
4. Open an issue in this repository

---

**Happy coding! 🚀**
