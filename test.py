import os
import logging

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# List of URLs to load documents from
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Remove angle brackets from the URLs (requests raises InvalidSchema if URLs include < >)
urls = [u.strip("<>") for u in urls]

# Load documents from the URLs with basic error handling
docs = []
for url in urls:
    try:
        logger.info("Loading URL: %s", url)
        before = len(docs)
        loaded = WebBaseLoader(url).load()
        docs.extend(loaded)
        logger.info(
            "Loaded %d documents from %s (total %d)", len(loaded), url, len(docs)
        )
    except Exception as e:
        logger.exception("Failed to load %r", url)

# Flattened list of loaded documents
docs_list = docs

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)
logger.info("Split documents into %d chunks", len(doc_splits))



# Create embeddings for documents using the local Ollama qwen3 embedding model
try:
    logger.info("Initializing OllamaEmbeddings model 'qwen3-embedding'")
    ollama_emb = OllamaEmbeddings(model="qwen3-embedding", validate_model_on_init=True)
    logger.info("OllamaEmbeddings initialized")
except Exception as e:
    logger.exception("Failed to initialize OllamaEmbeddings")
    raise

persist_path = "data/embeddings.json"
logger.info("Using persist path: %s", persist_path)
os.makedirs(os.path.dirname(persist_path), exist_ok=True)

if os.path.exists(persist_path):
    logger.info("Loading persisted vector store from %s", persist_path)
    vectorstore = SKLearnVectorStore(embedding=ollama_emb, persist_path=persist_path)
else:
    logger.info("Building vector store and persisting to disk")
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits, embedding=ollama_emb, persist_path=persist_path
    )
    vectorstore.persist()
    logger.info("Vector store built and persisted")

retriever = vectorstore.as_retriever(k=4)
logger.info("Retriever initialized (k=4)")

# Define the prompt template for the LLM
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Make sure to reply using exactly the same language as the question.
    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)

# Initialize the LLM with the Aya Expanse model
# which was trained across a big number of languages
llm = ChatOllama(
    model="aya-expanse",
    temperature=0,
)

# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()

# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    def run(self, question):
        logger.info("Retrieving documents for question: %s", question)
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        logger.info("Retrieved %d documents", len(documents))
        # Extract content from retrieved documents
        doc_texts = "\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        model_answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        logger.info("RAG answer generated")
        return model_answer

# Initialize the RAG application
rag_application = RAGApplication(retriever, rag_chain)

# Example usage
QUESTION = "Cosa è il prompt engineering?"
answer = rag_application.run(QUESTION)
logger.info("Question: %s", QUESTION)
logger.info("Answer: %s", answer)
