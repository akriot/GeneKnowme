import gradio as gr
import langchain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PubMedLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import utils as chromautils

def search_pubmed_and_query_chroma(pubmed_query, chroma_query):
    # Load documents from PubMed
    loader = PubMedLoader(pubmed_query)
    docs = loader.load()

    # Check the number of documents loaded
    document_count = len(docs)

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)

    # Filter and embed documents
    docs = chromautils.filter_complex_metadata(docs)
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embedding_function, persist_directory="./db")
    
    # Query the Chroma database
    docs = db.similarity_search(chroma_query)

    # Return the content and metadata of the most relevant document, if available
    if docs:
        return f"Document Count: {document_count}\n\nContent: {docs[0].page_content}\n\nMetadata: {docs[0].metadata}"
    else:
        return "No relevant documents found."

# Create Gradio interface
interface = gr.Interface(
    fn=search_pubmed_and_query_chroma,
    inputs=[gr.Textbox(label="PubMed Query"), gr.Textbox(label="Chroma Query")],
    outputs=gr.Textbox(label="Result"),
    title="PubMed and Chroma Search Tool",
    description="Enter a PubMed query to fetch documents and a query to search within those documents using Chroma."
)

# Launch the app
interface.launch()

