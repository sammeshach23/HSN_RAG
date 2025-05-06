import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

# Set page config
st.set_page_config(
    page_title="HSN Code Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Define relative paths that will create folders in the current directory
DATA_PATH = "C:/hsn_rag/hscodes.csv" # Relative path for the CSV file
DB_PATH = os.path.join(os.getcwd(), "hsn_data", "chroma_db")  # Relative path for the database

# Ensure directories exist
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

# Display paths for user information
st.sidebar.info(f"Using data path: {DATA_PATH}")
st.sidebar.info(f"Using database path: {DB_PATH}")

# Class for HSN Code RAG System
class HSNCodeRAG:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Define custom embedding function using sentence-transformers
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
        
        # Try to get collection or create a new one
        try:
            self.collection = self.client.get_collection(
                name="hsn_codes",
                embedding_function=self.embedding_function
            )
            st.sidebar.success(f"Database loaded with {self.collection.count()} HSN codes")
        except:
            self.collection = self.client.create_collection(
                name="hsn_codes",
                embedding_function=self.embedding_function
            )
            st.sidebar.info("Created new ChromaDB collection")
    
    def preprocess_description(self, description: str) -> str:
        """Clean and standardize HSN code descriptions"""
        # Remove special characters, convert to lowercase
        description = description.lower().strip()
        return description
    
    def ingest_data(self, df: pd.DataFrame) -> None:
        """Ingest data into ChromaDB with batch processing"""
        total_ingested = 0
        batch_size = 100  # Process 100 records at a time to stay well below the 166 limit
        
        # Process data in batches
        for i in range(0, len(df), batch_size):
            # Get the current batch
            batch_df = df.iloc[i:i+batch_size]
            
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in batch_df.iterrows():
                hscode = str(row['hscode'])
                description = self.preprocess_description(str(row['description']))
                
                documents.append(description)
                metadatas.append({"hscode": hscode, "original_description": row['description']})
                ids.append(f"doc_{idx}")
            
            # Add batch to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            total_ingested += len(documents)
            
            # Show progress
            progress_percentage = min(100, int((i + batch_size) / len(df) * 100))
            st.sidebar.progress(progress_percentage, text=f"Ingested {total_ingested}/{len(df)} records")
        
        return total_ingested
    
    def query(self, query_text: str, n_results: int = 5) -> Dict:
        """Query the vector database for similar HSN codes"""
        # Preprocess the query
        query_text = self.preprocess_description(query_text)
        
        # Query the collection
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        return results

# UI Functions
def display_results(results: Dict):
    """Display search results in a nice format"""
    if not results["ids"][0]:
        st.warning("No matching HSN codes found.")
        return
    
    st.subheader("Top Matching HSN Codes")
    
    # Create a DataFrame for better display
    result_data = []
    for i in range(len(results["ids"][0])):
        result_data.append({
            "HSN Code": results["metadatas"][0][i]["hscode"],
            "Description": results["metadatas"][0][i]["original_description"],
            "Similarity Score": round(float(results["distances"][0][i]), 4) if "distances" in results else "N/A"
        })
    
    # Create a DataFrame and display
    result_df = pd.DataFrame(result_data)
    st.dataframe(result_df, use_container_width=True)

def create_sample_data():
    """Create sample data file if it doesn't exist"""
    if os.path.exists(DATA_PATH):
        return
        
    # Sample data from the user's example
    sample_data = [
        {"hscode": "01012100", "description": "Live horses, asses, mules and hinnies:Horses:*Pure-bred breeding animals<951>"},
        {"hscode": "01012910", "description": "Live horses, asses, mules and hinnies:Horses:*Other:**For slaughter"},
        {"hscode": "01012990", "description": "Live horses, asses, mules and hinnies:Horses:*Other:**Other"},
        {"hscode": "01013000", "description": "Live horses, asses, mules and hinnies:Asses"},
        {"hscode": "01019000", "description": "Live horses, asses, mules and hinnies:Other"},
        {"hscode": "01022110", "description": "Live bovine animals:Cattle:*Pure-bred breeding animals<952>:**Heifers (female bovines that have never calved)"},
        {"hscode": "01022130", "description": "Live bovine animals:Cattle:*Pure-bred breeding animals<952>:**Cows"},
        {"hscode": "01022190", "description": "Live bovine animals:Cattle:*Pure-bred breeding animals<952>:**Other"},
        {"hscode": "01022910", "description": "Live bovine animals:Cattle:*Other:**Of the sub-genus Bibos or of the sub-genus Poephagus"},
        {"hscode": "01022921", "description": "Live bovine animals:Cattle:*Other:**Other:***Of a weight not exceeding 80kg<336>"},
        {"hscode": "01022922", "description": "Live bovine animals:Cattle:*Other:**Other:***Of a weight not exceeding 80kg<336>++++Young male bovine animals, intended for fattening"},
        {"hscode": "01022923", "description": "Live bovine animals:Cattle:*Other:**Other:***Of a weight not exceeding 80kg<336>++++Heifers of the grey, brown or yellow mountain breeds and spotted Pinzgau breed, other than for slaughter"},
        {"hscode": "01022924", "description": "Live bovine animals:Cattle:*Other:**Other:***Of a weight not exceeding 80kg<336>++++Heifers of the Schwyz and Fribourg breeds, other than for slaughter"},
        {"hscode": "01022925", "description": "Live bovine animals:Cattle:*Other:**Other:***Of a weight not exceeding 80kg<336>++++Heifers of the spotted Simmental breed, other than for slaughter"},
        {"hscode": "01022926", "description": "Live bovine animals:Cattle:*Other:**Other:***Of a weight not exceeding 80kg<336>++++Bulls of the Schwyz, Fribourg and spotted Simmental breeds, other than for slaughter"},
        {"hscode": "01022929", "description": "Live bovine animals:Cattle:*Other:**Other:***Of a weight not exceeding 80kg<336>++++Other"},
        {"hscode": "01022941", "description": "Live bovine animals:Cattle:*Other:**Other:***of a weight exceeding 80kg but not exceeding 160kg:****For slaughter<336>"},
        {"hscode": "01022949", "description": "Live bovine animals:Cattle:*Other:**Other:***of a weight exceeding 80kg but not exceeding 160kg:****Other"},
        {"hscode": "01022951", "description": "Live bovine animals:Cattle:*Other:**Other:***of a weight exceeding 80kg but not exceeding 160kg:****Other++++Young male bovine animals, intended for fattening"}
    ]
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(sample_data)
    df.to_csv(DATA_PATH, index=False)
    st.sidebar.success(f"Sample data created at {DATA_PATH}")

def load_and_process_data(rag_system):
    """Load HSN code data from the predefined path"""
    try:
        # Create sample data if needed
        create_sample_data()
        
        # Load data
        df = pd.read_csv(DATA_PATH)
        
        # Display info about the data
        st.sidebar.info(f"Loaded CSV with {len(df)} HSN codes")
        
        # Check if required columns exist
        required_cols = ["hscode", "description"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
            return False
        
        # Check if data needs to be ingested (collection is empty)
        if rag_system.collection.count() == 0:
            with st.spinner("Loading HSN codes into database (this may take a while for large datasets)..."):
                count = rag_system.ingest_data(df)
                st.sidebar.success(f"Successfully loaded {count} HSN codes")
        else:
            st.sidebar.success(f"Using existing database with {rag_system.collection.count()} HSN codes")
        
        return True
    except Exception as e:
        st.error(f"Error processing the HSN code data: {e}")
        st.error(f"Details: {str(e)}")
        st.error("If you're experiencing batch size errors, try reducing the batch_size in the HSNCodeRAG.ingest_data method")
        return False

def main():
    st.title("HSN Code Search Engine")
    
    # Initialize RAG system
    rag_system = HSNCodeRAG(DB_PATH)
    
    # Load data (automatically)
    data_loaded = load_and_process_data(rag_system)
    
    if not data_loaded:
        st.stop()
    
    # Search interface
    st.subheader("Search for HSN Codes")
    
    # Search input
    query = st.text_input("Enter product description:", placeholder="e.g., Live horses for breeding")
    
    if query:
        with st.spinner("Searching for similar HSN codes..."):
            results = rag_system.query(query)
            display_results(results)

if __name__ == "__main__":
    main()
    