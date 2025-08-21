import os
import sys
import nltk
import time
from pinecone import Pinecone
from typing import List, Optional

from src.config.settings import PINECONE_INDEX
from src.config.settings import PINECONE_PERSONA_INDEX
from src.config.settings import PINECONE_API_KEY
from src.config.settings import PINECONE_NAMESPACE
from src.config.settings import PINECONE_TOPK_SEARCH
from src.config.settings import PINECONE_EMBEDDING_MODEL


nltk.download('punkt')

pc = Pinecone(api_key=PINECONE_API_KEY)

dense_index: Optional[object] = None
persona_dense_index: Optional[object] = None

def get_or_create_index(index_name: str = PINECONE_INDEX) -> object:
    """Get the index, creating it if it doesn't exist"""
    global dense_index
    global persona_dense_index
    
    if index_name == PINECONE_PERSONA_INDEX:
        if persona_dense_index is None:
            # Check if index exists first
            if not pc.has_index(PINECONE_PERSONA_INDEX):
                print(f"Index {PINECONE_PERSONA_INDEX} does not exist. Creating it...")
                pc.create_index_for_model(
                    name=PINECONE_PERSONA_INDEX,
                    cloud="aws",
                    region="us-east-1",
                    embed={
                        "model": PINECONE_EMBEDDING_MODEL,
                        "field_map": {"text": "canonical_name"}
                    }
                )
                print(f"Index {PINECONE_PERSONA_INDEX} created successfully.")
            
            persona_dense_index = pc.Index(PINECONE_PERSONA_INDEX)
        return persona_dense_index

    if dense_index is None:
        # Check if index exists first
        if not pc.has_index(PINECONE_INDEX):
            print(f"Index {PINECONE_INDEX} does not exist. Creating it...")
            pc.create_index_for_model(
                name=PINECONE_INDEX,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": PINECONE_EMBEDDING_MODEL,
                    "field_map": {"text": "chunk_text"}
                }
            )
            print(f"Index {PINECONE_INDEX} created successfully.")
        
        dense_index = pc.Index(PINECONE_INDEX)
    
    return dense_index

def read_and_chunk_sentences(
    file_path: str,
    chunk_size: int = 40,
    overlap: int = 10
) -> List[str]:
    """
    Reads a text file, splits it into sentences, and chunks them with overlap.

    Args:
        file_path (str): Path to the text file.
        chunk_size (int): Number of sentences per chunk.
        overlap (int): Number of overlapping sentences between chunks.

    Returns:
        List[str]: List of sentence chunks as strings.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = nltk.sent_tokenize(text, language='spanish')
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i:i+chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def load_persona_into_vectordb(
    name: str,
    lastname: str,
    person_id: str
) -> None:
    """
    Loads a persona into the vector database.

    Args:
        name (str): Name of the person.
        lastname (str): Last name of the person.
        person_id (str): Unique identifier for the person.
    """
    index = get_or_create_index(index_name=PINECONE_PERSONA_INDEX)
    
    # Create a record for the persona
    persona_record = {
        "_id": f"{person_id}",
        "canonical_name": f"{name} {lastname}", # embedding sobre el nombre y apellido
        "name": name,
        "lastname": lastname,
        "category": "persona"
    }
    
    # Upsert the record into the index
    index.upsert_records(
        namespace=PINECONE_NAMESPACE,
        records=[persona_record]
    )
    time.sleep(10)  # Wait for the upserted vectors to be indexed

def load_data_into_vectordb(
    dataset: List[str], 
    name: str,
    lastname: str,
    profile_type: str,
    person_id: str,
    category: str = "cv",
    ) -> None:
    """
    Loads data into the vector database.
    This function iters over a list of files that represent the dataset
    and chunks them into smaller pieces before inserting them into the vector database.

    Args:
        dataset (List[str]): List of file paths to be processed.
    """
    for doc in dataset:
        chunks = read_and_chunk_sentences(doc, chunk_size=5, overlap=2)
        category = category
        cv_chunks = []
        
        for i, chunk in enumerate(chunks, start=1):
            cv_chunks.append({
                "_id": f"cv_chunk_{person_id}_{i}",
                "chunk_text": chunk,
                "category": category,
                "name": name,
                "lastname": lastname,
                "profile_type": profile_type,
                "person_id": person_id,
            })
        
        index = get_or_create_index()

        # Upsert the records into a namespace
        index.upsert_records(
            namespace=PINECONE_NAMESPACE,
            records=cv_chunks
        )
        # Wait for the upserted vectors to be indexed
        time.sleep(10)

def search_similar(
    text: str, 
    top_k: str = PINECONE_TOPK_SEARCH, 
    namespace: str = PINECONE_NAMESPACE, 
    debug: bool = True
    ) -> List[str]:
    """
    Searches for similar items in the vector database based on the input text.

    Args:
        text (str): _Description of the text to search for.
        top_k (str, optional): How many similar entries should this function return. Defaults to PINECONE_TOPK_SEARCH.
        namespace (str, optional): Name of the namespace. Defaults to PINECONE_NAMESPACE.
        debug (bool, optional): debug flag for troubleshooting. Defaults to True.

    Returns:
        List[str]: List of similar items found in the vector database.
    """
    
    # View stats for the index
    index = get_or_create_index()
    stats = index.describe_index_stats()
    print(stats)

    # Search the dense index
    results = dense_index.search(
        namespace=namespace,
        query={
            "top_k": top_k,
            "inputs": {
                'text': text
            }
        }
    )

    # Print the results
    data = []
    for hit in results['result']['hits']:
        tmp = f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['chunk_text']:<50}"
        if debug:
            print(tmp)
        data.append(tmp)
        
    return data

if __name__ == "__main__":
    try:
        while True:
            msg = input("Press Enter to search for similar items: ")
            search_similar(msg)
    except KeyboardInterrupt:
        print("\nNos vemos la proxima.")
        sys.exit(0)