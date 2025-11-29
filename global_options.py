# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 10:41:24 2025

@author: Lenovo
"""


import os
from pathlib import Path
from typing import Dict, List, Optional, Set

# ========================
# Hardware Resource Configuration
# ========================
N_CORES: int = os.cpu_count() - 1 or 1  # Use all CPU cores minus one, keep at least 1
RAM_CORENLP: str = "8G"  # Maximum memory allocation for CoreNLP parsing
PARSE_CHUNK_SIZE: int = 200  # Number of lines to process per batch (adjust based on memory)

# ========================
# Base Path Configuration
# ========================
BASE_DIR = Path("D:/OneDrive/MA/acquisition/NLP_tar_description")
os.environ["CORENLP_HOME"] = str("D:/OneDrive/NLP/stanford-corenlp-4.5.7/")

# ========================
# Directory Structure
# ========================
class DataPaths:
    # Raw data
    INPUT_DIR = BASE_DIR / "tardescription"  # Original 43 text files
    PROCESSED_DIR = BASE_DIR / "processed"
    
    # Processing stages
    PARSED_DIR = BASE_DIR / "processed/parsed"    # CoreNLP parsing results
    UNIGRAM_DIR = BASE_DIR / "processed/unigram"  # Basic tokenization results
    BIGRAM_DIR = BASE_DIR / "processed/bigram"    # Bigram processed results
    TRIGRAM_DIR = BASE_DIR / "processed/trigram"  # Trigram processed results
    DOC_IDS = BASE_DIR / "processed/document_ids.txt" 
    
    # Model storage
    MODEL_DIR = BASE_DIR / "models"
    BIGRAM_MODEL = MODEL_DIR / "phrases/bigram.mod"
    TRIGRAM_MODEL = MODEL_DIR / "phrases/trigram.mod"
    W2V_MODEL = MODEL_DIR / "w2v/w2v.mod"
    
    # Resource files
    RESOURCES_DIR = BASE_DIR
    DICTIONARIES_DIR = RESOURCES_DIR / "dictionaries"
    SENTIMENT_RESULTS = BASE_DIR / "sentiment_results"
    STOPWORDS_FILE = Path("D:/OneDrive/NLP/stanford-corenlp-4.5.7/patterns/stopwords.txt")  # Stopwords file path
    SEEDWORDS_FILE = BASE_DIR / "dictionaries/seedwords.txt"
    SCORE_FILE =  BASE_DIR /"outputs"
# ========================
# NLP Processing Parameters
# ========================
class AnalysisOptions:
    # Phrase model parameters
    PHRASE_THRESHOLD: float = 10  # Lower values generate more phrases
    PHRASE_MIN_COUNT: int = 10    # Minimum frequency threshold
    
    # Word2Vec parameters
    W2V_DIM: int = 300     # Vector dimensionality
    W2V_WINDOW: int = 10   # Context window size
    W2V_ITER: int = 20     # Training iterations

    # Text preprocessing parameters
    REMOVE_PUNCTUATION: bool = True  # Whether to remove punctuation
    REMOVE_STOPWORDS: bool = True    # Whether to remove stopwords
    
    # Dictionary configuration
    DICT_RESTRICT_VOCAB = None  # Can be set to 0.2 to restrict to top 20% frequent words
    
    
    # ========================
    # Hardware Resource Configuration
    # ========================
    N_CORES: int = os.cpu_count() - 1 or 1  # Use all CPU cores minus one, keep at least 1
    RAM_CORENLP: str = "8G"  # Maximum memory allocation for CoreNLP parsing
    PARSE_CHUNK_SIZE: int = 200  # Number of lines to process per batch

# ========================
# Directory Initialization
# ========================
def init_directories():
    """Create all required directory structures"""
    # Processing directories
    DataPaths.PARSED_DIR.mkdir(parents=True, exist_ok=True)
    DataPaths.UNIGRAM_DIR.mkdir(parents=True, exist_ok=True)
    DataPaths.BIGRAM_DIR.mkdir(parents=True, exist_ok=True)
    DataPaths.TRIGRAM_DIR.mkdir(parents=True, exist_ok=True)
    
    # Model directories
    DataPaths.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (DataPaths.MODEL_DIR / "phrases").mkdir(exist_ok=True)
    (DataPaths.MODEL_DIR / "w2v").mkdir(exist_ok=True)
    
    # Resource directories
    DataPaths.RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    DataPaths.DICTIONARIES_DIR.mkdir(exist_ok=True)
    DataPaths.SCORE_FILE.mkdir(parents=True, exist_ok=True)

# Initialize directories
init_directories()

# ========================
# Environment Verification
# ========================
if __name__ == "__main__":
    print("=== Path Verification ===")
    print(f"CoreNLP path: {os.environ['CORENLP_HOME']}")
    print(f"Input directory: {DataPaths.INPUT_DIR} (exists: {DataPaths.INPUT_DIR.exists()})")
    print(f"Stopwords file: {DataPaths.STOPWORDS_FILE} (exists: {DataPaths.STOPWORDS_FILE.exists()})")