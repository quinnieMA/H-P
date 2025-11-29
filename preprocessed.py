# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 10:45:02 2025

@author: Lenovo
"""


from stanfordnlp.server import CoreNLPClient
import datetime
import itertools
import os
from pathlib import Path
import re
import functools
import gensim
from gensim.models import Phrases
import tqdm
import subprocess
from global_options import DataPaths, BASE_DIR, AnalysisOptions

# ========================
# Phrase model training and processing functions
# ========================

def train_bigram_model(input_path: Path, model_path: Path) -> Phrases:
    """Train and save bigram model to specified path"""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    sentences = gensim.models.word2vec.PathLineSentences(str(input_path))
    bigram_model = Phrases(
        sentences,
        min_count=AnalysisOptions.PHRASE_MIN_COUNT,
        threshold=AnalysisOptions.PHRASE_THRESHOLD
    )
    bigram_model.save(str(model_path))
    return bigram_model

def train_trigram_model(input_path: Path, bigram_model_path: Path, trigram_model_path: Path) -> Phrases:
    """Train and save trigram model using existing bigram model"""
    trigram_model_path.parent.mkdir(parents=True, exist_ok=True)
    bigram_model = Phrases.load(str(bigram_model_path))
    
    # Read sentences and apply bigram first
    sentences = []
    for filename in os.listdir(input_path):
        file_path = input_path / filename
        with open(file_path, "r", encoding='utf-8') as file:
            for line in file:
                words = line.strip().split()
                sentences.append(bigram_model[words])
    
    # Train trigram model on bigram-processed sentences
    trigram_model = Phrases(
        sentences,
        min_count=AnalysisOptions.PHRASE_MIN_COUNT,
        threshold=AnalysisOptions.PHRASE_THRESHOLD
    )
    trigram_model.save(str(trigram_model_path))
    return trigram_model

def apply_phrases(input_path: Path, output_path: Path, phrase_model: Phrases) -> None:
    """Apply phrase model (bigram or trigram) and save results"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(input_path, "r", encoding='utf-8') as file:
            lines = file.readlines()
        transformed_lines = [" ".join(phrase_model[line.split()]) for line in lines]
        with open(output_path, "w", encoding='utf-8') as file:
            file.write("\n".join(transformed_lines))
        print(f"Processed: {input_path.name}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        
            

def apply_phrases_to_all_files(input_folder: Path, output_folder: Path, 
                             model_path: Path, document_path: Path, 
                             phrase_type: str = "bigram") -> None:
    """Process all files and save phrased text to single document"""
    output_folder.mkdir(parents=True, exist_ok=True)
    phrase_model = Phrases.load(str(model_path))
    all_transformed_lines = []
    
    for filename in os.listdir(input_folder):
        input_path = input_folder / filename
        output_path = output_folder / filename
        
        with open(input_path, "r", encoding='utf-8') as file:
            lines = file.readlines()
        transformed_lines = [" ".join(phrase_model[line.split()]) for line in lines]
        all_transformed_lines.extend(transformed_lines)
        
        # Write individual file
        with open(output_path, "w", encoding='utf-8') as out_file:
            out_file.write("\n".join(transformed_lines))
        
        print(f"Completed {phrase_type} processing: {input_path.name}")
    
    # Write combined document
    with open(document_path, "w", encoding='utf-8') as doc_file:
        doc_file.write("\n".join(all_transformed_lines))

def save_document_ids(folder_path: Path, output_file: Path) -> None:
    """Save all document filenames to text file"""
    filelist = os.listdir(folder_path)
    with open(output_file, 'w', encoding='utf-8') as file:
        for filename in filelist:
            file.write(filename + '\n')
    print(f"All filenames saved to {output_file}")

# ========================
# Main processing pipeline
# ========================

def process_phrases() -> None:
    """Execute phrase detection pipeline"""
    print("="*50)
    print("Starting phrase detection pipeline")
    print("="*50)
    
    # Train and apply bigram model
    print("\nTraining bigram model...")
    bigram_model = train_bigram_model(DataPaths.PARSED_DIR, DataPaths.BIGRAM_MODEL)
    
    print("\nApplying bigram model...")
    for filename in os.listdir(DataPaths.PARSED_DIR):
        file_path = DataPaths.PARSED_DIR / filename
        output_path = DataPaths.BIGRAM_DIR / filename
        apply_phrases(file_path, output_path, bigram_model)
    
    # Create combined bigram document
    apply_phrases_to_all_files(
        DataPaths.PARSED_DIR, 
        DataPaths.BIGRAM_DIR, 
        DataPaths.BIGRAM_MODEL, 
        DataPaths.PROCESSED_DIR / "document_bigram.txt",
        "bigram"
    )
    
    # Train and apply trigram model
    print("\nTraining trigram model...")
    trigram_model = train_trigram_model(
        DataPaths.PARSED_DIR, 
        DataPaths.BIGRAM_MODEL, 
        DataPaths.TRIGRAM_MODEL
    )
    
    print("\nApplying trigram model...")
    for filename in os.listdir(DataPaths.BIGRAM_DIR):  # Apply trigram to bigram output
        file_path = DataPaths.BIGRAM_DIR / filename
        output_path = DataPaths.TRIGRAM_DIR / filename
        apply_phrases(file_path, output_path, trigram_model)
    
    # Create combined trigram document
    apply_phrases_to_all_files(
        DataPaths.BIGRAM_DIR, 
        DataPaths.TRIGRAM_DIR, 
        DataPaths.TRIGRAM_MODEL, 
        DataPaths.PROCESSED_DIR / "document_trigram.txt",
        "trigram"
    )
    
    # Save document IDs
    save_document_ids(DataPaths.TRIGRAM_DIR, DataPaths.DOC_IDS)#previously it was INPUT_DIR

if __name__ == "__main__":
    process_phrases()
    print("\nPhrase detection pipeline completed!")