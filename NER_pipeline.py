# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 16:05:04 2025

@author: FM
"""

# Import required libraries
from stanfordcorenlp import StanfordCoreNLP
import subprocess
import os
import pandas as pd


java_bin_path = r"C:\Program Files\Java\jdk1.8.0_202\bin"
os.environ['PATH'] = java_bin_path + os.pathsep + os.environ['PATH']

def check_java():
    try:
        java_version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT)
        print("Java is installed, version info:")
        print(java_version.decode())
    except subprocess.CalledProcessError as e:
        print("Failed to execute java -version")
        print(e)
    except FileNotFoundError:
        print("Java is not installed or not added to PATH.")

def check_stanford_corenlp(corenlp_path):
    try:
        # Use a simpler command to check if CoreNLP can be loaded
        java_cmd = f'java -mx3g -cp "{corenlp_path}/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -version'
        process = subprocess.Popen(
            java_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            encoding='utf-8',  # Explicitly specify encoding
            errors='replace'   # Replace invalid characters with placeholders
        )
        stdout, stderr = process.communicate(timeout=30)
        
        if "Stanford CoreNLP version" in stdout:
            print("StanfordCoreNLP is available:")
            print(stdout.splitlines()[0])  # Print version info
        else:
            print("CoreNLP startup exception, please check:")
            print(stderr if stderr else stdout)
            
    except Exception as e:
        print(f"CoreNLP detection failed: {str(e)}")

# Replace the path below with the actual path to your StanfordCoreNLP installation
stanford_corenlp_path = r'D:/OneDrive/NLP/stanford-corenlp-4.5.7'

check_java()
check_stanford_corenlp(stanford_corenlp_path)

# Initialize StanfordCoreNLP
nlp = StanfordCoreNLP(r'D:/OneDrive/NLP/stanford-corenlp-4.5.7', lang='en')

def process_text(text):
    # Use NER to annotate text
    annotated = nlp.ner(text)
    
    # Build processed text
    processed_tokens = []
    for token, tag in annotated:
        if tag != 'O':
            processed_tokens.append(tag)
        else:
            processed_tokens.append(token)
    
    # Recombine tokens into text
    processed_text = ' '.join(processed_tokens)
    
    # Fix spaces around punctuation (simple fix)
    processed_text = processed_text.replace(' ,', ',').replace(' .', '.').replace(' )', ')').replace('( ', '(')
    
    return processed_text

# Set paths
input_dir = r'D:/OneDrive/MA/acquisition/NLP_tar_description/tardescription'
output_dir = r'D:/OneDrive/MA/acquisition/NLP_tar_description/processed/parsed'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process all txt files
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Read file content
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Process text
        processed_content = process_text(content)
        
        # Write processed content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        print(f'Processed: {filename}')

# Close NLP connection
nlp.close()
