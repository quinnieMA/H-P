# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 13:51:32 2025

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 09:24:12 2025

@author: Lenovo
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# ========================
# Path Configuration
# ========================
class DataPaths:
    # Target overviews
    TARGET_FOLDER = "D:/OneDrive/MA/acquisition/NLP_tar_description/processed-0/trigram"
    TARGET_IDS = "D:/OneDrive/MA/acquisition/NLP_tar_description/processed-0/document_ids.txt"
    
    # Output
    OUTPUT_FOLDER = "D:/OneDrive/MA/acquisition/HP_results"
    TARGET_SIMILARITY = os.path.join(OUTPUT_FOLDER, "target_similarity_scores_hp.csv")
    TARGET_TOP10_NEIGHBORS = os.path.join(OUTPUT_FOLDER, "target_top10_neighbors_hp.csv")
    TARGET_SIMILARITY_MATRIX = os.path.join(OUTPUT_FOLDER, "target_similarity_matrix_hp.npy")

# ========================
# Text Processing Utilities
# ========================

def load_document_ids(file_path):
    """Load document IDs from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def extract_year_from_filename(filename):
    """Extract year from filename pattern: YYYY_tardesXXXXX.txt"""
    match = re.match(r'^(\d{4})_tardes', filename)
    if match:
        return match.group(1)
    return None

def read_text_files_by_year(folder_path):
    """Read all text files from folder and group by year {year: {filename: text}}"""
    texts_by_year = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            year = extract_year_from_filename(filename)
            if year is None:
                print(f"Warning: Could not extract year from filename: {filename}")
                continue
                
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content and len(content) > 10:  # 确保内容不是空或太短
                        # Remove file extension for ID
                        doc_id = filename.replace('.txt', '')
                        if year not in texts_by_year:
                            texts_by_year[year] = {}
                        texts_by_year[year][doc_id] = content
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return texts_by_year

# ========================
# Hoberg-Phillips Similarity Calculation
# ========================

def calculate_hp_similarity(texts_dict, year):
    """
    Calculate Hoberg-Phillips similarity for documents within the same year
    Returns similarity matrix and document IDs for that year
    """
    if not texts_dict:
        return None, [], []
    
    # Get IDs and texts in order
    doc_ids = list(texts_dict.keys())
    documents = [texts_dict[doc_id] for doc_id in doc_ids]
    
    print(f"Year {year}: {len(documents)} documents for HP similarity")
    
    if len(documents) < 2:
        print(f"Year {year}: Not enough documents for similarity calculation (need at least 2)")
        return None, doc_ids, []
    
    try:
        # Step 1: Create binary bag-of-words representation
        # HP方法使用二进制向量，表示词汇是否存在
        vectorizer = CountVectorizer(
            max_features=10000,  # HP方法通常使用更多特征
            min_df=1,           # HP方法不过滤罕见词
            max_df=1.0,         # 不过滤常见词
            binary=True,        # 关键：使用二进制表示
            ngram_range=(1, 2), # HP方法使用1-2grams
            stop_words='english'
        )
        
        # Create document-term matrix with binary representation
        dtm = vectorizer.fit_transform(documents)
        
        print(f"Year {year}: Vocabulary size: {dtm.shape[1]}")
        
        # Step 2: Calculate pairwise cosine similarity (HP方法的核心)
        print(f"Year {year}: Calculating HP cosine similarity matrix...")
        similarity_matrix = cosine_similarity(dtm)
        
        # HP方法的特殊处理：设置对角线为0（不自相似）
        np.fill_diagonal(similarity_matrix, 0)
        
        # Step 3: 可选 - 应用阈值（HP方法常用）
        # 只保留显著的正相似度，将低相似度设为0
        similarity_threshold = 0.01  # 可根据数据调整
        similarity_matrix[similarity_matrix < similarity_threshold] = 0
        
        print(f"Year {year}: Similarity matrix calculated, shape: {similarity_matrix.shape}")
        print(f"Year {year}: Non-zero similarity pairs: {np.sum(similarity_matrix > 0)}")
        
        return similarity_matrix, doc_ids, vectorizer.get_feature_names_out()
    
    except Exception as e:
        print(f"Year {year}: Error in HP similarity processing: {e}")
        return None, doc_ids, []

def create_industry_network(similarity_matrix, doc_ids, threshold=0.1):
    """
    创建产业关联网络（HP方法的核心输出）
    """
    network_edges = []
    
    for i in range(len(doc_ids)):
        for j in range(i+1, len(doc_ids)):
            if similarity_matrix[i, j] >= threshold:
                network_edges.append({
                    'source': doc_ids[i],
                    'target': doc_ids[j],
                    'similarity': similarity_matrix[i, j]
                })
    
    return pd.DataFrame(network_edges)

def find_top_k_neighbors_by_year(similarity_matrix, doc_ids, year, k=10):
    """
    Find top K most similar documents within the same year using HP method
    """
    if similarity_matrix is None:
        return pd.DataFrame()
    
    neighbors_data = []
    
    for i, doc_id in enumerate(doc_ids):
        # Get similarity scores for this document
        similarities = similarity_matrix[i]
        
        # HP方法：只考虑显著的正相似度
        valid_indices = np.where(similarities > 0)[0]
        if len(valid_indices) == 0:
            continue
            
        # 获取前k个邻居
        top_k_indices = valid_indices[np.argsort(similarities[valid_indices])[-k:]][::-1]
        
        # Store results only for top neighbors
        for rank, idx in enumerate(top_k_indices, 1):
            neighbors_data.append({
                'year': year,
                'source_target': doc_id,
                'neighbor_target': doc_ids[idx],
                'similarity_score': similarities[idx],
                'rank': rank
            })
    
    return pd.DataFrame(neighbors_data)

def save_year_similarity_summary(similarity_matrix, doc_ids, year, all_summary_data):
    """
    Save summary statistics for each year using HP method
    """
    if similarity_matrix is None:
        return all_summary_data
    
    for i, doc_id in enumerate(doc_ids):
        # Get similarity scores for this document (excluding self)
        similarities = similarity_matrix[i]
        
        # HP方法：只考虑正相似度
        valid_similarities = similarities[similarities > 0]
        if len(valid_similarities) == 0:
            continue
        
        # Calculate summary statistics
        all_summary_data.append({
            'year': year,
            'target_id': doc_id,
            'mean_similarity': np.mean(valid_similarities),
            'max_similarity': np.max(valid_similarities),
            'min_similarity': np.min(valid_similarities),
            'std_similarity': np.std(valid_similarities),
            'num_similar_targets': len(valid_similarities),
            'network_degree': len(valid_similarities)  # HP网络中的度中心性
        })
    
    return all_summary_data

# ========================
# Main Processing Pipeline (Year-wise HP Method)
# ========================

def year_wise_hp_similarity_analysis_pipeline():
    """Main pipeline for year-wise Hoberg-Phillips similarity analysis"""
    print("Starting year-wise Hoberg-Phillips similarity analysis pipeline...")
    
    # Create output directory
    os.makedirs(DataPaths.OUTPUT_FOLDER, exist_ok=True)
    
    try:
        # 1. Load target texts grouped by year
        print("Loading target overviews grouped by year...")
        texts_by_year = read_text_files_by_year(DataPaths.TARGET_FOLDER)
        
        # Get sorted list of years
        years = sorted(texts_by_year.keys())
        print(f"Found documents from years: {years}")
        
        all_neighbors_data = []
        all_summary_data = []
        year_matrices = {}
        industry_networks = {}
        
        # 2. Process each year separately
        for year in years:
            print(f"\nProcessing year {year}...")
            year_texts = texts_by_year[year]
            
            # Calculate HP similarity for this year
            similarity_matrix, doc_ids, feature_names = calculate_hp_similarity(year_texts, year)
            
            if similarity_matrix is not None:
                # Find top 10 neighbors for this year
                year_neighbors = find_top_k_neighbors_by_year(similarity_matrix, doc_ids, year, k=10)
                all_neighbors_data.append(year_neighbors)
                
                # Save summary for this year
                all_summary_data = save_year_similarity_summary(similarity_matrix, doc_ids, year, all_summary_data)
                
                # Create industry network
                industry_network = create_industry_network(similarity_matrix, doc_ids)
                industry_networks[year] = industry_network
                
                # Store matrix for this year
                year_matrices[year] = similarity_matrix
                
                print(f"Year {year}: Found {len(year_neighbors)} neighbor pairs")
                print(f"Year {year}: Industry network edges: {len(industry_network)}")
        
        # 3. Combine all results
        print("\nCombining results from all years...")
        
        # Combine neighbors data
        if all_neighbors_data:
            neighbors_df = pd.concat(all_neighbors_data, ignore_index=True)
            neighbors_df.to_csv(DataPaths.TARGET_TOP10_NEIGHBORS, index=False)
            print(f"Top neighbors saved to {DataPaths.TARGET_TOP10_NEIGHBORS}")
            print(f"Total neighbor pairs: {len(neighbors_df)}")
        else:
            neighbors_df = pd.DataFrame()
            print("No neighbor pairs found")
        
        # Combine summary data
        if all_summary_data:
            summary_df = pd.DataFrame(all_summary_data)
            summary_df.to_csv(DataPaths.TARGET_SIMILARITY, index=False)
            print(f"Similarity summary saved to {DataPaths.TARGET_SIMILARITY}")
        else:
            summary_df = pd.DataFrame()
            print("No summary data found")
        
        # Save industry networks
        for year, network in industry_networks.items():
            network_file = os.path.join(DataPaths.OUTPUT_FOLDER, f"industry_network_{year}.csv")
            network.to_csv(network_file, index=False)
            print(f"Industry network for {year} saved to {network_file}")
        
        # 4. HP方法特有的统计信息
        print("\n=== Hoberg-Phillips Method Summary Statistics ===")
        if len(summary_df) > 0:
            print(f"Total documents analyzed: {len(summary_df)}")
            print(f"Total years processed: {len(years)}")
            print(f"Mean similarity across all pairs: {summary_df['mean_similarity'].mean():.4f}")
            print(f"Average network degree: {summary_df['network_degree'].mean():.2f}")
            
            # Show statistics by year
            print("\nHP Similarity statistics by year:")
            year_stats = summary_df.groupby('year').agg({
                'mean_similarity': ['mean', 'std'],
                'network_degree': 'mean',
                'target_id': 'count'
            })
            print(year_stats)
            
            # Show top 10 most similar pairs across all years
            if len(neighbors_df) > 0:
                top_pairs = neighbors_df.nlargest(10, 'similarity_score')
                print("\nTop 10 most similar target pairs (HP method):")
                for _, row in top_pairs.iterrows():
                    print(f"Year {row['year']}: {row['source_target']} - {row['neighbor_target']}: {row['similarity_score']:.4f}")
        
        return summary_df, neighbors_df, year_matrices, industry_networks
        
    except Exception as e:
        print(f"Error in HP similarity analysis pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise

# ========================
# HP Method Specific Analysis
# ========================

def analyze_hp_network_properties(summary_df, industry_networks):
    """Analyze network properties of HP similarity results"""
    if len(summary_df) > 0:
        print("\n=== HP Network Properties Analysis ===")
        
        import matplotlib.pyplot as plt
        
        # Network degree distribution
        plt.figure(figsize=(10, 6))
        plt.hist(summary_df['network_degree'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Network Degree (HP Method)')
        plt.xlabel('Number of Similar Companies')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Similarity score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(summary_df['mean_similarity'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Mean HP Similarity Scores')
        plt.xlabel('Mean HP Similarity')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()

# ========================
# Main Execution
# ========================

if __name__ == "__main__":
    # Run the year-wise HP similarity pipeline
    summary_results, neighbor_results, year_matrices, industry_networks = year_wise_hp_similarity_analysis_pipeline()
    
    # Additional HP-specific analysis
    if len(summary_results) > 0:
        analyze_hp_network_properties(summary_results, industry_networks)
    
    print("\nHoberg-Phillips similarity analysis completed!")