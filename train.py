# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 12:16:42 2025

@author: Lenovo
"""

"""
Module to train Doc2Vec model on TNIC data - Adapted for HP industry analysis
"""

import argparse
import csv
import json
import os
import shutil
import time
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import gensim
import nltk
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt

# 从你的现有代码导入路径配置
try:
    from global_options import DataPaths, BASE_DIR
    print("✓ 成功导入全局配置")
except:
    # 如果导入失败，使用硬编码路径
    class DataPaths:
        TRIGRAM_DIR = Path("D:/OneDrive/MA/acquisition/NLP_tar_overview/processed/trigram")
        OUTPUT_DIR = Path("D:/OneDrive/MA/acquisition/doc2vec_overview")
    BASE_DIR = Path("D:/OneDrive/MA/acquisition")

# 设置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger("hp_doc2vec")

class HP_Doc2Vec_Trainer:
    """适配HP行业分析数据的Doc2Vec训练器"""
    
    def __init__(self, args):
        self.args = args
        self.data_dir = DataPaths.TRIGRAM_DIR
        self.output_base = Path("D:/OneDrive/MA/acquisition/doc2vec_overview")
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # 确保使用多核训练
        assert gensim.models.doc2vec.FAST_VERSION > -1, "需要安装优化版本的gensim"
        
        # 模型超参数
        self.model_kwargs = {
            "vector_size": args.vector_size,
            "alpha": args.alpha,
            "min_alpha": args.min_alpha,
            "min_count": args.min_count,
            "dm": 0 if args.algorithm == "pv_dbow" else 1,
            "window": args.window,
            "dbow_words": 1,
            "hs": args.hs,
            "sample": 1e-05,
            "negative": args.negative,
            "ns_exponent": 0.75,
            "workers": args.workers,
        }
        
        # 如果是PV-DMC或PV-DMA模式
        if args.algorithm == "pv_dmc":
            self.model_kwargs["dm_concat"] = 1
        elif args.algorithm == "pv_dma":
            self.model_kwargs["dm_mean"] = 1
    
    def extract_company_id(self, filename):
        """从文件名提取公司ID，如：2015_tarov12345.txt → 12345"""
        match = re.search(r'tarov(\d+)', filename)
        return match.group(1) if match else filename.replace('.txt', '')
    
    def extract_year(self, filename):
        """从文件名提取年份"""
        match = re.match(r'^(\d{4})_', filename)
        return match.group(1) if match else "unknown"
    
    def preprocess_text(self, text):
        """简单的文本预处理"""
        import re
        
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符，保留字母和数字
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_documents(self):
        """加载你的文本文件，按年份分组"""
        documents_by_year = {}
        
        LOGGER.info(f"从目录加载文档: {self.data_dir}")
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.data_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if content and len(content) > 10:
                        year = self.extract_year(filename)
                        company_id = self.extract_company_id(filename)
                        
                        # 预处理文本
                        preprocessed = self.preprocess_text(content)
                        
                        if year not in documents_by_year:
                            documents_by_year[year] = []
                        
                        # 存储原始文件名和内容
                        documents_by_year[year].append({
                            'filename': filename,
                            'company_id': company_id,
                            'year': year,
                            'content': preprocessed,
                            'raw_content': content
                        })
                        
                except Exception as e:
                    LOGGER.warning(f"读取文件 {filename} 时出错: {e}")
        
        LOGGER.info(f"加载完成: 共有 {len(documents_by_year)} 个年份的数据")
        for year, docs in documents_by_year.items():
            LOGGER.info(f"  年份 {year}: {len(docs)} 个文档")
        
        return documents_by_year
    
    def create_tagged_documents(self, documents):
        """创建TaggedDocument对象用于Doc2Vec训练"""
        tagged_docs = []
        
        for doc_info in documents:
            # 使用文件名作为tag
            tag = doc_info['filename'].replace('.txt', '')
            # 分词
            words = doc_info['content'].split()
            tagged_docs.append(TaggedDocument(words=words, tags=[tag]))
        
        return tagged_docs
    
    def train_model(self, tagged_docs, year, output_dir):
        """训练Doc2Vec模型"""
        LOGGER.info(f"年份 {year}: 训练Doc2Vec模型")
        
        # 初始化模型
        if self.args.algorithm == "pv_dmc":
            model = Doc2Vec(dm_concat=1, **self.model_kwargs)
        elif self.args.algorithm == "pv_dma":
            model = Doc2Vec(dm_mean=1, **self.model_kwargs)
        else:
            model = Doc2Vec(**self.model_kwargs)
        
        # 构建词汇表
        LOGGER.info(f"年份 {year}: 构建词汇表")
        model.build_vocab(tagged_docs)
        LOGGER.info(f"年份 {year}: 词汇表大小: {len(model.wv)}")
        
        # 保存词汇表
        vocab_path = output_dir / f"vocabulary_{year}.txt"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            term_counts = [
                [term, model.wv.get_vecattr(term, "count")]
                for term in model.wv.key_to_index
            ]
            term_counts.sort(key=lambda x: -x[1])
            for term, count in term_counts:
                f.write(f"{term}, {count}\n")
        
        # 训练模型
        LOGGER.info(f"年份 {year}: 开始训练，共 {self.args.epochs} 个epochs")
        model.train(
            tagged_docs,
            total_examples=len(tagged_docs),
            epochs=self.args.epochs
        )
        
        return model
    
    def calculate_similarities(self, model, documents, year, output_dir):
        """计算文档相似度矩阵"""
        LOGGER.info(f"年份 {year}: 计算相似度矩阵")
        
        # 准备输出文件
        similarity_file = output_dir / f"similarity_{year}.tsv"
        matrix_file = output_dir / f"similarity_matrix_{year}.npy"
        
        # 获取所有文档向量
        doc_vectors = {}
        doc_ids = []
        
        for doc_info in documents:
            filename = doc_info['filename']
            tag = filename.replace('.txt', '')
            doc_vectors[tag] = model.dv[tag]
            doc_ids.append(tag)
        
        # 计算相似度矩阵
        n_docs = len(doc_ids)
        similarity_matrix = np.zeros((n_docs, n_docs))
        
        with open(similarity_file, 'w', newline='', encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            writer.writerow(['target_id1', 'target_id2', 'similarity_score', 'year'])
            
            for i in range(n_docs):
                for j in range(i+1, n_docs):
                    # 计算余弦相似度
                    vec1 = doc_vectors[doc_ids[i]]
                    vec2 = doc_vectors[doc_ids[j]]
                    
                    similarity = np.dot(vec1, vec2) / (
                        np.linalg.norm(vec1) * np.linalg.norm(vec2)
                    )
                    
                    # 保存到矩阵
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
                    
                    # 保存到TSV
                    writer.writerow([
                        doc_ids[i], 
                        doc_ids[j], 
                        f"{similarity:.6f}",
                        year
                    ])
        
        # 保存相似度矩阵
        np.save(matrix_file, similarity_matrix)
        
        # 保存文档ID列表
        with open(output_dir / f"doc_ids_{year}.txt", 'w', encoding='utf-8') as f:
            for doc_id in doc_ids:
                f.write(f"{doc_id}\n")
        
        return similarity_matrix, doc_ids
    
    def find_top_neighbors(self, similarity_matrix, doc_ids, year, output_dir, k=10):
        """找到每个文档的top-k最近邻"""
        LOGGER.info(f"年份 {year}: 寻找top-{k}最近邻")
        
        neighbors_file = output_dir / f"top{k}_neighbors_{year}.csv"
        neighbors_data = []
        
        for i, doc_id in enumerate(doc_ids):
            # 获取相似度（排除自己）
            similarities = similarity_matrix[i]
            
            # 找到top-k（排除自己）
            top_indices = np.argsort(similarities)[-k-1:-1][::-1]
            
            for rank, idx in enumerate(top_indices, 1):
                neighbors_data.append({
                    'year': year,
                    'source_target': doc_id,
                    'neighbor_target': doc_ids[idx],
                    'similarity_score': similarities[idx],
                    'rank': rank
                })
        
        # 保存为DataFrame
        neighbors_df = pd.DataFrame(neighbors_data)
        neighbors_df.to_csv(neighbors_file, index=False)
        
        return neighbors_df
    
    def run(self):
        """主运行流程"""
        LOGGER.info("=" * 60)
        LOGGER.info("HP行业分析 - Doc2Vec模型训练")
        LOGGER.info("=" * 60)
        
        # 创建时间戳目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = self.output_base / f"{timestamp}_{self.args.algorithm}_dim{self.args.vector_size}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        LOGGER.info(f"输出目录: {output_dir}")
        
        # 保存配置
        config = {
            'args': vars(self.args),
            'model_kwargs': self.model_kwargs,
            'data_dir': str(self.data_dir),
            'timestamp': timestamp
        }
        
        with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 加载文档
        documents_by_year = self.load_documents()
        
        all_results = {
            'similarities': {},
            'neighbors': {},
            'models': {}
        }
        
        # 按年份处理
        for year in sorted(documents_by_year.keys()):
            LOGGER.info(f"\n处理年份: {year}")
            
            documents = documents_by_year[year]
            
            if len(documents) < 2:
                LOGGER.warning(f"年份 {year}: 文档数量不足（{len(documents)}），跳过")
                continue
            
            # 创建年份输出目录
            year_dir = output_dir / str(year)
            year_dir.mkdir(exist_ok=True)
            
            # 创建TaggedDocument
            tagged_docs = self.create_tagged_documents(documents)
            
            # 训练模型
            model = self.train_model(tagged_docs, year, year_dir)
            
            # 保存模型
            model_path = year_dir / f"doc2vec_model_{year}.model"
            model.save(str(model_path))
            all_results['models'][year] = str(model_path)
            
            # 计算相似度
            similarity_matrix, doc_ids = self.calculate_similarities(
                model, documents, year, year_dir
            )
            
            # 寻找最近邻
            neighbors_df = self.find_top_neighbors(
                similarity_matrix, doc_ids, year, year_dir, k=self.args.top_k
            )
            
            # 保存结果
            all_results['similarities'][year] = {
                'matrix': similarity_matrix.tolist(),
                'doc_ids': doc_ids
            }
            
            all_results['neighbors'][year] = neighbors_df.to_dict('records')
            
            # 年度统计
            LOGGER.info(f"年份 {year} 完成:")
            LOGGER.info(f"  文档数量: {len(documents)}")
            LOGGER.info(f"  平均相似度: {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].mean():.4f}")
            LOGGER.info(f"  最大相似度: {similarity_matrix.max():.4f}")
        
        # 合并所有年份的结果
        LOGGER.info("\n合并所有年份的结果...")
        
        # 合并最近邻
        all_neighbors = []
        for year, neighbors in all_results['neighbors'].items():
            for neighbor in neighbors:
                all_neighbors.append(neighbor)
        
        if all_neighbors:
            all_neighbors_df = pd.DataFrame(all_neighbors)
            all_neighbors_df.to_csv(output_dir / 'all_years_top_neighbors.csv', index=False)
            LOGGER.info(f"保存合并的最近邻结果: {len(all_neighbors_df)} 条记录")
        
        # 生成分析报告
        self.generate_report(all_results, output_dir)
        
        LOGGER.info(f"\n训练完成！结果保存在: {output_dir}")
        
        return all_results
    
    def generate_report(self, results, output_dir):
        """生成分析报告"""
        report_file = output_dir / 'analysis_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("HP行业分析 - Doc2Vec模型训练报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"模型配置:\n")
            f.write(f"  算法: {self.args.algorithm}\n")
            f.write(f"  向量维度: {self.args.vector_size}\n")
            f.write(f"  窗口大小: {self.args.window}\n")
            f.write(f"  训练轮次: {self.args.epochs}\n")
            f.write(f"  Workers: {self.args.workers}\n\n")
            
            years = list(results['similarities'].keys())
            f.write(f"处理的年份: {', '.join(years)}\n\n")
            
            for year in years:
                if year in results['similarities']:
                    matrix = np.array(results['similarities'][year]['matrix'])
                    n_docs = matrix.shape[0]
                    
                    # 提取上三角矩阵（不包括对角线）
                    triu_indices = np.triu_indices_from(matrix, k=1)
                    similarities = matrix[triu_indices]
                    
                    f.write(f"年份 {year}:\n")
                    f.write(f"  文档数量: {n_docs}\n")
                    f.write(f"  相似度对数量: {len(similarities)}\n")
                    f.write(f"  平均相似度: {similarities.mean():.6f}\n")
                    f.write(f"  中位数相似度: {np.median(similarities):.6f}\n")
                    f.write(f"  标准差: {similarities.std():.6f}\n")
                    f.write(f"  最小相似度: {similarities.min():.6f}\n")
                    f.write(f"  最大相似度: {similarities.max():.6f}\n\n")
            
            # 总体统计
            f.write("\n总体统计:\n")
            f.write(f"  总年份数: {len(years)}\n")
            f.write(f"  总文档数: {sum([len(results['similarities'][y]['doc_ids']) for y in years if y in results['similarities']])}\n")

def main():
    parser = argparse.ArgumentParser(description='HP行业分析 - Doc2Vec模型训练')
    
    # 模型参数
    parser.add_argument('--algorithm', type=str, default='pv_dbow',
                       choices=['pv_dbow', 'pv_dmc', 'pv_dma'],
                       help='训练算法: pv_dbow, pv_dmc, pv_dma')
    parser.add_argument('--vector_size', type=int, default=300,
                       help='向量维度')
    parser.add_argument('--window', type=int, default=15,
                       help='窗口大小')
    parser.add_argument('--epochs', type=int, default=40,
                       help='训练轮次')
    parser.add_argument('--min_count', type=int, default=3,
                       help='最低词频')
    parser.add_argument('--alpha', type=float, default=0.025,
                       help='初始学习率')
    parser.add_argument('--min_alpha', type=float, default=0.0001,
                       help='最小学习率')
    parser.add_argument('--hs', type=int, default=1,
                       help='是否使用hierarchical softmax')
    parser.add_argument('--negative', type=int, default=5,
                       help='负采样数量')
    parser.add_argument('--workers', type=int, default=4,
                       help='并行工作进程数')
    parser.add_argument('--top_k', type=int, default=10,
                       help='每个文档的最近邻数量')
    
    args = parser.parse_args()
    
    # 训练模型
    trainer = HP_Doc2Vec_Trainer(args)
    results = trainer.run()

if __name__ == "__main__":
    main()