import os
import sys
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from FlagEmbedding import FlagModel
import logging
import scipy.sparse

# Setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_DIR = "/workspace/llmxtra/data/airitiv2"
VOCAB_CN = os.path.join(DATA_DIR, "vocab_cn")
VOCAB_EN = os.path.join(DATA_DIR, "vocab_en")
CORPUS_CN = os.path.join(DATA_DIR, "train_texts_cn.txt")
CORPUS_EN = os.path.join(DATA_DIR, "train_texts_en.txt")

OUTPUT_W2V_CN = os.path.join(DATA_DIR, "word2vec_cn.npz")
OUTPUT_W2V_EN = os.path.join(DATA_DIR, "word2vec_en.npz")
OUTPUT_BGEM3_CN = os.path.join(DATA_DIR, "word_embeddings_cn.npy")
OUTPUT_BGEM3_EN = os.path.join(DATA_DIR, "word_embeddings_en.npy")

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def train_and_save_word2vec(corpus_path, vocab_list, output_path, lang):
    print(f"Training Word2Vec for {lang}...")
    # Train Word2Vec
    # vector_size=300, window=5, min_count=1, workers=4
    model = Word2Vec(sentences=LineSentence(corpus_path), vector_size=300, window=5, min_count=1, workers=16)
    
    # Convert to numpy array
    embeddings_matrix = np.zeros((len(vocab_list), 300))
    for i, word in enumerate(vocab_list):
        if word in model.wv:
            embeddings_matrix[i] = model.wv[word]
        else:
            # print(f"Warning: Word '{word}' not found in Word2Vec model. Using zeros.")
            pass
            
    print(f"Saving Word2Vec embeddings for {lang} to {output_path} as sparse matrix...")
    # Save as sparse matrix (csr_matrix) in .npz format
    sparse_matrix = scipy.sparse.csr_matrix(embeddings_matrix)
    scipy.sparse.save_npz(output_path, sparse_matrix)

def generate_and_save_bgem3(vocab_list, output_path, lang):
    print(f"Generating BGEM3 embeddings for {lang}...")
    model = FlagModel('BAAI/bge-m3', use_fp16=True)
    
    # Encode in batches
    batch_size = 128
    embeddings = model.encode(vocab_list, batch_size=batch_size)
    
    print(f"Saving BGEM3 embeddings for {lang} to {output_path} as npy...")
    np.save(output_path, embeddings)

def main():
    # Load vocabs
    vocab_cn_list = load_vocab(VOCAB_CN)
    vocab_en_list = load_vocab(VOCAB_EN)
    
    # 1. Word2Vec
    train_and_save_word2vec(CORPUS_CN, vocab_cn_list, OUTPUT_W2V_CN, "CN")
    train_and_save_word2vec(CORPUS_EN, vocab_en_list, OUTPUT_W2V_EN, "EN")
    
    # 2. BGEM3
    generate_and_save_bgem3(vocab_cn_list, OUTPUT_BGEM3_CN, "CN")
    generate_and_save_bgem3(vocab_en_list, OUTPUT_BGEM3_EN, "EN")
    
    print("Done!")

if __name__ == "__main__":
    main()
